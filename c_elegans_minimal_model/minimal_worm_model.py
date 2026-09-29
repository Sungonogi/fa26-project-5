"""Minimal agent-based model for C. elegans aggregation.

The model compares two candidate mechanisms:
S: density-dependent slowing
T: finite-range taxis toward nearby animals

It runs a complete 2x2 ablation experiment: baseline, S, T, and ST.
The units are deliberately nondimensional. This is a mechanism study, not a
calibrated reproduction of a particular experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Parameters:
    n_agents: int = 40
    box_size: float = 8.0
    dt: float = 0.05
    steps: int = 3000
    sample_every: int = 20

    # Persistent active motion
    rotational_diffusion: float = 0.10
    speed_high: float = 1.0

    # Local sensing and density-dependent speed
    sensing_radius: float = 1.0
    speed_low: float = 0.08
    slowdown_steepness: float = 1.8

    # Turning toward the local centre of neighbouring agents
    taxis_strength: float = 0.9

    # Soft excluded-volume interaction
    body_diameter: float = 0.20
    repulsion_strength: float = 5.0

    # Two agents belong to the same cluster below this separation
    cluster_distance: float = 0.55


CONDITIONS = {
    "baseline": (False, False),
    "slowdown": (True, False),
    "taxis": (False, True),
    "slowdown+taxis": (True, True),
}


def pair_geometry(positions: np.ndarray, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return minimum-image displacement i->j and pairwise distance."""
    displacement = positions[None, :, :] - positions[:, None, :]
    displacement -= box_size * np.round(displacement / box_size)
    distance = np.linalg.norm(displacement, axis=2)
    np.fill_diagonal(distance, np.inf)
    return displacement, distance


def largest_cluster_fraction(distance: np.ndarray, threshold: float) -> float:
    """Fraction of agents in the largest distance-connected component."""
    n = distance.shape[0]
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    rows, cols = np.where(np.triu(distance < threshold, k=1))
    for i, j in zip(rows, cols):
        union(int(i), int(j))

    counts: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        counts[root] = counts.get(root, 0) + 1
    return max(counts.values()) / n


def simulate(
    params: Parameters,
    seed: int,
    use_slowdown: bool,
    use_taxis: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, params.box_size, size=(params.n_agents, 2))
    angles = rng.uniform(-np.pi, np.pi, size=params.n_agents)

    times: list[float] = []
    cluster_fraction: list[float] = []
    nearest_neighbor: list[float] = []
    frames: list[np.ndarray] = []

    for step in range(params.steps + 1):
        displacement, distance = pair_geometry(positions, params.box_size)

        if step % params.sample_every == 0:
            times.append(step * params.dt)
            cluster_fraction.append(
                largest_cluster_fraction(distance, params.cluster_distance)
            )
            nearest_neighbor.append(float(np.mean(np.min(distance, axis=1))))
            frames.append(positions.copy())

        if step == params.steps:
            break

        neighbours = distance < params.sensing_radius
        neighbour_count = neighbours.sum(axis=1)
        local_density = neighbour_count / (np.pi * params.sensing_radius**2)

        if use_slowdown:
            speed = params.speed_low + (params.speed_high - params.speed_low) * np.exp(
                -params.slowdown_steepness * local_density
            )
        else:
            speed = np.full(params.n_agents, params.speed_high)

        if use_taxis:
            # Each nearby neighbour contributes a unit vector. Dividing by the
            # count prevents the turning strength from growing without bound.
            unit_to_neighbour = np.divide(
                displacement,
                distance[:, :, None],
                out=np.zeros_like(displacement),
                where=np.isfinite(distance)[:, :, None],
            )
            taxis_vector = np.sum(unit_to_neighbour * neighbours[:, :, None], axis=1)
            taxis_vector /= np.maximum(neighbour_count[:, None], 1)
            desired_angle = np.arctan2(taxis_vector[:, 1], taxis_vector[:, 0])
            angular_error = np.arctan2(
                np.sin(desired_angle - angles), np.cos(desired_angle - angles)
            )
            angles += params.taxis_strength * angular_error * params.dt * (
                neighbour_count > 0
            )

        angles += np.sqrt(2.0 * params.rotational_diffusion * params.dt) * rng.normal(
            size=params.n_agents
        )
        heading = np.column_stack((np.cos(angles), np.sin(angles)))

        # Short-range repulsion prevents nonphysical collapse to one point.
        overlap = distance < params.body_diameter
        unit_to_neighbour = np.divide(
            displacement,
            distance[:, :, None],
            out=np.zeros_like(displacement),
            where=np.isfinite(distance)[:, :, None],
        )
        overlap_amount = np.where(
            overlap,
            (params.body_diameter - np.minimum(distance, params.body_diameter))
            / params.body_diameter,
            0.0,
        )
        repulsion = -np.sum(
            unit_to_neighbour * overlap_amount[:, :, None], axis=1
        )

        velocity = speed[:, None] * heading + params.repulsion_strength * repulsion
        positions = (positions + params.dt * velocity) % params.box_size

    return {
        "time": np.asarray(times),
        "cluster_fraction": np.asarray(cluster_fraction),
        "nearest_neighbor": np.asarray(nearest_neighbor),
        "frames": np.asarray(frames),
    }


def run_experiment(params: Parameters, seeds: int) -> dict[str, list[dict[str, np.ndarray]]]:
    results: dict[str, list[dict[str, np.ndarray]]] = {}
    for condition, (slowdown, taxis) in CONDITIONS.items():
        results[condition] = [
            simulate(params, seed, slowdown, taxis) for seed in range(seeds)
        ]
    return results


def save_summary_csv(
    results: dict[str, list[dict[str, np.ndarray]]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "condition",
                "seed",
                "late_mean_largest_cluster_fraction",
                "late_mean_nearest_neighbor_distance",
            ]
        )
        for condition, runs in results.items():
            for seed, run in enumerate(runs):
                late = slice(int(0.8 * len(run["time"])), None)
                writer.writerow(
                    [
                        condition,
                        seed,
                        float(np.mean(run["cluster_fraction"][late])),
                        float(np.mean(run["nearest_neighbor"][late])),
                    ]
                )


def plot_results(
    results: dict[str, list[dict[str, np.ndarray]]],
    params: Parameters,
    output_path: Path,
) -> None:
    labels = list(CONDITIONS)
    colors = ["#777777", "#2878B5", "#D95F02", "#6A3D9A"]
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)

    for index, (condition, color) in enumerate(zip(labels, colors)):
        ax = fig.add_subplot(grid[0, index])
        final_positions = results[condition][0]["frames"][-1]
        ax.scatter(
            final_positions[:, 0],
            final_positions[:, 1],
            s=34,
            c=color,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.set(
            title=condition,
            xlim=(0, params.box_size),
            ylim=(0, params.box_size),
            aspect="equal",
            xlabel="x",
            ylabel="y" if index == 0 else "",
        )

    ax_cluster = fig.add_subplot(grid[1, :2])
    ax_neighbour = fig.add_subplot(grid[1, 2:])
    for condition, color in zip(labels, colors):
        runs = results[condition]
        time = runs[0]["time"]
        cluster_values = np.stack([run["cluster_fraction"] for run in runs])
        neighbour_values = np.stack([run["nearest_neighbor"] for run in runs])

        mean_cluster = cluster_values.mean(axis=0)
        sd_cluster = cluster_values.std(axis=0)
        mean_neighbour = neighbour_values.mean(axis=0)
        sd_neighbour = neighbour_values.std(axis=0)

        ax_cluster.plot(time, mean_cluster, color=color, label=condition)
        ax_cluster.fill_between(
            time,
            mean_cluster - sd_cluster,
            mean_cluster + sd_cluster,
            color=color,
            alpha=0.16,
        )
        ax_neighbour.plot(time, mean_neighbour, color=color, label=condition)
        ax_neighbour.fill_between(
            time,
            mean_neighbour - sd_neighbour,
            mean_neighbour + sd_neighbour,
            color=color,
            alpha=0.16,
        )

    ax_cluster.set(
        xlabel="time (model units)",
        ylabel="largest cluster fraction",
        ylim=(0, 1.02),
    )
    ax_neighbour.set(
        xlabel="time (model units)",
        ylabel="mean nearest-neighbour distance",
    )
    ax_cluster.legend(frameon=False, ncol=2)
    ax_neighbour.legend(frameon=False, ncol=2)
    fig.suptitle("Minimal C. elegans aggregation model: 2 x 2 ablation", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8, help="replicates per condition")
    parser.add_argument("--steps", type=int, default=3000, help="simulation steps")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="result directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = Parameters(steps=args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(params, args.seeds)
    save_summary_csv(results, args.output_dir / "summary.csv")
    plot_results(results, params, args.output_dir / "aggregation_comparison.png")
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
