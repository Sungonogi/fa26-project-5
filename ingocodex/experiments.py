"""Reproducible long-run comparison; run with python3 experiments.py."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import Parameters, random_reference, run_experiment


def comparison(output="results", duration=3000, seeds=(4, 17, 29), base=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = Parameters() if base is None else base
    summary, all_records, runs = [], [], {}
    for mode in ("density", "oxygen"):
        for seed in seeds:
            p = replace(base, mode=mode, seed=seed)
            records, snapshots, sim = run_experiment(
                p, duration=duration, sample_every=50,
                snapshot_times=(duration/6, duration/2, duration))
            late = [r for r in records if r["time"] >= duration*2/3]
            a, b = snapshots[duration/2], snapshots[duration]
            summary.append({"mode": mode, "seed": seed, "duration": duration,
                            "random_cv": random_reference(p),
                            "late_cv_mean": np.mean([r["cv"] for r in late]),
                            "late_cv_sd": np.std([r["cv"] for r in late]),
                            "late_largest_mean": np.mean([r["largest"] for r in late]),
                            "late_map_change_50": np.mean([r["density_change"] for r in late]),
                            "half_to_end_map_correlation": np.corrcoef(a.ravel(), b.ravel())[0, 1],
                            "final_oxygen_min": sim.oxygen.min(),
                            "final_oxygen_max": sim.oxygen.max()})
            all_records.extend({"mode": mode, "seed": seed, **r} for r in records)
            runs[(mode, seed)] = (records, snapshots)
            print(f"{mode}, seed {seed}: late CV {summary[-1]['late_cv_mean']:.2f}; "
                  f"largest region {summary[-1]['late_largest_mean']:.0%}", flush=True)
    for filename, rows in (("summary.csv", summary), ("timeseries.csv", all_records)):
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (output/filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    (output/"parameters.json").write_text(json.dumps({"base": asdict(base), "seeds": seeds,
                                                    "duration": duration, "sample_every": 50}, indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.3), layout="constrained")
    maximum = max(x.max() for mode in ("density", "oxygen")
                  for x in runs[(mode, seeds[0])][1].values())
    for row, mode in enumerate(("density", "oxygen")):
        for ax, (t, rho) in zip(axes[row], sorted(runs[(mode, seeds[0])][1].items())):
            im = ax.imshow(rho, origin="lower", cmap="magma", vmin=0, vmax=maximum,
                           extent=(0, base.size, 0, base.size))
            ax.set_title(f"{'No oxygen' if mode == 'density' else 'Oxygen'} · t = {t:g}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), label="worms / body length²", shrink=0.7)
    fig.suptitle(f"Same random start, seed {seeds[0]} · identical density color scale")
    fig.savefig(output/"comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for mode, color in (("density", "#bd5635"), ("oxygen", "#16877f")):
        for i, seed in enumerate(seeds):
            records, _ = runs[(mode, seed)]
            for ax, key in zip(axes, ("cv", "largest", "density_change")):
                data = records if key != "density_change" else records[1:]
                ax.plot([r["time"] for r in data], [r[key] for r in data], color=color,
                        alpha=0.75, lw=1, label=mode if i == 0 else None)
    axes[0].axhline(summary[0]["random_cv"], color="gray", linestyle="--", label="random placement")
    for ax, title in zip(axes, ("Density CV", "Mass in largest dense region", "Density-map change over 50 time units")):
        ax.set(title=title, xlabel="simulation time")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    fig.savefig(output/"long_run_metrics.png", dpi=150)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    comparison(Path(__file__).parent/"results")
