"""Independent minimal worm aggregation models; lengths in body lengths.

Worms are point agents. A trailing polyline is display-only. No attraction,
alignment, hard-core collisions, bacteria agents, or fitted biological units.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter, label


@dataclass(frozen=True)
class Parameters:
    mode: str = "density"
    n: int = 1000
    seed: int = 4
    size: float = 32.0
    grid: int = 64
    dt: float = 0.1
    speed: float = 1.0
    persistence: float = 8.0
    sensing: float = 1.0
    density_scale: float = 1.0
    min_speed: float = 0.01
    slowing: float = 1.5
    diffusion: float = 1.0
    replenishment: float = 0.08
    uptake: float = 0.035
    preferred_oxygen: float = 0.4
    oxygen_width: float = 0.25
    steering: float = 15.0
    body_length: float = 1.0
    body_points: int = 9

    def __post_init__(self):
        if self.mode not in ("density", "oxygen"):
            raise ValueError("mode must be density or oxygen")
        if self.n < 1 or self.grid < 8 or self.body_points < 2:
            raise ValueError("n >= 1, grid >= 8, body_points >= 2 required")
        for key in ("size", "dt", "speed", "persistence", "sensing",
                    "density_scale", "oxygen_width", "body_length"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < self.min_speed <= 1:
            raise ValueError("min_speed must be in (0, 1]")
        if not 0 <= self.preferred_oxygen <= 1:
            raise ValueError("preferred_oxygen must be in [0, 1]")
        for key in ("slowing", "diffusion", "replenishment", "uptake", "steering"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.dt > 0.25 * self.persistence:
            raise ValueError("Reduce dt relative to persistence")
        if self.speed * self.dt > min(self.sensing, self.size / self.grid) / 2:
            raise ValueError("Reduce dt: a step should resolve the sensing/grid scale")


class Simulation:
    def __init__(self, p=Parameters(), trails=True):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.xy = self.rng.uniform(0, p.size, (p.n, 2))
        self.theta = self.rng.uniform(-np.pi, np.pi, p.n)
        self.time = 0.0
        self.steps = 0
        self.dx = p.size / p.grid
        self.oxygen = np.ones((p.grid, p.grid))
        # Exact-in-time diffusion for the discrete periodic grid Laplacian.
        k = np.arange(p.grid)
        eigenvalues = -4 * (np.sin(np.pi * k / p.grid) ** 2)
        self.diffusion_factor = np.exp(
            p.diffusion * p.dt * (eigenvalues[:, None] + eigenvalues[None, :]) / self.dx**2)
        self.trails = trails
        if trails:
            direction = np.column_stack((np.cos(self.theta), np.sin(self.theta)))
            distances = np.linspace(0, p.body_length, p.body_points)
            self.body = self.xy[:, None, :] - direction[:, None, :] * distances[None, :, None]
        self.rho = self.density()

    def _weights(self):
        coordinates = self.xy / self.dx
        low = np.floor(coordinates).astype(int)
        fraction = coordinates - low
        indices, weights = [], []
        for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
            indices.append(((low[:, 1] + b) % self.p.grid,
                            (low[:, 0] + a) % self.p.grid))
            weights.append((fraction[:, 0] if a else 1 - fraction[:, 0]) *
                           (fraction[:, 1] if b else 1 - fraction[:, 1]))
        return indices, weights

    def density(self):
        indices, weights = self._weights()
        flat = np.zeros(self.p.grid**2)
        for (iy, ix), w in zip(indices, weights):
            flat += np.bincount(iy * self.p.grid + ix, weights=w,
                                minlength=self.p.grid**2)
        # Gaussian kernel has unit integral; total integrated density = n.
        return gaussian_filter(flat.reshape(self.p.grid, self.p.grid) / self.dx**2,
                               self.p.sensing / self.dx, mode="wrap")

    def sample(self, field):
        indices, weights = self._weights()
        return sum(w * field[index] for index, w in zip(indices, weights))

    def speed_fraction(self, value):
        p = self.p
        if p.mode == "density":
            response = np.exp(-p.slowing * value / p.density_scale)
        else:
            response = 1 - np.exp(-((value - p.preferred_oxygen) / p.oxygen_width)**2)
        return p.min_speed + (1 - p.min_speed) * response

    def _advance_oxygen(self):
        p = self.p
        # Lie splitting: exact discrete diffusion, then exact local reaction.
        diffused = np.fft.irfft2(np.fft.rfft2(self.oxygen) *
                                self.diffusion_factor[:, :p.grid // 2 + 1],
                                s=self.oxygen.shape)
        rate = p.replenishment + p.uptake * self.rho
        equilibrium = np.divide(p.replenishment, rate,
                                out=np.zeros_like(rate), where=rate > 0)
        self.oxygen = equilibrium + (diffused - equilibrium) * np.exp(-rate * p.dt)

    def step(self, count=1):
        p = self.p
        for _ in range(count):
            self.rho = self.density()
            turn = 0.0
            if p.mode == "oxygen":
                self._advance_oxygen()
                value = self.sample(self.oxygen)
                gx = (np.roll(self.oxygen, -1, 1) - np.roll(self.oxygen, 1, 1)) / (2*self.dx)
                gy = (np.roll(self.oxygen, -1, 0) - np.roll(self.oxygen, 1, 0)) / (2*self.dx)
                # Descend the mismatch potential (o-o*)^2/2 by turning.
                fx = -(value - p.preferred_oxygen) * self.sample(gx)
                fy = -(value - p.preferred_oxygen) * self.sample(gy)
                turn = p.steering * (-np.sin(self.theta) * fx + np.cos(self.theta) * fy)
            else:
                value = self.sample(self.rho)
            self.theta += turn * p.dt + np.sqrt(2 * p.dt / p.persistence) * self.rng.normal(size=p.n)
            self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi
            velocity = p.speed * self.speed_fraction(value)
            displacement = velocity[:, None] * p.dt * np.column_stack((np.cos(self.theta), np.sin(self.theta)))
            self.xy = (self.xy + displacement) % p.size
            if self.trails:
                # Resample each path to fixed spatial length, not fixed time.
                self.body[:, 0] += displacement
                spacing = p.body_length / (p.body_points - 1)
                for j in range(1, p.body_points):
                    delta = self.body[:, j] - self.body[:, j-1]
                    distance = np.linalg.norm(delta, axis=1)
                    self.body[:, j] = self.body[:, j-1] + delta * (spacing / np.maximum(distance, 1e-12))[:, None]
                self.body -= np.floor(self.body[:, :1] / p.size) * p.size
            self.steps += 1
        self.time = self.steps * p.dt
        self.rho = self.density()

    def segments(self):
        if not self.trails:
            return np.empty((0, 2, 2))
        wrapped = self.body % self.p.size
        segments = np.stack((wrapped[:, :-1], wrapped[:, 1:]), axis=2).reshape(-1, 2, 2)
        # Hide crossing segments instead of drawing lines across the arena.
        return segments[np.all(np.abs(segments[:, 1] - segments[:, 0]) < self.p.size/2, axis=1)]

    def metrics(self):
        rho = self.rho
        threshold = 1.5 * self.p.n / self.p.size**2
        labels, number = label(rho > threshold)  # four-neighbor connectivity
        parent = np.arange(number + 1)

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for left, right in ((labels[:, 0], labels[:, -1]), (labels[0], labels[-1])):
            for a, b in zip(left, right):
                if a and b:
                    parent[root(a)] = root(b)
        mapping = np.array([root(a) for a in range(number+1)])
        masses = np.bincount(mapping[labels].ravel(), weights=rho.ravel(), minlength=number+1)
        masses[0] = 0
        return {"time": self.time, "cv": float(rho.std()/rho.mean()),
                "largest": float(masses.max()/rho.sum()),
                "oxygen_min": float(self.oxygen.min()), "oxygen_max": float(self.oxygen.max())}


def random_reference(p, samples=12):
    """Finite-N null, measured at the identical spatial smoothing scale."""
    return float(np.mean([Simulation(replace(p, seed=80000+i), trails=False).metrics()["cv"]
                          for i in range(samples)]))


def run_experiment(p, duration=1000.0, sample_every=10.0, snapshot_times=()):
    """Headless run for analysis. No visual body work or display overhead."""
    simulation = Simulation(p, trails=False)
    records = [simulation.metrics()]
    snapshots = {0.0: simulation.rho.copy()}
    previous = simulation.rho.copy()
    nsteps = int(round(duration / p.dt))
    stride = max(1, int(round(sample_every / p.dt)))
    requested = {int(round(t/p.dt)) for t in snapshot_times if 0 < t <= duration}
    stops = sorted(set(range(stride, nsteps+1, stride)) | requested | {nsteps})
    for stop in stops:
        simulation.step(stop - simulation.steps)
        item = simulation.metrics()
        item["density_change"] = float(np.linalg.norm(simulation.rho-previous) /
                                       max(np.linalg.norm(previous), 1e-12))
        item["change_interval"] = item["time"] - records[-1]["time"]
        previous = simulation.rho.copy()
        records.append(item)
        if stop in requested or stop == nsteps:
            snapshots[simulation.time] = simulation.rho.copy()
    return records, snapshots, simulation
