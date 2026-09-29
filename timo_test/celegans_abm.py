#!/usr/bin/env python3
"""
Minimal agent-based model of C. elegans (npr-1) pattern formation.
Based on the feedback loop in Demir et al. 2020 (eLife 9:e52781):

  1. Worms are run-and-tumble walkers; speed depends on LOCAL O2:  v = V(O)
  2. Worms consume O2:  dO/dt = D_O lap(O) + f (O_am - O) - k_c W
  3. O2 relaxes to ambient by lateral diffusion and air penetration.

No steric interaction, no alignment, no explicit gradient steering.
Accumulation arises purely from slowing down (steady state rho ~ 1/V).

Units: length um, time s, O2 in percent (internally), worm density in worms/m^2
for the consumption term only (so k_c = 7.3e-10 as in the paper).

Examples
  python celegans_abm.py                          # single run, 21% O2
  python celegans_abm.py --o2 1                   # random regime
  python celegans_abm.py --sweep 1 7 16 21        # compare ambient O2 levels
  python celegans_abm.py --ablate flatV           # ablation: V independent of O2
  python celegans_abm.py --ablate noconsume
  python celegans_abm.py --tau 0.1                # vary tumble rate
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
def default_params():
    return dict(
        L=3000.0,        # box side (um)
        dx=50.0,         # O2 grid spacing (um)
        dt=0.1,          # time step (s)  (stability: dt < dx^2/(4 D_O) = 0.31 s)
        T=600.0,         # total time (s)
        density=30.0,    # worms / mm^2
        Vmin=20.0,       # slow-plateau speed (um/s)
        rise_c=18.0,     # O2 (%) where speed rises again
        rise_w=1.5,      # width of that rise (%)
        body_len=0.0,    # >0: track a body of this length (um) trailing each head (visual only)
        n_seg=10,        # body segments
        frame_dt=0.0,    # >0: store a frame (pos, O) every frame_dt seconds for animation
        sigma=100.0,     # consumption footprint of a worm (um, ~ body scale)
        o2=21.0,         # ambient O2 (%)
        tau=0.5,         # tumble rate (1/s)
        D_O=2000.0,      # O2 diffusion (um^2/s)  = 2e-5 cm^2/s
        f=0.65,          # air penetration rate (1/s)
        k_c=7.3e-10,     # consumption per (worm/m^2) per s, in O2 FRACTION units
        seed=0,
        ablate="none",   # none | flatV | noconsume | noRise
    )


def V_npr1(O, Vmin=20.0, rise_c=18.0, rise_w=1.5):
    """Speed (um/s) vs O2 (%), fit by eye to Fig. 2b (npr-1 on food).
    ~170 at 1%, ~20 between 5 and 15%, ~150 at 21%."""
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    return Vmin + 160.0 * (1.0 - sig((O - 3.0) / 0.8)) + 150.0 * sig((O - rise_c) / rise_w)


def V_noRise(O):
    """Ablation: only the low-O2 branch (speed never rises at high O2)."""
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    return 20.0 + 160.0 * (1.0 - sig((O - 3.0) / 0.8))


def make_V(ablate, p=None):
    if ablate == "flatV":
        return lambda O: np.full_like(np.asarray(O, float), 60.0)
    if ablate == "noRise":
        return V_noRise
    if p is not None:
        return lambda O: V_npr1(O, p["Vmin"], p["rise_c"], p["rise_w"])
    return V_npr1


# ----------------------------------------------------------------------------
# Grid helpers (cell-centred grid, bilinear / cloud-in-cell)
# ----------------------------------------------------------------------------
def cic_weights(pos, dx, N):
    g = pos / dx - 0.5
    i0 = np.floor(g).astype(int)
    fx = g[:, 0] - i0[:, 0]
    fy = g[:, 1] - i0[:, 1]
    ix = [np.clip(i0[:, 0], 0, N - 1), np.clip(i0[:, 0] + 1, 0, N - 1)]
    iy = [np.clip(i0[:, 1], 0, N - 1), np.clip(i0[:, 1] + 1, 0, N - 1)]
    wx = [1 - fx, fx]
    wy = [1 - fy, fy]
    return ix, iy, wx, wy


def deposit(pos, dx, N):
    """Number of worms per cell (CIC)."""
    ix, iy, wx, wy = cic_weights(pos, dx, N)
    n = np.zeros(N * N)
    for a in range(2):
        for b in range(2):
            n += np.bincount(ix[a] * N + iy[b], weights=wx[a] * wy[b], minlength=N * N)
    return n.reshape(N, N)


def interpolate(field, pos, dx, N):
    ix, iy, wx, wy = cic_weights(pos, dx, N)
    out = 0.0
    for a in range(2):
        for b in range(2):
            out = out + wx[a] * wy[b] * field[ix[a], iy[b]]
    return out


def laplacian(O, dx):
    P = np.pad(O, 1, mode="edge")  # no-flux walls
    return (P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] - 4 * O) / dx**2


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(p, snapshots=(), verbose=True):
    rng = np.random.default_rng(p["seed"])
    L, dx, dt = p["L"], p["dx"], p["dt"]
    N = int(round(L / dx))
    n_agents = int(p["density"] * (L / 1000.0) ** 2)
    V = make_V(p["ablate"], p)
    kc = 0.0 if p["ablate"] == "noconsume" else p["k_c"]

    pos = rng.uniform(0, L, (n_agents, 2))
    theta = rng.uniform(0, 2 * np.pi, n_agents)
    O = np.full((N, N), p["o2"])  # percent
    M = int(p["n_seg"]); seg = p["body_len"] / M
    body = None
    if p["body_len"] > 0:   # straight body trailing the head; purely cosmetic
        e0 = np.stack([np.cos(theta), np.sin(theta)], 1)[:, None, :]
        body = np.clip(pos[:, None, :] - np.arange(M + 1)[None, :, None] * seg * e0, 0, L)
    frame_bodies, snap_bodies = [], {}
    if verbose:
        print(f"agents={n_agents}, grid={N}x{N}, steps={int(p['T']/dt)}")

    steps = int(round(p["T"] / dt))
    snap_steps = {int(round(t / dt)): t for t in snapshots}
    snaps = {}
    frames = []
    fstep = int(round(p["frame_dt"] / dt)) if p["frame_dt"] > 0 else 0
    hist = []
    for it in range(steps + 1):
        if fstep and it % fstep == 0:
            frames.append((it * dt, pos.copy(), O.copy()))
            if body is not None: frame_bodies.append(body.copy())
        if it in snap_steps:
            snaps[snap_steps[it]] = (pos.copy(), O.copy())
            if body is not None: snap_bodies[snap_steps[it]] = body.copy()
        if it == steps:
            break

        # --- oxygen field ----------------------------------------------
        n = deposit(pos, dx, N)                       # worms per cell
        if p["sigma"] > 0:
            n = gaussian_filter(n, p["sigma"] / dx, mode="nearest")
        W = n / (dx * 1e-6) ** 2                      # worms / m^2
        sink = kc * W * 100.0                         # % per s (k_c is per fraction)
        O = O + dt * (p["D_O"] * laplacian(O, dx) + p["f"] * (p["o2"] - O) - sink)
        O = np.clip(O, 0.0, 100.0)

        # --- agents: tumble + move at V(O(x)) --------------------------
        tumble = rng.random(n_agents) < p["tau"] * dt
        theta[tumble] = rng.uniform(0, 2 * np.pi, tumble.sum())
        v = V(interpolate(O, pos, dx, N))
        pos[:, 0] += v * np.cos(theta) * dt
        pos[:, 1] += v * np.sin(theta) * dt

        # reflecting walls
        for d in range(2):
            lo = pos[:, d] < 0
            hi = pos[:, d] > L
            pos[lo, d] = -pos[lo, d]
            pos[hi, d] = 2 * L - pos[hi, d]
            # heading flips on the normal component
            if d == 0:
                theta[lo | hi] = np.pi - theta[lo | hi]
            else:
                theta[lo | hi] = -theta[lo | hi]
        pos = np.clip(pos, 0, L)

        if body is not None:  # each body point follows the one ahead (rope constraint)
            body[:, 0] = pos
            for j in range(1, M + 1):
                d = body[:, j - 1] - body[:, j]
                dist = np.linalg.norm(d, axis=1, keepdims=True)
                body[:, j] += d * np.maximum(1 - seg / np.maximum(dist, 1e-9), 0)

        if it % int(round(10 / dt)) == 0:
            hist.append((it * dt, O.mean(), O.min(), v.mean()))
            if verbose and it % int(round(100 / dt)) == 0:
                print(f"t={it*dt:6.0f}s  <O2>={O.mean():5.2f}%  minO2={O.min():5.2f}%  <v>={v.mean():6.1f} um/s")
    return dict(pos=pos, O=O, snaps=snaps, frames=frames, frame_bodies=frame_bodies,
                snap_bodies=snap_bodies, body=body, hist=np.array(hist), N=N, params=p)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def density_field(pos, L, bins=60):
    H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins, range=[[0, L], [0, L]])
    return H


def structure_factor(pos, L, bins=60):
    """Radially averaged structure factor of the coarse-grained density."""
    H = density_field(pos, L, bins)
    d = H - H.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(d))) ** 2 / H.size
    c = bins // 2
    y, x = np.indices(F.shape)
    r = np.hypot(x - c, y - c).astype(int)
    S = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = 2 * np.pi * np.arange(len(S)) / L  # 1/um
    m = slice(1, c)
    return k[m], S[m]


def clustering_metrics(pos, L, bins=60):
    """Peak of S(k), and fraction of density variance (index of dispersion)."""
    k, S = structure_factor(pos, L, bins)
    H = density_field(pos, L, bins)
    disp = H.var() / max(H.mean(), 1e-12)  # =1 for Poisson (random)
    kp = k[np.argmax(S)]
    return dict(Speak=S.max(), k_peak=kp, wavelength_um=2 * np.pi / kp, dispersion=disp)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_single(res, out):
    p = res["params"]
    L = p["L"]
    times = sorted(res["snaps"].keys())
    fig, ax = plt.subplots(2, len(times), figsize=(3.2 * len(times), 6.4), squeeze=False)
    for j, t in enumerate(times):
        pos, O = res["snaps"][t]
        ax[0, j].scatter(pos[:, 0], pos[:, 1], s=3, c="k")
        ax[0, j].set_title(f"t = {t:.0f} s")
        ax[0, j].set_xlim(0, L); ax[0, j].set_ylim(0, L); ax[0, j].set_aspect("equal")
        im = ax[1, j].imshow(O.T, origin="lower", extent=[0, L, 0, L], cmap="viridis",
                             vmin=0, vmax=p["o2"] + 1e-9)
        ax[1, j].set_title("O2 (%)")
    fig.colorbar(im, ax=ax[1, :].tolist(), shrink=0.8)
    fig.suptitle(f"ambient O2={p['o2']}%, density={p['density']}/mm², ablate={p['ablate']}")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(results, out):
    n = len(results)
    fig, ax = plt.subplots(2, n, figsize=(3.4 * n, 7), squeeze=False)
    for j, (o2, res) in enumerate(results):
        L = res["params"]["L"]
        pos = res["pos"]
        ax[0, j].scatter(pos[:, 0], pos[:, 1], s=3, c="k")
        ax[0, j].set_xlim(0, L); ax[0, j].set_ylim(0, L); ax[0, j].set_aspect("equal")
        m = clustering_metrics(pos, L)
        ax[0, j].set_title(f"O2={o2}%  disp={m['dispersion']:.2f}")
        k, S = structure_factor(pos, L)
        ax[1, j].plot(k * 1e3, S); ax[1, j].set_xlabel("k (1/mm)"); ax[1, j].set_ylabel("S(k)")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    d = default_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=d["L"], help="box side (um)")
    ap.add_argument("--T", type=float, default=d["T"])
    ap.add_argument("--dt", type=float, default=d["dt"])
    ap.add_argument("--density", type=float, default=d["density"], help="worms/mm^2")
    ap.add_argument("--o2", type=float, default=d["o2"], help="ambient O2 (%%)")
    ap.add_argument("--tau", type=float, default=d["tau"], help="tumble rate (1/s)")
    ap.add_argument("--sigma", type=float, default=d["sigma"], help="worm footprint (um)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="none", choices=["none", "flatV", "noconsume", "noRise"])
    ap.add_argument("--sweep", type=float, nargs="+", help="list of ambient O2 values")
    ap.add_argument("--out", default="celegans_abm.png")
    a = ap.parse_args()

    p = d
    p.update(L=a.L, T=a.T, dt=a.dt, density=a.density, o2=a.o2, tau=a.tau, sigma=a.sigma, seed=a.seed, ablate=a.ablate)

    if a.sweep:
        results = []
        for o2 in a.sweep:
            q = dict(p, o2=o2)
            print(f"\n--- ambient O2 = {o2}% ---")
            res = simulate(q)
            print("metrics:", {k: round(v, 3) for k, v in clustering_metrics(res["pos"], q["L"]).items()})
            results.append((o2, res))
        plot_sweep(results, a.out)
    else:
        snaps = [0, a.T / 4, a.T / 2, a.T]
        res = simulate(p, snapshots=snaps)
        print("metrics:", {k: round(v, 3) for k, v in clustering_metrics(res["pos"], p["L"]).items()})
        plot_single(res, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# Worm rendering + side-by-side animation
# ----------------------------------------------------------------------------
def worm_segments(body, amp=30.0, seed=0):
    """(N, M+1, 2) body chains -> list of polylines with a light sinusoidal wiggle."""
    N, M1, _ = body.shape
    if amp <= 0:
        return list(body)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, (N, 1))
    j = np.arange(M1)[None, :]
    t = np.gradient(body, axis=1)
    t /= np.maximum(np.linalg.norm(t, axis=2, keepdims=True), 1e-9)
    nrm = np.stack([-t[..., 1], t[..., 0]], -1)
    off = amp * np.sin(2 * np.pi * 1.2 * j / (M1 - 1) + ph) * np.sin(np.pi * j / (M1 - 1)) ** 0.5
    return list(body + off[..., None] * nrm)


def draw_worms(ax, body, L, lw=1.4, amp=30.0, color="0.1"):
    """Draw worms as curved bodies with a red head dot. Returns (LineCollection, head scatter)."""
    from matplotlib.collections import LineCollection
    lc = LineCollection(worm_segments(body, amp), colors=color, linewidths=lw, capstyle="round")
    ax.add_collection(lc)
    hs = ax.scatter(body[:, 0, 0], body[:, 0, 1], s=3, c="crimson", zorder=3)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return lc, hs


def animate_runs(results, labels, show_o2=True, interval=80, lw=1.4, amp=30.0):
    """results: list of simulate() outputs run with frame_dt > 0.
    If they were run with body_len > 0, worms are drawn as bodies, else as points."""
    from matplotlib.animation import FuncAnimation
    n = len(results)
    L = results[0]["params"]["L"]
    rows = 2 if show_o2 else 1
    fig, ax = plt.subplots(rows, n, figsize=(3.6 * n, 3.7 * rows), squeeze=False)
    bodies = [len(r["frame_bodies"]) > 0 for r in results]
    arts, ims, titles = [], [], []
    for j, (res, lab) in enumerate(zip(results, labels)):
        t0, pos0, O0 = res["frames"][0]
        if bodies[j]:
            arts.append(draw_worms(ax[0, j], res["frame_bodies"][0], L, lw, amp))
        else:
            ax[0, j].set_xlim(0, L); ax[0, j].set_ylim(0, L); ax[0, j].set_aspect("equal")
            ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
            arts.append((ax[0, j].scatter(pos0[:, 0], pos0[:, 1], s=2, c="k"),))
        titles.append(ax[0, j].set_title(f"{lab}   t = 0 s"))
        if show_o2:
            ims.append(ax[1, j].imshow(O0.T, origin="lower", extent=[0, L, 0, L],
                                       vmin=0, vmax=res["params"]["o2"], cmap="viridis"))
            ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
    fig.tight_layout()

    def update(k):
        for j, res in enumerate(results):
            t, pos, O = res["frames"][k]
            if bodies[j]:
                b = res["frame_bodies"][k]
                arts[j][0].set_segments(worm_segments(b, amp))
                arts[j][1].set_offsets(b[:, 0])
            else:
                arts[j][0].set_offsets(pos)
            titles[j].set_text(f"{labels[j]}   t = {t:.0f} s")
            if show_o2:
                ims[j].set_data(O.T)
        return []

    anim = FuncAnimation(fig, update, frames=len(results[0]["frames"]), interval=interval, blit=False)
    plt.close(fig)
    return anim
