#!/usr/bin/env python3
"""
Agent-based model of C. elegans (npr-1) pattern formation with soft-core repulsion
and fully periodic boundary conditions.
Based on the oxygen-kinesis model described in model.pdf & Demir et al. 2020.
"""

import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
def default_params():
    return dict(
        L=3000.0,        # periodic box side (um)
        dx=50.0,         # O2 grid spacing (um)
        dt=0.1,          # time step (s)
        T=600.0,         # total time (s)
        density=30.0,    # worms / mm^2
        Vmin=20.0,       # slow-plateau speed (um/s)
        rise_c=18.0,     # O2 (%) where speed rises again
        rise_w=1.5,      # width of that rise (%)
        body_len=0.0,    # >0: track a body of this length (um) trailing each head
        n_seg=10,        # body segments
        frame_dt=0.0,    # >0: store frames for animation
        sigma=100.0,     # consumption footprint radius sigma (um)
        o2=21.0,         # ambient O2 (%)
        lam=0.5,         # tumble rate lambda (1/s)
        tau=0.5,         # alias for tumble rate lambda
        r_c=30.0,        # soft-core repulsion cutoff distance r_c (um)
        # --- extra coupling to the oxygen field / density (all OFF by default) ---
        chi=0.0,         # aerotaxis strength (s/%): tumble rate lowered when moving toward O_star
        O_star=8.5,      # preferred O2 (%), npr-1 aerotaxis target (7-10%)
        lam_max=20.0,    # cap on tumble-rate modulation factor (keeps lam_i in [0, lam_max*lam])
        rho_s=0.0,       # >0: density slowing, v = V(O)/(1+S(O)*(rho_loc/rho_s)^rho_n)
        rho_n=2.0,       # Hill exponent; must be >1 for density slowing alone to cause clustering
        rho_o2_c=15.0,   # O2 threshold (%) at which density slowing turns on
        rho_o2_w=1.5,    # O2 transition width (%); smaller = sharper threshold
        r_s=150.0,       # radius (um) over which a worm senses its neighbours
        D_O=2000.0,      # O2 diffusion (um^2/s)
        f=0.8,          # air penetration rate (1/s)
        k_c=7.3e-10,     # consumption parameter
        seed=0,
        ablate="none",   # none | flatV | noconsume | noRise
    )


def V_npr1(O, Vmin=20.0, rise_c=18.0, rise_w=1.5):
    """Speed law V(O) from Eq. (5) in model.pdf.""" #KEEP THIS EXACT SPEED FORMULA
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    return 20.0 + 250.0 * (1.0 - sig((O - 1.0) / 1)) + 80 * sig((O - 8.0) / 1.2)


def V_noRise(O):
    """Ablation: speed never rises at high O2."""
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


def density_o2_gate(O, O_c=15.0, O_w=1.5):
    """
    Oxygen gate for the density-dependent motility feedback.

    S(O) = 1 / (1 + exp(-(O-O_c)/O_w))

    Thus S ~ 0 at low O2, suppressing density slowing, and S ~ 1
    at high O2, recovering the original density-slowing rule.
    """
    O = np.asarray(O, dtype=float)
    if O_w <= 0:
        return (O >= O_c).astype(float)
    z = np.clip((O - O_c) / O_w, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------
# Periodic-boundary helpers
# ----------------------------------------------------------------------------
def wrap_positions(pos, L):
    """Wrap positions into [0, L) in both dimensions."""
    return np.mod(pos, L)


def minimum_image(disp, L):
    """Apply the minimum-image convention to displacement vectors."""
    return disp - L * np.rint(disp / L)


# ----------------------------------------------------------------------------
# Grid helpers (Cloud-In-Cell bilinear interpolation)
# ----------------------------------------------------------------------------
def cic_weights(pos, dx, N):
    """
    Periodic CIC weights.

    The grid is treated as a torus: indices that leave one side of the
    grid re-enter on the opposite side.
    """
    g = pos / dx - 0.5
    i0 = np.floor(g).astype(int)
    fx = g[:, 0] - i0[:, 0]
    fy = g[:, 1] - i0[:, 1]

    ix = [np.mod(i0[:, 0], N), np.mod(i0[:, 0] + 1, N)]
    iy = [np.mod(i0[:, 1], N), np.mod(i0[:, 1] + 1, N)]

    wx = [1 - fx, fx]
    wy = [1 - fy, fy]
    return ix, iy, wx, wy


def deposit(pos, dx, N):
    """Number of worms per grid cell (CIC interpolation)."""
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
    """Periodic 5-point Laplacian."""
    return (
        np.roll(O, -1, axis=0)
        + np.roll(O, 1, axis=0)
        + np.roll(O, -1, axis=1)
        + np.roll(O, 1, axis=1)
        - 4 * O
    ) / dx**2


def compute_softcore_forces(pos, r_c, L, tree=None):
    """
    Computes soft-core repulsion with periodic boundary conditions.

    u(r) = (r - r_c)^2
    f_ij = -grad(u) = 2 * (r_c - r) * r_hat, for r < r_c.

    The current model uses the same 0.20 force prefactor as the original code.
    """
    if r_c <= 0:
        return np.zeros_like(pos)

    if tree is None:
        tree = cKDTree(pos, boxsize=L)

    # Periodic pair search: particles near opposite sides of the box interact.
    pairs = tree.query_pairs(r_c, output_type="ndarray")
    forces = np.zeros_like(pos)

    if len(pairs) > 0:
        i, j = pairs[:, 0], pairs[:, 1]

        # Minimum-image displacement on the torus.
        disp = minimum_image(pos[j] - pos[i], L)
        dist = np.linalg.norm(disp, axis=1)

        valid = dist > 1e-6
        i, j, disp, dist = i[valid], j[valid], disp[valid], dist[valid]

        r_hat = disp / dist[:, None]
        force_mag = 0.20 * (r_c - dist)[:, None]
        f_ij = force_mag * r_hat

        np.add.at(forces, i, -f_ij)
        np.add.at(forces, j, f_ij)

    return forces



# ----------------------------------------------------------------------------
# Simulation Engine
# ----------------------------------------------------------------------------
def simulate(p, snapshots=(), verbose=True):
    rng = np.random.default_rng(p["seed"])
    L, dx, dt = p["L"], p["dx"], p["dt"]
    N = int(round(L / dx))
    n_agents = int(p["density"] * (L / 1000.0) ** 2)
    V = make_V(p["ablate"], p)
    kc = 0.0 if p["ablate"] == "noconsume" else p["k_c"]
    tumble_rate = p.get("lam", p.get("tau", 0.5))

    pos = rng.uniform(0, L, (n_agents, 2))
    pos = wrap_positions(pos, L)
    theta = rng.uniform(0, 2 * np.pi, n_agents)
    O = np.full((N, N), p["o2"])
    O_prev = interpolate(O, pos, dx, N)   # O2 sensed by each worm at the previous step

    M = int(p["n_seg"])
    seg = p["body_len"] / M if M > 0 else 0.0
    body = None
    if p["body_len"] > 0:
        e0 = np.stack([np.cos(theta), np.sin(theta)], 1)[:, None, :]
        body = wrap_positions(
            pos[:, None, :]
            - np.arange(M + 1)[None, :, None] * seg * e0,
            L,
        )

    frame_bodies, snap_bodies = [], {}
    if verbose:
        print(f"agents={n_agents}, grid={N}x{N}, steps={int(p['T']/dt)}")

    steps = int(round(p["T"] / dt))
    snap_steps = {int(round(t / dt)): t for t in snapshots}
    snaps, frames = {}, []
    fstep = int(round(p["frame_dt"] / dt)) if p["frame_dt"] > 0 else 0
    hist = []

    for it in range(steps + 1):
        if fstep and it % fstep == 0:
            frames.append((it * dt, pos.copy(), O.copy()))
            if body is not None:
                frame_bodies.append(body.copy())
        if it in snap_steps:
            snaps[snap_steps[it]] = (pos.copy(), O.copy())
            if body is not None:
                snap_bodies[snap_steps[it]] = body.copy()
        if it == steps:
            break

        # --- 1. Oxygen Field PDE with Gaussian Footprint G_sigma ---
        n_raw = deposit(pos, dx, N)
        if p["sigma"] > 0:
            # Gaussian consumption footprint is also periodic.
            n_smooth = gaussian_filter(n_raw, p["sigma"] / dx, mode="wrap")
        else:
            n_smooth = n_raw

        W = n_smooth / (dx * 1e-6) ** 2
        sink = kc * W * 100.0

        O = O + dt * (p["D_O"] * laplacian(O, dx) + p["f"] * (p["o2"] - O) - sink)
        O = np.clip(O, 0.0, 100.0)

        # --- 2. Microscopic Worm Dynamics ---
        O_here = interpolate(O, pos, dx, N)          # O2 at each worm's position
        # Toroidal neighbour structure for this step.
        tree = cKDTree(pos, boxsize=L)

        # (a) speed: oxygen kinesis plus oxygen-gated density slowing.
        #
        #     v = V(O) / [1 + S(O) * (rho_loc/rho_s)^rho_n]
        #
        #     S(O) is ~0 below rho_o2_c, so density slowing is suppressed
        #     at low O2.  S(O) is ~1 above rho_o2_c, recovering the
        #     original density-slowing mechanism at high O2.
        v = V(O_here)
        if p["rho_s"] > 0:
            cnt = tree.query_ball_point(pos, p["r_s"], return_length=True) - 1   # exclude self
            rho_loc = cnt / (np.pi * p["r_s"] ** 2) * 1e6                       # worms / mm^2

            S_o2 = density_o2_gate(
                O_here,
                p.get("rho_o2_c", 15.0),
                p.get("rho_o2_w", 1.5),
            )
            v = v / (1.0 + S_o2 * (rho_loc / p["rho_s"]) ** p["rho_n"])

        # (b) explicit aerotaxis: tumble less when the O2 change along the path moves
        #     the worm closer to O_star (temporal-gradient sensing, klinokinesis)
        lam_i = tumble_rate
        if p["chi"] > 0:
            dU = -(np.abs(O_here - p["O_star"]) - np.abs(O_prev - p["O_star"])) / dt  # %/s, >0 = improving
            lam_i = tumble_rate * np.clip(np.exp(-p["chi"] * dU), 0.0, p["lam_max"])
        O_prev = O_here

        tumble = rng.random(n_agents) < lam_i * dt
        theta[tumble] = rng.uniform(0, 2 * np.pi, tumble.sum())

        dr_propulsion = v[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1) * dt
        f_repulsion = compute_softcore_forces(
            pos, p.get("r_c", 30.0), L, tree
        )

        pos += dr_propulsion + f_repulsion * dt

        # Periodic boundary conditions: leaving one side re-enters
        # through the opposite side.  Heading is unchanged.
        pos = wrap_positions(pos, L)

        if body is not None:
            body[:, 0] = pos
            for j in range(1, M + 1):
                # Follow the previous segment using minimum-image geometry,
                # then wrap the segment back onto the periodic domain.
                d = minimum_image(body[:, j - 1] - body[:, j], L)
                dist = np.linalg.norm(d, axis=1, keepdims=True)
                body[:, j] = wrap_positions(
                    body[:, j] + d * np.maximum(1 - seg / np.maximum(dist, 1e-9), 0),
                    L,
                )

        if it % int(round(10 / dt)) == 0:
            hist.append((it * dt, O.mean(), O.min(), v.mean()))
            if verbose and it % int(round(100 / dt)) == 0:
                print(f"t={it*dt:6.0f}s  <O2>={O.mean():5.2f}%  minO2={O.min():5.2f}%  <v>={v.mean():6.1f} um/s")

    return dict(
        pos=pos, O=O, snaps=snaps, frames=frames, frame_bodies=frame_bodies,
        snap_bodies=snap_bodies, body=body, hist=np.array(hist), N=N, params=p
    )


# ----------------------------------------------------------------------------
# Animation Helpers
# ----------------------------------------------------------------------------
def worm_segments(body, L=None, amp=30.0, seed=0):
    """
    Generates smooth sinusoidal body curves.

    If L is supplied, neighboring points are first unwrapped using the
    minimum-image convention. This prevents a worm crossing x=0/L or y=0/L
    from being interpreted as a very long body segment.
    """
    N, M1, _ = body.shape

    if L is not None:
        # Unwrap each worm along its body using minimum-image displacements.
        unwrapped = np.empty_like(body)
        unwrapped[:, 0] = body[:, 0]
        for j in range(1, M1):
            d = minimum_image(body[:, j] - body[:, j - 1], L)
            unwrapped[:, j] = unwrapped[:, j - 1] + d
    else:
        unwrapped = body

    if amp <= 0:
        return list(unwrapped)

    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, (N, 1))
    j = np.arange(M1)[None, :]

    t = np.gradient(unwrapped, axis=1)
    t /= np.maximum(np.linalg.norm(t, axis=2, keepdims=True), 1e-9)
    nrm = np.stack([-t[..., 1], t[..., 0]], -1)

    off = (
        amp
        * np.sin(2 * np.pi * 1.2 * j / (M1 - 1) + ph)
        * np.sin(np.pi * j / (M1 - 1)) ** 0.5
    )
    return list(unwrapped + off[..., None] * nrm)


def periodic_line_segments(curve, L):
    """
    Split a continuous, unwrapped worm curve at periodic boundaries.

    Returns short line segments whose endpoints are wrapped into [0, L).
    This avoids matplotlib drawing a spurious line across the entire box.
    """
    curve = np.asarray(curve)
    if len(curve) < 2:
        return []

    segments = []

    for k in range(len(curve) - 1):
        p0 = curve[k].copy()
        p1 = curve[k + 1].copy()
        d = p1 - p0

        # The curve from worm_segments is already locally unwrapped.
        # Find all boundary crossings in x and y.
        ts = [0.0, 1.0]

        for axis in (0, 1):
            if abs(d[axis]) < 1e-12:
                continue

            # Boundaries x = mL / y = mL crossed by this segment.
            lo = min(p0[axis], p1[axis])
            hi = max(p0[axis], p1[axis])
            m0 = int(np.floor(lo / L)) + 1
            m1 = int(np.ceil(hi / L)) - 1

            for m in range(m0, m1 + 1):
                t = (m * L - p0[axis]) / d[axis]
                if 1e-10 < t < 1.0 - 1e-10:
                    ts.append(float(t))

        ts = sorted(set(ts))

        for a, b in zip(ts[:-1], ts[1:]):
            q0 = p0 + a * d
            q1 = p0 + b * d

            # Midpoint determines which periodic image this small piece belongs to.
            mid = 0.5 * (q0 + q1)
            image = np.floor(mid / L)
            q0 = q0 - image * L
            q1 = q1 - image * L

            segments.append(np.vstack([q0, q1]))

    return segments


def draw_worms(ax, body, L, lw=1.4, amp=30.0, color="0.1", head_s=1.0):
    from matplotlib.collections import LineCollection

    curves = worm_segments(body, L=L, amp=amp)

    # Split each worm at periodic boundaries before sending it to matplotlib.
    all_segments = []
    for curve in curves:
        all_segments.extend(periodic_line_segments(curve, L))

    lc = LineCollection(
        all_segments,
        colors=color,
        linewidths=lw,
        capstyle="round",
    )
    ax.add_collection(lc)

    # Heads are already stored in wrapped coordinates.
    hs = ax.scatter(
        body[:, 0, 0],
        body[:, 0, 1],
        s=head_s,
        c="crimson",
        linewidths=0,
        zorder=3,
    )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return lc, hs


def animate_runs(results, labels, show_o2=True, interval=80, lw=1.4, amp=30.0, head_s=0.5):
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
            arts.append(draw_worms(ax[0, j], res["frame_bodies"][0], L, lw, amp, head_s=head_s))
        else:
            ax[0, j].set_xlim(0, L)
            ax[0, j].set_ylim(0, L)
            ax[0, j].set_aspect("equal")
            ax[0, j].set_xticks([])
            ax[0, j].set_yticks([])
            arts.append((ax[0, j].scatter(pos0[:, 0], pos0[:, 1], s=2, c="k"),))
        titles.append(ax[0, j].set_title(f"{lab}   t = 0 s"))

        if show_o2:
            ims.append(
                ax[1, j].imshow(
                    O0.T, origin="lower", extent=[0, L, 0, L],
                    vmin=0, vmax=res["params"]["o2"], cmap="viridis"
                )
            )
            ax[1, j].set_xticks([])
            ax[1, j].set_yticks([])

    fig.tight_layout()

    def update(k):
        for j, res in enumerate(results):
            t, pos, O = res["frames"][k]
            if bodies[j]:
                b = res["frame_bodies"][k]
                curves = worm_segments(b, L=L, amp=amp)
                segs = []
                for curve in curves:
                    segs.extend(periodic_line_segments(curve, L))
                arts[j][0].set_segments(segs)
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