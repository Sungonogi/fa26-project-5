"""Notebook controls using ipywidgets and inline Matplotlib (no ipympl needed)."""
import asyncio
from dataclasses import replace
import html
from io import BytesIO

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from IPython.display import display

from model import Simulation, random_reference


class LiveSimulation:
    def __init__(self, parameters):
        self.base = parameters
        self.task = None
        self.controls = {
            "n": widgets.IntSlider(value=parameters.n, min=300, max=2000, step=100,
                                   description="Worms", continuous_update=False),
            "persistence": widgets.FloatSlider(value=parameters.persistence, min=2, max=20, step=1,
                                              description="Persistence", continuous_update=False),
            "seed": widgets.IntText(value=parameters.seed, description="Seed"),
        }
        if parameters.mode == "density":
            self.controls["slowing"] = widgets.FloatSlider(value=parameters.slowing, min=0, max=5, step=0.25,
                                                           description="Slowing", continuous_update=False)
        else:
            self.controls["uptake"] = widgets.FloatSlider(value=parameters.uptake, min=0, max=0.08, step=0.005,
                                                          readout_format=".3f", description="O₂ depletion", continuous_update=False)
            self.controls["steering"] = widgets.FloatSlider(value=parameters.steering, min=0, max=40, step=2.5,
                                                            description="O₂ turning", continuous_update=False)
        for control in self.controls.values():
            control.style.description_width = "100px"
            control.layout.width = "360px"
        self.start_button = widgets.Button(description="Start / resume", button_style="success")
        self.pause_button = widgets.Button(description="Pause")
        self.reset_button = widgets.Button(description="Apply & reset", button_style="info")
        self.step_button = widgets.Button(description="Advance 10 time units")
        self.status = widgets.HTML()
        self.playback = widgets.IntSlider(value=50, min=10, max=200, step=10,
                                           description="Steps / frame", continuous_update=False)
        self.output = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1200px"))
        self.start_button.on_click(self.start)
        self.pause_button.on_click(self.pause)
        self.reset_button.on_click(self.reset)
        self.step_button.on_click(self.advance)
        self._new_simulation(parameters)
        with plt.ioff():
            self.fig = plt.figure(figsize=(12, 6), layout="constrained")
            layout = self.fig.add_gridspec(2, 3, height_ratios=(2.4, 1))
            self.axes = [self.fig.add_subplot(layout[0, i]) for i in range(3)]
            self.metric_ax = self.fig.add_subplot(layout[1, :])
            self.lines = LineCollection([], colors="#283e45", linewidths=0.55, alpha=0.8)
            self.axes[0].add_collection(self.lines)
            self.axes[0].set_facecolor("#f3f1e9")
            self.heads = self.axes[0].scatter([], [], s=1, c="#172c33")
            extent = (0, parameters.size, 0, parameters.size)
            self.density_image = self.axes[1].imshow(self.sim.rho, origin="lower", extent=extent,
                                                    vmin=0, vmax=6, cmap="magma")
            self.fig.colorbar(self.density_image, ax=self.axes[1], label="worms / body length²", shrink=0.8)
            for ax in self.axes[:2]:
                ax.set(xlim=(0, parameters.size), ylim=(0, parameters.size), aspect="equal",
                       xlabel="body lengths", ylabel="body lengths")
            self.axes[0].set_title("Heads + visual bodies")
            self.axes[1].set_title("Smoothed population density")
            if parameters.mode == "oxygen":
                self.oxygen_image = self.axes[2].imshow(self.sim.oxygen, origin="lower", extent=extent,
                                                       vmin=0, vmax=1, cmap="viridis")
                self.fig.colorbar(self.oxygen_image, ax=self.axes[2], label="oxygen / ambient", shrink=0.8)
                self.axes[2].set(title="Local oxygen (preferred = 0.40)", xlabel="body lengths")
            else:
                self.response, = self.axes[2].plot([], [], color="#168c87", lw=2)
                self.axes[2].set(xlim=(0, 6), ylim=(0, 1.05), xlabel="local density", ylabel="speed / free speed",
                                 title="Density-dependent speed")
            self.cv_line, = self.metric_ax.plot([], [], color="#bf4e30", label="Density CV (aggregation)")
            self.mass_line, = self.metric_ax.plot([], [], color="#168c87", label="Mass in largest dense region")
            self.null_line = self.metric_ax.axhline(self.reference, color="gray", linestyle="--", label="Random-placement CV")
            self.metric_ax.set(xlabel="simulation time", ylabel="dimensionless")
            self.metric_ax.legend(loc="upper left", ncols=3, fontsize=8)
            self.metric_ax.grid(alpha=0.15)
        plt.close(self.fig)
        display(widgets.VBox([
            widgets.HTML("<b>Change parameters, then Apply &amp; reset.</b> Start resumes the current run. "
                         "Time and lengths are model units; Pause before running batch analysis."),
            widgets.HBox([widgets.VBox(list(self.controls.values())[:3]),
                          widgets.VBox(list(self.controls.values())[3:])]),
            widgets.HBox([self.start_button, self.pause_button, self.reset_button, self.step_button]),
            self.playback, self.status, self.output]))
        self.refresh()

    def _new_simulation(self, parameters):
        self.sim = Simulation(parameters)
        self.reference = random_reference(parameters)
        self.history = [self.sim.metrics()]

    def _draw(self):
        item = self.history[-1]
        self.lines.set_segments(self.sim.segments())
        self.heads.set_offsets(self.sim.xy)
        self.density_image.set_data(self.sim.rho)
        if self.sim.p.mode == "oxygen":
            self.oxygen_image.set_data(self.sim.oxygen)
        else:
            x = np.linspace(0, 6, 200)
            self.response.set_data(x, self.sim.speed_fraction(x))
        times = [x["time"] for x in self.history]
        self.cv_line.set_data(times, [x["cv"] for x in self.history])
        self.mass_line.set_data(times, [x["largest"] for x in self.history])
        self.null_line.set_ydata([self.reference, self.reference])
        self.metric_ax.set_xlim(0, max(20, times[-1]))
        self.metric_ax.set_ylim(0, max(1.2, 1.15 * max(x["cv"] for x in self.history)))
        self.fig.suptitle(f"{'Without oxygen' if self.sim.p.mode == 'density' else 'With oxygen'} · t = {self.sim.time:.1f}")
        self.status.value = (f"t = <b>{self.sim.time:.1f}</b> · CV = {item['cv']:.2f} "
                             f"(random ≈ {self.reference:.2f}) · largest dense region = {item['largest']:.0%} "
                             f"· max density = {self.sim.rho.max():.1f} (color scale capped at 6)")

    def refresh(self):
        self._draw()
        # Updating an Image widget also works from a background asyncio task,
        # without relying on the currently executing cell's output context.
        with BytesIO() as image:
            self.fig.savefig(image, format="png", dpi=100)
            self.output.value = image.getvalue()

    async def _run(self):
        try:
            while True:
                self.sim.step(self.playback.value)
                self.history.append(self.sim.metrics())
                self.refresh()
                await asyncio.sleep(0.06)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.status.value = "Stopped: " + html.escape(str(exc))
            raise

    def start(self, _=None):
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self._run())

    def pause(self, _=None):
        if self.task is not None:
            self.task.cancel()
            self.task = None

    def reset(self, _=None):
        self.pause()
        try:
            parameters = replace(self.base, **{key: control.value for key, control in self.controls.items()})
            self._new_simulation(parameters)
            self.refresh()
        except ValueError as exc:
            self.status.value = "Cannot reset: " + html.escape(str(exc))

    def advance(self, _=None):
        self.pause()
        self.sim.step(round(10/self.sim.p.dt))
        self.history.append(self.sim.metrics())
        self.refresh()

    def close(self):
        self.pause()
        self.output.close()


def plot_experiment(records, snapshots, parameters):
    times = sorted(snapshots)
    fig, axes = plt.subplots(1, len(times), figsize=(3.2*len(times), 3.2), layout="constrained", squeeze=False)
    maximum = max(6, max(x.max() for x in snapshots.values()))
    for ax, t in zip(axes[0], times):
        im = ax.imshow(snapshots[t], origin="lower", vmin=0, vmax=maximum,
                       extent=(0, parameters.size, 0, parameters.size), cmap="magma")
        ax.set(title=f"t = {t:g}", xlabel="body lengths")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="worms / body length²", shrink=0.8)
    display(fig)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    t = [r["time"] for r in records]
    axes[0].plot(t, [r["cv"] for r in records], label="density CV")
    axes[0].axhline(random_reference(parameters), color="gray", ls="--", label="random CV")
    axes[0].plot(t, [r["largest"] for r in records], label="largest dense-region mass")
    axes[0].legend(fontsize=8)
    axes[1].plot(t[1:], [r["density_change"] for r in records[1:]])
    axes[1].set_title("Relative density-map change between samples")
    for ax in axes:
        ax.set_xlabel("simulation time")
        ax.grid(alpha=0.2)
    display(fig)
    plt.close(fig)
