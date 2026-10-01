# Minimal individual-based C. elegans aggregation

Open **aggregation.ipynb** in JupyterLab or VS Code with a Python kernel. Run the setup and either live section, then press **Start / resume**. Parameter changes take effect with **Apply & reset**. The notebook uses ordinary inline Matplotlib and ipywidgets; ipympl is not required. Static notebook previews cannot run the controls.

Install the dependencies in your notebook environment if needed:

```sh
python3 -m pip install -r requirements.txt
```

The notebook has two main sections:

1. **No oxygen:** persistent point agents slow down with increasing local population density.
2. **Oxygen:** the same point agents deplete a replenishing, diffusing oxygen grid and change speed and turning according to an intermediate oxygen preference. Direct density-dependent slowing is absent.

Bodies are fixed-length visual trails only. Periodic boundaries avoid wall-driven accumulation. No physical collisions, explicit attraction, alignment, food depletion, bacteria agents, or fitted biological time scale are included. Density is smoothed from agents, never evolved as a continuum worm population.

## Supplied results

Three seeds (4, 17, 29), 1,000 worms, 3,000 model time units each:

| Condition | Mean density CV in final third | Random-placement reference |
|---|---:|---:|
| No oxygen | 1.55–1.60 | 0.28 |
| Oxygen | 2.30 | 0.28 |

Both show sustained accumulation. Without oxygen, multiple clusters persist but continue slowly coarsening/rearranging. Oxygen produces a persistent ring-like aggregate with about 96% of smoothed population mass in the largest above-threshold region. Neither default reproduces a many-cell network. These runs demonstrate finite-time persistence, not mathematical or biological proof of permanent stability.

The minimum speed is 1% of the free speed in both models, a deliberately strong retention assumption. Try larger values through `replace(COMMON, min_speed=0.04)` to test sensitivity; stronger slowing is not automatically faster pattern formation. The initial development checks included these larger floors, but the reference figures use only the parameters in `results/parameters.json`.

`results/comparison.png` shows density snapshots on a shared color scale. `results/long_run_metrics.png` shows all seeds. `summary.csv` and `timeseries.csv` contain measurements; `parameters.json` records the exact configuration. `validation.json` records controls and temporal/spatial resolution checks, when generated.

Additional checks (three seeds, 1,000 time units) gave:

- No slowing: late density CV 0.277–0.284, consistent with random placement.
- No oxygen depletion: CV 0.276–0.286, also consistent with random placement.
- Oxygen feedback with turning disabled: CV 0.275–0.306. The selected oxygen parameters therefore need directed turning to produce the strong aggregation seen here; oxygen-dependent speed alone did not do so within this test.
- Halving the time step or doubling the grid resolution changed the mean late CV by less than 2% in these tests. Largest-region statistics fluctuate more, especially for the density model. This is a limited robustness check, not full convergence proof.

Eight unit tests cover numerical invariants, periodic cluster labeling, visual-body independence and turning toward the preferred oxygen range on either side of that range.

## Reproducibility

Run the default comparison from this folder:

```sh
python3 experiments.py
python3 -m unittest -v
```

The live interface starts paused and batch cells are opt-in. Long-run notebook cells use the currently **applied** settings. The comparison cell uses `COMMON`, keeping the shared settings equal across models, and writes to `results_custom` so supplied reference results survive.

The notebook explains equations, parameters, limitations and biological motivation. `model.py` implements the dynamics, `live.py` supplies controls and plots, and `experiments.py` reproduces the comparison. Earlier project simulations were not used or modified.
