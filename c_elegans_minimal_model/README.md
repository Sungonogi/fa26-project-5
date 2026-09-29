# Minimal *C. elegans* aggregation model

## Research question

Can density-dependent slowing generate persistent aggregation by itself, and how does finite-range taxis change the outcome?

The code uses a standard two-dimensional active-particle framework and compares two candidate mechanisms:

- **S (slowdown):** an agent moves more slowly at high local density.
- **T (taxis):** an agent gradually turns toward the local centre of nearby agents.

It automatically runs the full 2 x 2 ablation design:

| Condition | Slowdown | Taxis |
|---|---:|---:|
| `baseline` | off | off |
| `slowdown` | on | off |
| `taxis` | off | on |
| `slowdown+taxis` | on | on |

## Literature connection

- Gray et al. (2004) show that oxygen sensing regulates aggregation and bordering. Oxygen is biological motivation here, but it is not explicitly simulated.
- Ding et al. (2019) motivate density-dependent motility and finite-range taxis as candidate rules for aggregation.
- Cates and Tailleur (2015) motivate testing whether sufficiently steep density-dependent slowing can generate motility-induced phase separation.

The taxis rule is therefore an **effective local interaction**, not evidence that the agents explicitly sense oxygen.

## Model rules

Each agent has position \(\mathbf{x}_i\) and direction \(\theta_i\):

\[
\mathbf{x}_i(t+\Delta t)=\mathbf{x}_i(t)+v_i\mathbf{e}(\theta_i)\Delta t.
\]

Direction undergoes rotational diffusion. When taxis is active, the direction also turns toward the local centre of neighbouring agents.

Local density is the number of neighbours inside a sensing disk divided by its area. When slowdown is active,

\[
v(\rho)=v_{\min}+(v_0-v_{\min})\exp(-a\rho).
\]

This form can decrease steeply enough to enter the mean-field MIPS instability region

\[
v(\rho)+\rho v'(\rho)<0,
\]

for part of the tested density range. That criterion is a theoretical guide; it does not guarantee phase separation in a finite, nonlocal simulation.

A short-range soft repulsion prevents agents from collapsing onto one point. Periodic boundaries remove wall accumulation.

## Run

From this directory:

```powershell
python minimal_worm_model.py --seeds 8 --steps 3000 --output-dir results
```

The program creates:

- `results/aggregation_comparison.png`: four representative final states and time-dependent metrics.
- `results/summary.csv`: late-time mean metrics for every condition and seed.

## Interpretation

The primary metric is the fraction of agents in the largest distance-connected cluster. Mean nearest-neighbour distance provides a second check.

Use conclusions of the following form:

> Within the tested model, parameter range, cluster definition, and observation time, mechanism S/T was sufficient or insufficient to generate persistent aggregation.

Avoid claiming that a mechanism is biologically necessary based on this simulation alone. A mechanism that fails at one parameter value may work after retuning, and an effective taxis rule does not identify its biological sensory source.

## Suggested report figures

1. The four-condition snapshot row.
2. Largest-cluster fraction versus time.
3. Mean nearest-neighbour distance versus time.

If time remains, vary only `slowdown_steepness` and `taxis_strength` over a small grid. Report that as a sensitivity analysis rather than adding another biological mechanism.
