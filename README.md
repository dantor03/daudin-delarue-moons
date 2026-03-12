# Mean-field Neural ODE on Make Moons

Empirical verification of **Daudin & Delarue (2025)**: mean-field Neural ODEs with entropic regularization, the Polyak-Łojasiewicz condition, and exponential convergence guarantees — applied to the `make_moons` dataset.

> Daudin, S. & Delarue, F. (2025). *Genericity of the Polyak-Łojasiewicz inequality for mean-field Neural ODEs with entropic regularization.* [arXiv:2507.08486](https://arxiv.org/abs/2507.08486)

---

## What this is

A neural ODE learns a vector field $F(x,t)$ that continuously transforms the data distribution $\gamma_0$ (two interleaved moons) into a linearly separable $\gamma_T$ — all in the original $\mathbb{R}^2$ space, without any latent embedding.

The objective includes an **entropic regularization** term (KL divergence w.r.t. a supercoercive prior $\nu^\infty$) that, for any $\varepsilon > 0$, guarantees:

- A unique stable minimizer (Meta-Theorem 1 — genericity)
- Local Polyak-Łojasiewicz condition → **exponential convergence** without convexity (Meta-Theorem 2)

## Experiments

| | Experiment | What it shows |
|---|---|---|
| A | Feature evolution $\gamma_t$ | The ODE transports the moons into a separable distribution |
| B | Effect of $\varepsilon$ | Convergence curves, decision boundaries, Gibbs distribution, velocity field |
| C | PL condition verification | $\|\nabla J\|^2 \geq 2\mu(J - J^*)$ holds throughout training |
| D | Genericity of stable minimizer | Low noise → unique minimum; multiple inits converge to same $J^*$ |

## Results

| Paper result | Empirical evidence |
|---|---|
| ODE transforms $\gamma_0$ into separable $\gamma_T$ | Exp. A: moons → separated points, acc = 100% |
| $\varepsilon > 0$ forces Gibbs form on $\nu^*$ | Exp. B3: std($\theta$) decreases with $\varepsilon$ |
| PL condition holds with $\mu > 0$ | Exp. C: PL ratio $> 0$ across all epochs |
| Exponential convergence under PL | Exp. C2: linear decay in log scale |
| Genericity of unique minimizer | Exp. D: Std$(J^*) \approx 0$ for clean data |

## Quickstart

```bash
git clone https://github.com/dantor03/daudin-delarue-moons.git
cd daudin-delarue-moons
pip install -r requirements.txt
python daudin_delarue_moons.py
```

Requires Python 3.10+. Figures are saved as `A_*.png` … `D_*.png`.

## Architecture

```
X_0 ∈ ℝ²  →  [ODE: dX/dt = F(X,t), t ∈ [0,1]]  →  X_T ∈ ℝ²  →  [linear W,b]  →  logit
```

- Vector field: $F(x,t) = W_0 \tanh(W_1 [x,t]^\top + b_1)$, with $M = 64$ neurons
- Integrator: RK4, 10 steps
- Entropic penalty: $\varepsilon \cdot \sum_j [c_1 \theta_j^4 + c_2 \theta_j^2]$ with $c_1 = 0.05$, $c_2 = 0.5$

## References

- Chen et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
- Polyak (1963). *Gradient methods for minimizing functionals.*
- Villani (2003). *Topics in Optimal Transportation.* AMS.
