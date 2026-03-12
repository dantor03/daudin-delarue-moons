# Neural ODEs de Campo Medio sobre Make Moons

Verificación empírica de **Daudin & Delarue (2025)**: Neural ODEs de campo medio con regularización entrópica, la condición de Polyak-Łojasiewicz y garantías de convergencia exponencial — aplicadas al dataset `make_moons`.

> Daudin, S. & Delarue, F. (2025). *Genericity of the Polyak-Łojasiewicz inequality for mean-field Neural ODEs with entropic regularization.* [arXiv:2507.08486](https://arxiv.org/abs/2507.08486)

---

## En qué consiste

Una Neural ODE aprende un campo vectorial $F(x,t)$ que transforma continuamente la distribución de datos $\gamma_0$ (dos medias lunas entrelazadas) en una distribución $\gamma_T$ linealmente separable — todo en el espacio original $\mathbb{R}^2$, sin ningún embedding a un espacio latente.

La función objetivo incluye un término de **regularización entrópica** (divergencia KL respecto a un prior supercoercivo $\nu^\infty$) que, para cualquier $\varepsilon > 0$, garantiza:

- Un minimizador estable único (Meta-Teorema 1 — genericidad)
- La condición local de Polyak-Łojasiewicz → **convergencia exponencial** sin necesidad de convexidad (Meta-Teorema 2)

## Experimentos

| | Experimento | Qué muestra |
|---|---|---|
| A | Evolución de features $\gamma_t$ | La ODE transporta las lunas a una distribución separable |
| B | Efecto de $\varepsilon$ | Curvas de convergencia, fronteras de decisión, distribución de Gibbs, campo de velocidad |
| C | Verificación de la condición PL | $\|\nabla J\|^2 \geq 2\mu(J - J^*)$ se cumple durante todo el entrenamiento |
| D | Genericidad del minimizador estable | Ruido bajo → mínimo único; múltiples inicializaciones convergen al mismo $J^*$ |

## Resultados

| Resultado del paper | Evidencia empírica |
|---|---|
| La ODE transforma $\gamma_0$ en $\gamma_T$ separable | Exp. A: lunas → puntos separados, acc = 100% |
| $\varepsilon > 0$ fuerza la forma de Gibbs en $\nu^*$ | Exp. B3: std($\theta$) decrece con $\varepsilon$ |
| La condición PL se cumple con $\mu > 0$ | Exp. C: ratio PL $> 0$ en todas las épocas |
| Convergencia exponencial bajo PL | Exp. C2: decaimiento lineal en escala log |
| Genericidad del minimizador único | Exp. D: Std$(J^*) \approx 0$ para datos limpios |

## Guía de inicio rápido

```bash
git clone [https://github.com/dantor03/daudin-delarue-moons.git](https://github.com/dantor03/daudin-delarue-moons.git)
cd daudin-delarue-moons
pip install -r requirements.txt
python daudin_delarue_moons.py

```

Requiere Python 3.10+. Las figuras se guardarán como `A_*.png` … `D_*.png`.

## Arquitectura

```text
X_0 ∈ ℝ²  →  [ODE: dX/dt = F(X,t), t ∈ [0,1]]  →  X_T ∈ ℝ²  →  [lineal W,b]  →  logit

```

* Campo vectorial: $F(x,t) = W_0 \tanh(W_1 [x,t]^\top + b_1)$, con $M = 64$ neuronas
* Integrador: RK4, 10 pasos
* Penalización entrópica: $\varepsilon \cdot \sum_j [c_1 \theta_j^4 + c_2 \theta_j^2]$ con $c_1 = 0.05$, $c_2 = 0.5$

## Referencias

* Chen et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
* Polyak (1963). *Gradient methods for minimizing functionals.*
* Villani (2003). *Topics in Optimal Transportation.* AMS.
