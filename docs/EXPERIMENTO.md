# Neural ODEs de Campo Medio con Regularización Entrópica sobre Make Moons

**Referencia:** Daudin, S. & Delarue, F. (2025). *Genericity of the Polyak-Łojasiewicz inequality for mean-field Neural ODEs with entropic regularization.* arXiv:2507.08486.

**Código:** `daudin_delarue_moons.py`

---

## Índice

- [Neural ODEs de Campo Medio con Regularización Entrópica sobre Make Moons](#neural-odes-de-campo-medio-con-regularización-entrópica-sobre-make-moons)
  - [Índice](#índice)
  - [1. Contexto y motivación](#1-contexto-y-motivación)
  - [2. Marco teórico](#2-marco-teórico)
    - [2.1 Neural ODEs en el límite de campo medio](#21-neural-odes-en-el-límite-de-campo-medio)
    - [2.2 Campo vectorial prototípico](#22-campo-vectorial-prototípico)
    - [2.3 Ecuación de continuidad](#23-ecuación-de-continuidad)
    - [2.4 Regularización entrópica y control óptimo](#24-regularización-entrópica-y-control-óptimo)
    - [2.5 La condición Polyak-Łojasiewicz](#25-la-condición-polyak-łojasiewicz)
    - [2.6 Los dos Meta-Teoremas](#26-los-dos-meta-teoremas)
  - [3. Implementación](#3-implementación)
    - [3.1 Dataset: Make Moons](#31-dataset-make-moons)
    - [3.2 Arquitectura](#32-arquitectura)
    - [3.3 Función objetivo](#33-función-objetivo)
  - [4. Experimento A — Evolución de $\gamma_t$ (marginal en $x$)](#4-experimento-a--evolución-de-gamma_t-marginal-en-x)
    - [Lectura de la figura](#lectura-de-la-figura)
    - [Interpretación](#interpretación)
  - [5. Experimento B — Efecto del parámetro $\\varepsilon$](#5-experimento-b--efecto-del-parámetro-varepsilon)
    - [B1 — Curvas de convergencia](#b1--curvas-de-convergencia)
    - [B2 — Fronteras de decisión](#b2--fronteras-de-decisión)
    - [B3 — Prior de Gibbs: comportamiento MAP de los parámetros](#b3--prior-de-gibbs-comportamiento-map-de-los-parámetros)
    - [B4 — Campo de velocidad](#b4--campo-de-velocidad)
  - [6. Experimento C — Verificación empírica de la condición PL](#6-experimento-c--verificación-empírica-de-la-condición-pl)
    - [C1 — Diagrama log-log: $|\\nabla J|^2$ vs $(J - J^\*)$](#c1--diagrama-log-log-nabla-j2-vs-j---j)
    - [C2 — Convergencia exponencial (escala semilog)](#c2--convergencia-exponencial-escala-semilog)
    - [C3 — Constante PL estimada $\\hat{\\mu}$ por $\\varepsilon$](#c3--constante-pl-estimada-hatmu-por-varepsilon)
    - [C4 — Ratio PL a lo largo del entrenamiento](#c4--ratio-pl-a-lo-largo-del-entrenamiento)
  - [7. Experimento D — Genericidad del minimizador estable](#7-experimento-d--genericidad-del-minimizador-estable)
    - [Lectura por filas](#lectura-por-filas)
    - [Interpretación](#interpretación-1)
  - [8. Conclusiones](#8-conclusiones)
  - [Referencias](#referencias)

---

## 1. Contexto y motivación

Las redes neuronales profundas se pueden entender como sistemas de control: cada capa transforma la representación de los datos, y el objetivo es aprender los parámetros de esa transformación para que la representación final sea fácilmente clasificable. En el límite de capas continuas, esta idea da lugar a las **Neural ODEs** (Chen et al., 2018). En el límite de neuronas infinitas por capa, aparece el **límite de campo medio**.

El paper de Daudin & Delarue (2025) estudia la intersección de ambos límites y demuestra dos resultados sorprendentes:

- La existencia de un **minimizador estable único** es genérica (ocurre para casi toda distribución inicial de datos).
- Cerca de ese minimizador, se satisface la **desigualdad de Polyak-Łojasiewicz (PL)**, lo que garantiza **convergencia exponencial** del descenso en gradiente hacia el óptimo global — sin ninguna hipótesis de convexidad.

Estos resultados son especialmente notables porque el paisaje de pérdida de las redes neuronales es en general altamente no convexo.

Este documento presenta la verificación empírica de estos resultados sobre el dataset `make_moons` de scikit-learn.

---

## 2. Marco teórico

### 2.1 Neural ODEs en el límite de campo medio

Una red ResNet profunda con $L$ capas se puede escribir como:

$$X_{k+1} = X_k + \frac{1}{L} F_k(X_k), \quad k = 0, 1, \ldots, L-1$$

donde $X_k \in \mathbb{R}^{d_1}$ es la representación en la capa $k$ y $F_k$ es el campo vectorial aprendido en esa capa. Tomando el límite $L \to \infty$ con paso $dt = 1/L$, la red converge a la **ODE**:

$$\frac{dX_t}{dt} = F(X_t, t), \quad t \in [0, T], \quad X_0 = \text{dato de entrada}$$

A su vez, si cada capa tiene $M \to \infty$ neuronas, la distribución de parámetros converge a una **medida** $\nu_t \in \mathcal{P}(A)$ sobre el espacio de parámetros $A$. En este límite de campo medio, el campo vectorial efectivo es:

$$F(x, t) = \int_A b(x, a) \, d\nu_t(a)$$

donde $b(x, a)$ es la contribución de un único parámetro $a$ a la dinámica. Así, en lugar de optimizar sobre un vector de parámetros finito $\theta \in \mathbb{R}^p$, el problema se convierte en optimizar sobre **una trayectoria de medidas** $(\nu_t)_{t \in [0,T]} \subset \mathcal{P}(A)$.

### 2.2 Campo vectorial prototípico

El paper usa el campo prototípico (Ejemplo 1.1, ec. 1.8):

$$b(x, a) = \sigma(a_1 \cdot x + a_2) \cdot a_0, \quad \sigma = \tanh$$

donde $a = (a_0, a_1, a_2) \in A = \mathbb{R}^{d_1} \times \mathbb{R}^{d_1} \times \mathbb{R}$. Cada "neurona" $a$ es una transformación de tipo perceptrón: proyección lineal $a_1 \cdot x + a_2$, seguida de activación $\tanh$, seguida de reescalado $a_0$.

El campo efectivo con $M$ partículas (aproximación de $\nu_t$ por $M$ muestras discretas) es:

$$F(x, t) \approx \frac{1}{M} \sum_{m=1}^{M} \sigma\left(a_1^m \cdot x + a_2^m\right) \cdot a_0^m$$

que es exactamente una **red neuronal de una capa oculta** con $M$ neuronas y pesos que varían en el tiempo $t$.

### 2.3 Ecuación de continuidad

La distribución de datos $\gamma_t \in \mathcal{P}(\mathbb{R}^{d_1} \times \mathbb{R}^{d_2})$ en el instante $t$ evoluciona según la **ecuación de continuidad** (ec. 1.3 del paper):

$$\partial_t \gamma_t + \text{div}_x\left(F(x, t) \, \gamma_t\right) = 0$$

Esta PDE expresa que la "masa" (densidad de datos) se conserva y se transporta con el campo $F$: no se crean ni destruyen puntos, simplemente se mueven. El operador $\text{div}_x$ actúa **solo sobre la componente de features** $x \in \mathbb{R}^{d_1}$; la componente de etiquetas $y \in \mathbb{R}^{d_2}$ no aparece bajo la divergencia porque $F$ no actúa sobre ella. Intuitivamente, la ODE empuja la componente $x$ de cada punto a lo largo de trayectorias determinadas por $F$, de modo que la marginal en $x$ de $\gamma_0$ (no separable) se transforma en la marginal en $x$ de $\gamma_T$ (linealmente separable).

La solución formal es el push-forward:

$$\gamma_t = (\phi_t)_{\sharp} \gamma_0$$

donde $\phi_t : \mathbb{R}^{d_1} \times \mathbb{R}^{d_2} \to \mathbb{R}^{d_1} \times \mathbb{R}^{d_2}$ es el flujo del campo $F$ sobre la distribución conjunta, definido por

$$\phi_t(x_0, y) = (X_t, y)$$

La ODE integra $dX_t/dt = F(X_t, t)$ solo para la componente de features $x_0 \to X_t$, mientras la etiqueta $y$ se transporta pasivamente sin cambiar. Dicho de otro modo: cada "partícula" de $\gamma_t$ es el par $(X_t^i, Y_0^i)$; los features evolucionan, las etiquetas son invariantes bajo el flujo.

### 2.4 Regularización entrópica y control óptimo

La función objetivo del problema de control es (ec. 1.6):

$$J(\gamma_0, \nu) = \underbrace{\int L(x, y) \, d\gamma_T(x, y)}_{\text{coste terminal}} + \underbrace{\varepsilon \int_0^T \mathcal{E}(\nu_t \mid \nu^\infty) \, dt}_{\text{penalización entrópica}}$$

donde:
- $L(x, y) = \text{BCE}(W \cdot x + b, y)$ es el coste de clasificación binaria
- $\mathcal{E}(\nu_t \mid \nu^\infty) = \int \log\left(\frac{d\nu_t}{d\nu^\infty}\right) d\nu_t$ es la divergencia KL respecto al prior $\nu^\infty$
- $\varepsilon > 0$ controla la intensidad de la regularización

El prior es $\nu^\infty(da) \propto e^{-\ell(a)} \, da$ con potencial **supercoercivo** (Assumption Regularity (i)):

$$\ell(a) = c_1 |a|^4 + c_2 |a|^2, \quad c_1 = 0.05, \quad c_2 = 0.5$$

La supercoercividad ($c_1 > 0$, es decir, crecimiento cuártico) es esencial porque garantiza la **desigualdad de log-Sobolev** para $\nu^\infty$:

$$\mathcal{E}(\mu \mid \nu^\infty) \leq \frac{1}{2\rho} \mathcal{I}(\mu \mid \nu^\infty)$$

donde $\mathcal{I}$ es la información de Fisher. Esta desigualdad es el ingrediente técnico que conecta la regularización entrópica con la condición PL.

El control óptimo $\nu_t^*$ tiene la **forma de Gibbs** (ec. 1.9):

$$\nu_t^*(da) \propto \exp\left(-\ell(a) - \frac{1}{\varepsilon} \int_{\mathbb{R}^{d_1} \times \mathbb{R}^{d_2}} b(x, a) \cdot \nabla u_t(x,y) \, d\gamma_t(x,y)\right) da$$

donde $u_t$ es la función de valor (solución de la ecuación de Hamilton-Jacobi-Bellman hacia atrás). Esto dice que el control óptimo concentra $\nu_t^*$ en los parámetros $a$ que minimizan $L$ pero "penalizados" por el potencial $\ell(a)$.

### 2.5 La condición Polyak-Łojasiewicz

La **condición PL** (también llamada condición de Kurdyka-Łojasiewicz o desigualdad de gradiente suficiente) establece que existe $\mu > 0$ tal que:

$$\|\nabla J(\theta)\|^2 \geq 2\mu \cdot (J(\theta) - J^*)$$

donde $J^*$ es el valor mínimo de $J$. Su importancia es enorme: si la condición PL se cumple a lo largo de la trayectoria de descenso en gradiente con tasa de aprendizaje $\eta$, entonces la convergencia es **exponencial**:

$$J(\theta_k) - J^* \leq (1 - 2\eta\mu)^k \cdot (J(\theta_0) - J^*)$$

La condición PL es **más débil que la convexidad estricta** y que la condición de Łojasiewicz estándar, pero suficiente para garantizar convergencia global al mínimo.

En el contexto de campo medio, la condición análoga sobre $\nu$ es:

$$\mathcal{I}(\gamma_0, \nu) \geq c \cdot (J(\gamma_0, \nu) - J(\gamma_0, \nu^*))$$

donde $\mathcal{I}$ es la información de Fisher en el espacio de medidas.

### 2.6 Los dos Meta-Teoremas

**Meta-Teorema 1** (genericidad): Existe un conjunto abierto y denso $\mathcal{O}$ de condiciones iniciales $\gamma_0$ (en la topología de convergencia débil sobre $\mathcal{P}(\mathbb{R}^{d_1} \times \mathbb{R}^{d_2})$) tal que para todo $\gamma_0 \in \mathcal{O}$, el problema de control tiene un único minimizador **estable**.

> *"Casi toda distribución inicial de datos produce un paisaje de pérdida con un único mínimo profundo."*

**Meta-Teorema 2** (condición PL local): Para $\gamma_0 \in \mathcal{O}$ y $\varepsilon > 0$, la condición PL se cumple localmente cerca del minimizador estable con constante $c > 0$ que depende de $\varepsilon$ pero no necesita ser grande.

> *"Con cualquier $\varepsilon > 0$ (aunque sea arbitrariamente pequeño), el descenso en gradiente converge exponencialmente al mínimo global — sin convexidad."*

---

## 3. Implementación

### 3.1 Dataset: Make Moons

Se usa `make_moons` de scikit-learn con $N = 400$ puntos y ruido $\sigma = 0.12$, estandarizado con `StandardScaler`. Este dataset es canónico para probar clasificación no lineal: las dos clases forman medias lunas entrelazadas que no son separables linealmente en el espacio original.

En el lenguaje del paper, los datos estandarizados constituyen la **distribución inicial empírica conjunta** (features + etiquetas):

$$\gamma_0 = \frac{1}{N} \sum_{i=1}^{N} \delta_{(X_0^i,\, Y_0^i)} \;\in\; \mathcal{P}(\mathbb{R}^{d_1} \times \mathbb{R}^{d_2})$$

La ODE desplaza solo la componente de features $X_0^i \to X_t^i$; la etiqueta $Y_0^i$ permanece fija. Por eso $\gamma_t = \frac{1}{N}\sum_i \delta_{(X_t^i, Y_0^i)}$ sigue viviendo en $\mathbb{R}^{d_1} \times \mathbb{R}^{d_2}$ para todo $t$. Las figuras muestran la **marginal en $x$** de $\gamma_t$, coloreada según $Y_0^i$.

### 3.2 Arquitectura

La arquitectura implementa el setup del paper en el espacio original de features, evitando el embedding a espacio latente que usan implementaciones previas:

```
X_0 ∈ ℝ²  →  [ODE: dX/dt = F(X,t), t ∈ [0,1]]  →  X_T ∈ ℝ²  →  [lineal W,b]  →  logit
```

| Componente | Dimensiones | Ecuación |
|---|---|---|
| Input | $\mathbb{R}^2$ | $X_0 =$ dato |
| Campo vectorial | $\mathbb{R}^{2+1} \to \mathbb{R}^M \to \mathbb{R}^2$ | $F(x,t) = W_0 \tanh(W_1 [x, t]^\top + b_1)$ |
| Integrador | RK4, 10 pasos | $X_{t+dt} = X_t + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$ |
| Clasificador | $\mathbb{R}^2 \to \mathbb{R}$ | $\text{logit} = W \cdot X_T + b$ |

**Parámetros:** $M = 64$ neuronas, $T = 1.0$, `n_steps = 10` → $\approx 450$ parámetros entrenables.

> **Aproximación respecto al paper (dependencia temporal de $\nu_t$).**
> En el problema de control de Daudin & Delarue, la medida de parámetros $\nu_t$ es una *trayectoria* arbitraria sobre $\mathcal{P}(A)$, con $a^m(t)$ variando libremente en $t$. En la implementación se usa la técnica estándar de las Neural ODEs: los pesos $(W_0, W_1, b_1)$ son **estáticos** y $t$ se concatena como feature adicional. Esto equivale a restringir el espacio de control a la familia paramétrica donde $a_1^m$ y $a_0^m$ son constantes en $t$ y solo la componente de sesgo varía linealmente con $t$. El campo resultante $F(x,t)$ **sí depende genuinamente de $t$**, pero la medida $\nu_t$ subyacente no es arbitraria — es una familia concreta. Esta aproximación es estándar en la literatura de Neural ODEs y suficientemente expresiva para los experimentos, pero constituye una restricción del espacio de control óptimo del paper.

**Integrador RK4** en lugar de Euler: el error de truncamiento **local** de RK4 es $O(dt^5)$, lo que tras acumular sobre el intervalo $[0,T]$ da un error **global** de $O(dt^4)$. El método de Euler tiene error global $O(dt)$. Con $dt = T/n\_\text{steps} = 0.1$:

$$\text{Error global RK4} \sim dt^4 = 0.1^4 = 10^{-4}, \qquad \text{Error global Euler} \sim dt = 0.1 = 10^{-1}$$

La diferencia de dos órdenes de magnitud en el error hace que RK4 sea esencial para integrar fielmente la ODE del paper.

### 3.3 Función objetivo

La implementación de $J$ combina BCE y una penalización de regularización:

$$J(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^N \text{BCE}(\text{logit}_i, y_i)}_{\text{coste terminal}} + \varepsilon \cdot \underbrace{\frac{1}{N_\theta} \sum_j \left[c_1 \theta_j^4 + c_2 \theta_j^2\right]}_{\text{aprox. energía de } \mathcal{E}(\nu \mid \nu^\infty)}$$

> **Aproximación respecto al paper (término entrópico).**
> La divergencia KL completa se descompone como $\mathcal{E}(\nu_t \mid \nu^\infty) = \mathbb{E}_{\nu_t}[\ell(a)] - H(\nu_t)$, donde $H(\nu_t) = -\int \log(d\nu_t)\,d\nu_t$ es la entropía diferencial. La implementación usa solo el **término de energía** $\mathbb{E}_{\nu_t}[\ell(a)] \approx \frac{1}{M}\sum_m \ell(a^m)$. Para parámetros deterministas (suma de deltas de Dirac, $\nu_t = \frac{1}{M}\sum_m \delta_{a^m}$), la entropía diferencial es $H(\nu_t) = -\infty$ respecto a un prior continuo, de modo que la KL completa es técnicamente infinita. La aproximación por el término de energía es la única opción práctica con estimadores puntuales, y equivale a regularización L4+L2. Para implementar la verdadera regularización entrópica serían necesarias dinámicas de Langevin (añadir ruido gaussiano a los gradientes) o inferencia variacional (distribuir probabilísticamente cada peso). Los experimentos verifican propiedades del sistema *con esta aproximación*, no del control entrópico completo del paper.

La penalización cuártica $c_1 \theta^4$ (L4) es la diferencia clave respecto a la regularización L2 estándar: es el mínimo crecimiento que garantiza la desigualdad de log-Sobolev para $\nu^\infty$, que es el ingrediente técnico que implica la condición PL.

---

## 4. Experimento A — Evolución de $\gamma_t$ (marginal en $x$)

**Objetivo:** Visualizar cómo la ODE de campo medio transforma la distribución conjunta $\gamma_t \in \mathcal{P}(\mathbb{R}^{d_1} \times \mathbb{R}^{d_2})$ a medida que avanza el "tiempo de red" $t \in [0, T]$. En la práctica, cada snapshot muestra la **marginal en $x$** de $\gamma_t$, es decir, $(\gamma_t)_x = \int \gamma_t(dx, dy)$, coloreada según la etiqueta $y$ (que permanece constante para cada partícula).

**Configuración:** $\varepsilon = 0.01$, $M = 64$, $T = 1.0$, 800 épocas de entrenamiento.

![Evolución de features γ_t](../figuras/A_feature_evolution.png)

### Lectura de la figura

**Fila superior (izquierda a derecha):**
- **$\gamma_{t=0}$** — La distribución inicial: las dos lunas entrelazadas. Ningún clasificador lineal puede separarlas. La ODE parte de aquí.
- **$\gamma_{t=0.20}$** (paso 2/10) — Las lunas comienzan a deformarse. El campo $F(x, t)$ ya está empujando los puntos de cada clase en direcciones distintas.
- **$\gamma_{t=0.50}$** (paso 5/10) — A mitad del flujo, las dos clases están claramente más separadas, aunque todavía hay cierto solapamiento.
- **Curvas de pérdida** — La pérdida total $J$ (verde), la BCE pura (azul) y la penalización entrópica (naranja). La penalización crece gradualmente a medida que los parámetros se alejan del prior $\nu^\infty$ para resolver la clasificación. La BCE decrece y converge a casi 0.

**Fila inferior:**
- **$\gamma_{t=0.70}$** (paso 7/10) y **$\gamma_{t=1.0}$** (paso 10/10) — Las dos clases están ya bien separadas. El clasificador lineal $W \cdot X_T + b$ puede separarlas con acc $= 1.000$. El rectángulo discontinuo indica la extensión original de $\gamma_0$: los puntos se han movido considerablemente.
- **Trayectorias $X_t$** — 40 partículas seleccionadas (20 de cada clase) con su trayectoria completa de $t=0$ (punto) a $t=T$ (estrella). Los puntos de la misma clase siguen trayectorias coherentes y coordinadas, lo que refleja que $F(x,t)$ actúa de forma colectiva sobre $\gamma_t$ — característica del campo medio.
- **Frontera de decisión** — La isocurva $P(y=1|x) = 0.5$ en el espacio original $\mathbb{R}^2$. A pesar de que el clasificador final es lineal sobre $X_T$, la frontera en $X_0$ es altamente no lineal, pues incorpora toda la geometría del flujo $\phi_T$.

### Interpretación

Este experimento es la materialización visual de la ecuación de continuidad:

$$\partial_t \gamma_t + \text{div}_x(F(x,t) \, \gamma_t) = 0$$

La "masa" de datos se conserva y se transporta. El campo $F$ aprendido es el que hace que las clases se separen, y el clasificador lineal final es simplemente un hiperplano en $\mathbb{R}^2$ sobre la representación transformada.

---

## 5. Experimento B — Efecto del parámetro $\varepsilon$

**Objetivo:** Estudiar cómo la intensidad de la regularización entrópica $\varepsilon$ afecta a la convergencia, las fronteras de decisión, la distribución de parámetros y el campo de velocidad aprendido.

**Configuración:** $\varepsilon \in \{0, 0.001, 0.01, 0.1, 0.5\}$, misma inicialización para todos (misma semilla antes de cada modelo), 700 épocas.

### B1 — Curvas de convergencia

![Curvas de convergencia](../figuras/B1_convergence_curves.png)

**Panel izquierdo (Pérdida total $J$):** Todos los modelos convergen con curvas similares. Para $\varepsilon$ mayores, el valor asintótico de $J$ es más alto porque incluye un término de penalización mayor. Sin embargo, la **velocidad de convergencia** es comparable, lo que confirma que $\varepsilon$ no frena el aprendizaje.

**Panel central (Accuracy):** Todos los modelos alcanzan el 100% de exactitud, incluyendo $\varepsilon = 0.5$. Esto confirma que incluso regularizaciones fuertes no degradan la capacidad del modelo en este dataset. La distinción entre $\varepsilon = 0$ y $\varepsilon > 0$ no es en accuracy sino en **garantías teóricas de convergencia**.

**Panel derecho (Penalización entrópica $\mathcal{E}/N_\text{params}$):** Un resultado intuitivamente correcto: modelos con $\varepsilon = 0$ tienen la penalización más alta (los parámetros están más alejados del prior $\nu^\infty$, porque nada los atrae hacia él). A medida que $\varepsilon$ aumenta, los parámetros están más concentrados cerca del origen, reduciendo $\mathcal{E}$.

### B2 — Fronteras de decisión

![Fronteras de decisión](../figuras/B2_decision_boundaries.png)

Las cinco fronteras clasifican perfectamente las lunas (acc $= 1.000$). La diferencia es geométrica: mayor $\varepsilon$ produce fronteras más suaves y regulares, mientras que $\varepsilon = 0$ puede producir fronteras más irregulares o sobreajustadas. Esto es el efecto de regularización clásico: $\varepsilon$ controla la complejidad de la solución.

La forma de la frontera no es arbitraria: refleja la geometría del flujo $\phi_T$ aprendido. Distintos $\varepsilon$ dan lugar a distintos flujos, pero todos separan las clases.

### B3 — Prior de Gibbs: comportamiento MAP de los parámetros

![Distribución de Gibbs](../figuras/B3_gibbs_parameter_dist.png)

Los histogramas muestran la distribución empírica de todos los parámetros de la red al final del entrenamiento, comparada con el prior teórico $\nu^\infty \propto e^{-\ell(a)}$ (curva blanca discontinua).

> **Precisión metodológica.** La forma de Gibbs del control óptimo del paper, $\nu_t^*(da) \propto \exp(-\ell(a) - \frac{1}{\varepsilon}(\cdot))da$, es una *distribución de probabilidad* sobre parámetros que requeriría muestreo (Langevin, MCMC o inferencia variacional) para verificarse. La implementación, al usar parámetros **deterministas** + penalización L4+L2, realiza una estimación **MAP** (Maximum A Posteriori) bajo el prior de Gibbs $\nu^\infty$. Los histogramas no son muestras de $\nu_t^*$ — son los $M = 64$ estimadores MAP individuales de cada "neurona". Por tanto, el panel ilustra un *comportamiento consistente con la predicción cualitativa de la forma de Gibbs*, no una verificación directa de la distribución de Gibbs óptima.

Con esa aclaración, la observación central es válida: la desviación estándar de los parámetros **decrece monótonamente con $\varepsilon$**:

| $\varepsilon$ | std($\theta$) |
|---|---|
| 0.0   | 0.481 |
| 0.001 | 0.480 |
| 0.01  | 0.471 |
| 0.1   | 0.365 |
| 0.5   | 0.284 |

Este patrón es exactamente lo que predice la forma de Gibbs. El exponente tiene dos términos en competencia:

$$\nu_t^*(da) \propto \exp\!\left(\underbrace{-\ell(a)}_{\text{prior}} \underbrace{-\frac{1}{\varepsilon}\int b(x,a)\cdot\nabla u_t \, d\gamma_t}_{\text{clasificación} \times 1/\varepsilon}\right)da$$

El término de clasificación lleva un factor $1/\varepsilon$. Cuando $\varepsilon$ es **pequeño**, $1/\varepsilon$ es grande y ese término domina: en el MAP, los parámetros se colocan donde mejor clasifican, ignorando el prior, y los pesos resultantes están dispersos. Cuando $\varepsilon$ es **grande**, $1/\varepsilon$ se hace pequeño, el prior $-\ell(a)$ domina y el MAP encoge los parámetros hacia cero — comportamiento clásico de *weight decay*. Este es el mecanismo MAP análogo a la concentración de la distribución de Gibbs hacia $\nu^\infty$.

### B4 — Campo de velocidad

![Campo de velocidad](../figuras/B4_velocity_field.png)

El campo vectorial efectivo $F(x, t=0.5)$ evaluado a mitad del flujo para cada $\varepsilon$. Las flechas muestran la dirección normalizada del campo (la velocidad con la que el flujo mueve cada punto); el color indica la magnitud.

El campo muestra cómo la ODE "empuja" los puntos hacia regiones separables: los puntos de clase 0 (rojo) son empujados en una dirección y los de clase 1 (azul) en otra, creando la separación que el clasificador lineal después explota. Las diferencias entre modelos son sutiles porque todos logran el 100% de accuracy mediante flujos cualitativamente similares, pero el grado de regularización cambia la suavidad del campo.

---

## 6. Experimento C — Verificación empírica de la condición PL

**Objetivo:** Comprobar que la desigualdad $\|\nabla J(\theta)\|^2 \geq 2\mu \cdot (J(\theta) - J^*)$ se cumple empíricamente durante todo el entrenamiento, verificando el Meta-Teorema 2.

**Datos:** Se reutilizan los modelos ya entrenados del Experimento B (sin reentrenar).

![Verificación PL](../figuras/C_pl_verification.png)

### C1 — Diagrama log-log: $\|\nabla J\|^2$ vs $(J - J^*)$

Cada punto en el diagrama corresponde a una época de entrenamiento. Las coordenadas son:
- **Eje X:** Excess cost $J(\theta) - J^*$ (cuánto le falta al modelo para llegar al óptimo)
- **Eje Y:** $\|\nabla J(\theta)\|^2$ (norma al cuadrado del gradiente)

Las dos rectas blancas son **referencias**: $\|\nabla J\|^2 = 2(J-J^*)$ para $c=1$ (discontinua) y $\|\nabla J\|^2 = 20(J-J^*)$ para $c=10$ (punteada). La condición PL con constante $\mu$ se satisface si los puntos están por encima de la recta $y = 2\mu x$.

Los puntos coloreados aparecen **por debajo** de ambas líneas de referencia, lo que simplemente indica que la constante empírica $\hat{\mu} \ll 1$ (de hecho $\hat{\mu} \approx 0.002$, consistente con la tabla de C3). Esto no es una violación de PL: la condición solo exige $\mu > 0$, no $\mu \geq 1$.

**Lo que sí confirma el gráfico:** la nube de puntos sigue una tendencia aproximadamente paralela a las rectas blancas (pendiente $\approx 1$ en log-log). Esto es la firma visual de la condición PL: $\|\nabla J\|^2$ crece proporcionalmente a $(J-J^*)$, con la constante de proporcionalidad $2\hat{\mu} \approx 0.004$.

### C2 — Convergencia exponencial (escala semilog)

El exceso de coste $J(\theta^s) - J^*$ en escala logarítmica muestra que todas las curvas son aproximadamente lineales (en escala log), lo que confirma el decay exponencial garantizado por la condición PL:

$$J(\theta_s) - J^* \lesssim (J(\theta_0) - J^*) \cdot e^{-2\mu s}$$

La ligera curvatura al final se debe al cosine annealing (el scheduler reduce el lr a casi 0 al final del entrenamiento), no a una violación de PL.

### C3 — Constante PL estimada $\hat{\mu}$ por $\varepsilon$

La constante PL se estima de forma conservadora como el **percentil 10** del ratio $\|\nabla J\|^2 / (2(J - J^*))$ a lo largo de todas las épocas:

| $\varepsilon$ | $\hat{\mu}_{PL}$ |
|---|---|
| 0.0   | 0.0035 |
| 0.001 | 0.0035 |
| 0.01  | 0.0032 |
| 0.1   | 0.0022 |
| 0.5   | 0.0021 |

**Interpretación cuidadosa:** El paper garantiza $\mu > 0$ para todo $\varepsilon > 0$, pero **no** que $\mu$ crezca con $\varepsilon$. Empíricamente, $\hat{\mu}$ decrece ligeramente con $\varepsilon$ porque valores mayores de $\varepsilon$ elevan $J^*$ (el óptimo global cuesta más en presencia de mayor regularización), lo que aumenta el denominador $(J - J^*)$ del ratio. El resultado clave es que $\hat{\mu} > 0$ para todos los $\varepsilon$, incluyendo $\varepsilon = 0$ (donde, sin embargo, el paper no da garantía teórica).

### C4 — Ratio PL a lo largo del entrenamiento

El ratio $\|\nabla J\|^2 / (2(J - J^*))$ se mantiene positivo y por encima de $\mu \approx 0.002$ durante todo el entrenamiento para todos los modelos. La condición PL no solo se cumple al inicio (cuando el modelo está lejos del óptimo) sino también en las etapas tardías del entrenamiento, confirmando la hipótesis **local** del Meta-Teorema 2.

---

## 7. Experimento D — Genericidad del minimizador estable

**Objetivo:** Verificar empíricamente el Meta-Teorema 1: que la unicidad del minimizador estable es una propiedad "genérica" (válida para casi toda distribución inicial $\gamma_0$).

### Diseño mejorado (Ideas A + B + D)

El diseño anterior variaba el ruido del dataset para cambiar $\gamma_0$, pero esto mezclaba la variación de $\gamma_0$ con la dificultad intrínseca de la tarea. El diseño actual emplea tres mejoras:

**Idea A — Variar $\gamma_0$ via semilla del dataset:**
Se generan $n_{\rm datasets} = 8$ datasets `make_moons` con el **mismo nivel de ruido** ($\sigma = 0.12$) pero distintas semillas aleatorias. Cada semilla produce una distribución $\gamma_0$ diferente manteniendo la misma dificultad de clasificación. Para cada $\gamma_0$ se entrenan $n_{\rm inits} = 3$ modelos con semillas de parámetros independientes.

**Idea B — Criterio de convergencia adaptativo:**
En lugar de fijar un número fijo de épocas (que puede dejar algunos modelos sin converger y a otros con tiempo malgastado), se detiene el entrenamiento cuando la norma cuadrada del gradiente satisface

$$\|\nabla J\|^2 < \delta = 5 \times 10^{-5} \quad \text{durante 5 épocas consecutivas}$$

con un mínimo de 300 épocas y un máximo de 800. Esto garantiza que todos los modelos han convergido realmente, haciendo la comparación de $J^*$ entre inicializaciones más limpia.

**Idea D — Métrica de distancia de frontera ($\Delta_{\rm boundary}$):**
La variabilidad escalar $\text{Std}(J^*)$ puede ser pequeña aunque las fronteras de decisión sean geométricamente distintas (dos mínimos con el mismo valor de pérdida). Para capturar la variabilidad geométrica, se evalúan todos los modelos en una rejilla densa $80 \times 80$ sobre el espacio de features y se calcula:

$$\Delta_{\rm boundary}(\gamma_0) = \frac{1}{\binom{n_{\rm inits}}{2}} \sum_{i < j} \frac{\|\sigma(m_i(\text{grid})) - \sigma(m_j(\text{grid}))\|_F}{\sqrt{n_{\rm grid}}}$$

donde $\sigma$ es la sigmoide y $n_{\rm grid} = 6400$. Si $\Delta_{\rm boundary} \approx 0$, todas las inicializaciones producen la **misma frontera geométrica**, evidencia directa de que el minimizador es único para esa $\gamma_0$.

### Figura

![Genericidad](../figuras/D_stability_genericity.png)

La figura tiene cuatro paneles:

- **(0,0) Barras de $J^*$:** grupos por $\gamma_0$ (8 grupos), barras por inicialización (3 colores). Permite ver qué $\gamma_0$ tienen inicializaciones dispersas y cuáles concentradas.

- **(0,1) $\Delta_{\rm boundary}$ y Std$(J^*)$:** barras en coral para $\Delta_{\rm boundary}$ (eje izquierdo), línea en azul para Std$(J^*)$ (eje derecho). Ambas métricas deben ser pequeñas para afirmar genericidad. El coeficiente entre ambas revela si la variabilidad es escalar (solo $J^*$) o geométrica (fronteras distintas).

- **(1,0) Fronteras superpuestas — $\gamma_0$ con $\Delta$ mínimo:** las 3 fronteras de decisión (contornos de nivel 0.5) se superponen casi perfectamente. El fondo muestra el promedio de las salidas del clasificador. Esta $\gamma_0$ es la "más genérica" del conjunto.

- **(1,1) Fronteras superpuestas — $\gamma_0$ con $\Delta$ máximo:** las 3 fronteras difieren visiblemente entre sí. Esta $\gamma_0$ está más cerca del borde de $\mathcal{O}$ o fuera de él.

### Interpretación

- **$\Delta_{\rm boundary} \approx 0$ y Std$(J^*) \approx 0$:** todas las inicializaciones encuentran la misma solución tanto en valor ($J^*$) como en geometría (frontera). Evidencia de que $\gamma_0 \in \mathcal{O}$ y el minimizador es único.

- **$\Delta_{\rm boundary}$ grande:** distintas inicializaciones convergen a fronteras geométricamente diferentes. Esto puede ocurrir aunque $\text{Std}(J^*)$ sea pequeño (simetría del paisaje) e indica que $\gamma_0$ puede estar fuera de $\mathcal{O}$ o en su frontera.

- **El criterio adaptativo** asegura que las diferencias observadas son genuinas (convergencia real a distintos mínimos) y no artefactos de un presupuesto de épocas insuficiente.

- **Genericidad $\neq$ convergencia garantizada:** el Meta-Teorema 1 dice que el minimizador es único para casi toda $\gamma_0$ (abierto y denso $\mathcal{O}$), no que el descenso en gradiente siempre lo encuentre. Las $\gamma_0$ con $\Delta_{\rm boundary}$ grande no falsifican el teorema: podrían estar en el complemento de $\mathcal{O}$ (que es cerrado y sin interior, pero no vacío).

**Limitación importante:** Este experimento varía la semilla del **dataset** (y por tanto la muestra empírica $\hat{\gamma}_0$), mientras el paper trabaja con la medida de probabilidad verdadera $\gamma_0 \in \mathcal{P}(\mathbb{R}^{d_1})$ (espacio infinito-dimensional). Los resultados son cualitativamente ilustrativos del Meta-Teorema 1, pero no constituyen una verificación formal.

---

## 8. Conclusiones

Los cuatro experimentos proporcionan evidencia empírica consistente con los resultados teóricos de Daudin & Delarue (2025):

| Resultado del paper | Verificación empírica |
|---|---|
| La ODE transforma $\gamma_0$ en $\gamma_T$ separable | Exp. A: lunas → puntos separados, acc=100% |
| $\varepsilon > 0$ concentra el control cerca de $\nu^\infty$ | Exp. B3: std($\theta$) decrece con $\varepsilon$ — comportamiento MAP consistente con la predicción de Gibbs |
| Condición PL: $\|\nabla J\|^2 \geq 2\mu(J-J^*)$ con $\mu > 0$ | Exp. C: ratio PL $> 0$ en todas las épocas |
| Convergencia exponencial bajo PL | Exp. C2: decay lineal en escala log |
| Genericidad del minimizador único (Meta-Teo. 1) | Exp. D: $\Delta_{\rm boundary} \approx 0$ y Std$(J^*) \approx 0$ para $\gamma_0 \in \mathcal{O}$ |

La contribución más importante del paper es la robustez del resultado: **$\varepsilon$ no necesita ser grande** para garantizar la condición PL y la convergencia exponencial. Esto elimina el tradicional dilema entre regularización (que garantiza convergencia pero degrada la solución) y precisión (que da buenas soluciones pero sin garantías). Con cualquier $\varepsilon > 0$, por pequeño que sea, el descenso en gradiente converge exponencialmente al mínimo global.

---

## Referencias

- Daudin, S. & Delarue, F. (2025). *Genericity of the Polyak-Łojasiewicz inequality for mean-field Neural ODEs with entropic regularization.* arXiv:2507.08486.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
- Polyak, B. T. (1963). *Gradient methods for minimizing functionals.* Zh. Vychisl. Mat. Mat. Fiz., 3(4), 643–653.
- Łojasiewicz, S. (1963). *Une propriété topologique des sous-ensembles analytiques réels.* Colloques internationaux du CNRS, 117, 87–89.
- Villani, C. (2003). *Topics in Optimal Transportation.* AMS Graduate Studies in Mathematics, vol. 58.
