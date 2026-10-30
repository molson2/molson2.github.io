---
layout: post
title: "Epoch-wise double descent in OLS"
date: 2026-07-21
description: "Double descent in plain OLS, traced over training epochs rather than model size, where the implicit bias of gradient descent does the work."
---

I was initially a bit skeptical of double descent in the OLS regression setting. In the usual exposition one plots test error against the number of predictors and gets the second descent right at the $p=n$ interpolation threshold. That always felt like a bit of a sleight of hand: once $p>n$, you have to specify which interpolating solution you're talking about. In practice this means baking in an inductive bias, such as the minimum-norm solution. As it happens, many of the optimization algorithms we already use—gradient descent, for example—produce exactly this bias.

Gradient descent (GD) on OLS feels like a cleaner comparison. The model class is fixed throughout training, and only the amount of optimization changes. If the x-axis in the bias-variance tradeoff is meant to represent "complexity," then the number of GD iterations is a natural candidate: each step reduces training error and progressively removes bias.

This post looks at double descent through that lens. The surprising part is that the phenomenon survives even when "complexity" is measured by optimization time rather than model size. In the end, the picture becomes quite simple: gradient descent turns on the spectrum one eigendirection at a time, and double descent is just what happens when those directions alternate between carrying more signal than noise and less. What looks mysterious in parameter space becomes almost inevitable in spectral coordinates.

## Setup

Take the linear model $y = X\beta^\star + \varepsilon$, with $\mathbb{E}[\varepsilon] = 0$, $\operatorname{Cov}(\varepsilon) = \sigma_\varepsilon^2 I$, and the thin SVD

$$
X = U\Sigma V^\top, \qquad \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_p), \quad \sigma_1 \ge \dots \ge \sigma_p > 0.
$$

Here $V$ is a generic rotation, so $X$ has correlated, non-orthogonal columns—nothing is diagonal in the coordinate basis. Everything below is diagonal in $U$ instead.

## Gradient descent is a spectral filter

Run GD on $\tfrac12\|y - X\beta\|^2$ from $\beta^{(0)} = 0$ with step $\eta$. In the $U$ basis the iteration decouples, and the fitted values after $t$ steps are

$$
\hat y^{(t)} = U\, g_t(\Sigma^2)\, U^\top y,
\qquad
g_t(\sigma^2) = 1-(1-\eta\sigma^2)^t
$$

The filter $g_t$ climbs from $0$ to $1$ in each direction, fastest at the top of the spectrum: **direction $i$ turns on at $t \sim 1/(\eta\sigma_i^2)$.** Stability requires $\eta < 2/\sigma_1^2$. (Equivalently, GD at step $t$ behaves like ridge regression with $\lambda \sim 1/(\eta t)$.)

## Deriving the Risk of Gradient Descent

Take in-sample prediction risk against the noiseless signal. With $\theta_i = u_i^\top X\beta^\star$ the signal along $u_i$, the risk splits into independent per-direction terms (appendix):

$$
R(t) = \frac1n\,\mathbb{E}\big\|\hat y^{(t)} - X\beta^\star\big\|^2 = \frac1n\sum_i \Big[\underbrace{(1-g_t(\sigma_i^2))^2\,\theta_i^2}_{\text{bias}^2,\ \downarrow} + \underbrace{g_t(\sigma_i^2)^2\,\sigma_\varepsilon^2}_{\text{variance},\ \uparrow}\Big].
$$

At the extremes:

$$
R(0) = \frac1n\sum_i \theta_i^2, \qquad R(\infty) = \frac{p\,\sigma_\varepsilon^2}{n}.
$$

Both pieces are monotone in $t$. Therefore, the shape of the risk curve is determined entirely by the *order* in which the directions switch on.

As the number of training rounds increases, the filter $g_t$ approaches 1 for directions ordered from largest to smallest singular values. As each direction turns on, its contribution to the risk shifts from signal $\theta_i^2$ to noise $\sigma_\varepsilon^2$, producing a net change of:

$$
\Delta_i \;=\; \frac{\sigma_\varepsilon^2 - \theta_i^2}{n}.
$$

Therefore, $R(t)$ is simply a running cumulative sum of $\Delta_i$ walked down the spectrum. Double descent is nothing more than traversing regions of the spectrum where the signal alternates between exceeding and falling below the noise level. In other words, each eigendirection is either worth learning ($\theta_i^2 > \sigma_\varepsilon^2$) or not ($\theta_i^2 < \sigma_\varepsilon^2$). Gradient descent simply decides *when* each of those bets gets placed.

## A minimal example

To see this numerically, we can construct an example where the spectrum is divided into three distinct regions (A, B, and C):
* **Block A:** Clear, fast signal.
* **Block B:** Pure noise—400 directions with zero signal, representing pure noise-fitting.
* **Block C:** A slow block carrying strong signal, separated by $100\times$ gaps in $\sigma^2$ to prevent phase overlap.

Using three blocks with $n = 2000$, $\sigma_\varepsilon^2 = 1$, and $\eta = 0.1/\sigma_1^2$:

| block | $k$ | $\sigma^2$ | $\theta^2/\sigma_\varepsilon^2$ | $\sum\Delta$ | turns on |
|---|---|---|---|---|---|
| A | 50 | $2000$ | $10$ | $-0.225$ | $t\approx 10$ |
| B | 400 | $20$ | $0$ | $+0.200$ | $t\approx 10^{3}$ |
| C | 50 | $0.2$ | $12$ | $-0.275$ | $t\approx 10^{5}$ |

Individually, bias and variance behave exactly as expected: more rounds of GD reduce bias and increase variance. However, their interaction creates a funky double-descent shape.

![double descent]({{ "/assets/posts/gd-dd-blog.png" | relative_url }})

## Some extra food for thought

Modern statistical learning increasingly treats the optimizer as a first-class citizen. Especially in overparameterized settings—or whenever early stopping is used—the optimizer is not merely a computational tool, it is part of the estimator itself.

In spectral coordinates this becomes almost unavoidable. OLS, Ridge, PCR, early-stopped gradient descent, momentum methods, and many other algorithms are all spectral estimators: they differ only in the filters they apply across the eigenspectrum. Which estimator is "best" therefore depends not on some abstract notion of complexity, but on how the signal is distributed over that spectrum.

Thinking this way also changes how I view the bias-variance tradeoff. Complexity is not just about reducing bias. Different optimization procedures reveal different spectral modes at different rates, and the resulting risk depends on whether those modes carry signal or only noise. From that perspective, double descent is not a mysterious exception to the classical story—it is simply one of the natural behaviors that emerge when learning unfolds spectrally

---

## Appendix: derivations

### Risk decomposition

Let $z_i = u_i^\top\varepsilon$, so $\mathbb{E}[z_i] = 0$ and $\operatorname{Var}(z_i) = \sigma_\varepsilon^2$. The data along $u_i$ is $u_i^\top y = \theta_i + z_i$, the fit keeps $g_i(\theta_i + z_i)$ with $g_i = g_t(\sigma_i^2)$, and the target is $\theta_i$. Since $U$ is orthonormal and $X\beta^\star$ lies in its span,

$$
R(t) = \frac1n\sum_i \mathbb{E}\big[g_i(\theta_i + z_i) - \theta_i\big]^2.
$$

Split each error into a deterministic and a mean-zero part:

$$
g_i(\theta_i + z_i) - \theta_i = (g_i - 1)\theta_i + g_i z_i.
$$

The cross term vanishes since $\mathbb{E}[z_i] = 0$, leaving $(1-g_i)^2\theta_i^2 + g_i^2\sigma_\varepsilon^2$ per direction. At $t = 0$ every $g_i = 0$; as $t \to \infty$ every $g_i \to 1$, which gives the two extremes $R(0)$ and $R(\infty)$.
