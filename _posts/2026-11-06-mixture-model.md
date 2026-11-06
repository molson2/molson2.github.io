---
layout: post
title: "Almost Amortized Bayes"
date: 2026-11-06
description: "Transformers can approximate amortized Bayes in nonobvious ways."
---

A popular view of transformers is that they perform amortized Bayesian inference (Müller et al., 2022; Reuter et al., 2025). A training set specifies a prior, and an autoregressive model trained on it learns to approximate the posterior predictive. In small models this often holds quite literally. Train a small transformer on sequences from random Markov chains and it builds bigram counts in context, an identifiable internal program that implements the Bayes rule (see my [induction heads post]({% post_url 2026-07-30-induction-heads %})).

This post grew out of a small TabPFN-style model I trained on a simple prior. It matched the posterior predictive in distribution but failed under small perturbations, because the posterior needed a count that the architecture couldn't compute. **Getting close to the posterior predictive behaviorally does not guarantee the model is computing it.**

Here I distill that lesson into a standard transformer and a simple problem, a mixture of two Gaussians, where the Bayes rule depends on how many observations you have seen. A transformer without positional encoding learns the best prediction it can make from the running mean, which matches Bayes closely in loss but is a different function. Adding a beginning-of-sequence token gives it a count, and including the first token in the loss asks for a discontinuous rule, which the network can only approximate.

## Setup

The problem we'll study is sequence prediction from a mixture of Gaussians. For each sequence we flip a fair coin to choose one of two sources, $A$ or $B$, and then draw 32 independent observations from it:

$$x_s \sim \begin{cases}
\mathcal N(-\mu, 1) & \text{under } A, \\
\mathcal N(+\mu, 1) & \text{under } B,
\end{cases}
\qquad \mu = 0.3.$$

The sources overlap heavily, so a single token says little, but evidence builds up as tokens arrive. Each token contributes a log-likelihood ratio of $-2\mu x$, and tokens are independent given the source, so the ideal log-odds of $A$ after $t$ tokens is

$$\ell^\star_{\text{Bayes}} = -2\mu S_t, \qquad S_t = \sum_{s \le t} x_s.$$

**The Bayes rule depends on the sum rather than the mean.**

The model is a one-layer transformer trained on next-token log-loss. After each token it outputs a log-odds $\ell$ for $A$ and predicts the next token as the mixture $\sigma(\ell)\,\mathcal N(-\mu, 1) + (1 - \sigma(\ell))\,\mathcal N(+\mu, 1)$[^1][^training]. Log-loss is minimized by the posterior predictive, which here is exactly such a mixture with weight $P(A \mid \text{context})$. So a model that learns Bayes outputs $\ell^\star_{\text{Bayes}}$, and a model that sees only part of the context should output the posterior log-odds given that part. We train two versions, neither with positional encoding (the no-BOS version is trained both with and without the first-token loss, as discussed below). The first sees only the tokens, so nothing in its input tells it where it is in the sequence. The second adds a beginning-of-sequence (BOS) token, which gives it a way to recover the count.

[^1]: Fixing the output to a mixture of the two known sources means the model only has to learn its belief, not the shape of the sources. This makes "the model's belief" a single number we can compare directly with Bayes.

[^training]: Width 32, MLP width 128. Adam with learning rate decayed linearly to zero, batch 512. The BOS model and the no-BOS model trained without the first-token loss use learning rate $10^{-3}$ for 25k steps; the no-BOS model trained on all positions uses $3 \times 10^{-4}$ for 100k steps.

## Counting with attention

Without positional encoding, a token's key and value depend only on its value, so a single attention layer computes a softmax-weighted average over the tokens in context. Repeating the whole context leaves that average unchanged, so the no-BOS model cannot tell a context from the same context repeated, even though Bayes treats them differently.

In practice the trained no-BOS model gives every token the same score, so each gets weight $1/t$ and the head outputs a function of the running mean $\bar x_t$. The current token also reaches the MLP through the residual connection, so the MLP sees two numbers, $\bar x_t$ and $x_t$, and nothing else.

The BOS token changes this because it appears exactly once regardless of context length. If BOS gets a score $b$ above the ordinary tokens and the ordinary tokens all score the same, the weight on BOS after $t$ tokens is

$$w_t = \frac{e^b}{e^b + t} = \frac{1}{1 + e^{-b}\,t}.$$

The weight falls as tokens arrive, so it encodes $t$, and the model only has to learn a single constant $b$ to get it.

![Attention patterns]({{ "/assets/posts/mixture-model-attention.png" | relative_url }})

*Left: attention from the last position. Both no-BOS models (with and without the first-token loss) attend uniformly, and the BOS model attends uniformly over ordinary tokens with the remaining weight on BOS. Right: the weight on BOS at each query position, with the fitted $w_t$.*

In the trained model the BOS weight follows this curve with $R^2 = 1.0000$ and $e^{-b} = 0.333$, and the ordinary tokens share the remaining weight equally. The head therefore outputs the BOS value weighted by $w_t$ and the running mean weighted by $1 - w_t$, and both $t$ and $S_t = t\bar x_t$ can be recovered from these. The information Bayes needs reaches the MLP.

## Bayes without the count

The two models now face different problems. The BOS model has the count, so it can compute the Bayes rule $-2\mu S_t$ and we can check whether it does. The trained no-BOS model attends uniformly, so its MLP sees the running mean,[^uniform] and the best it can do with that is the posterior given the mean, treating the hidden $t$ as a latent variable with prior $\pi(t)$:

$$P(A \mid \bar x) \propto \sum_t \pi(t)\, p(\bar x \mid t, A).$$

[^uniform]: Uniform attention is not forced by the architecture. Since a token's key depends on its value, the head could instead compute a tilted mean, weighting tokens by $e^{a x_s}$. A tilted mean picks up a little information about the spread of the tokens, and the sample spread carries a little information about $t$. The gain from this is tiny, and the trained model doesn't bother: it attends uniformly, and given uniform attention it learns the best rule available from the running mean.

Since $\bar x_t \sim \mathcal N(\mp\mu, 1/t)$, the best log-odds from the mean are

$$g^\star(\bar x) = \log \sum_t \sqrt t\; e^{-t(\bar x + \mu)^2/2} \;-\; \log \sum_t \sqrt t\; e^{-t(\bar x - \mu)^2/2},$$

with $t$ uniform over the positions in the loss, $2, \dots, 31$. We leave $t = 1$ out of the loss for now and put it back in the next section. The model also sees the current token, which changes this rule only slightly (Appendix A).

![Real contexts against the best rule]({{ "/assets/posts/mixture-model-curves.png" | relative_url }})

*Each dot is a prediction on a held-out context. Left: the no-BOS model against the running mean, with the best rule from the mean dashed. Right: the BOS model against the running sum, with Bayes dashed.*

Each model learns the rule available to it. The BOS model follows Bayes. The no-BOS model follows $g^\star$, which is a function of the mean instead of the sum and hedges over every context length that could have produced that mean.

The no-BOS model's beliefs are wrong in a systematic way, overconfident early in the sequence and underconfident late, but the loss barely registers it. The model captures 94% of the improvement over ignoring the context and finishes $1.55 \times 10^{-3}$ nats per token above Bayes. The sources overlap heavily, so quite different beliefs produce similar next-token distributions. A model can be nearly optimal in loss while holding a belief that is computed from the wrong statistic.

## Recognizing the first token

Now put the prediction after the first token back into the no-BOS model's loss. Even without positional information, the model can recognize the first token. The MLP sees $\bar x_t$ and $x_t$, and these two numbers identify the first position because $\bar x_1 = x_1$, while for two or more tokens drawn from continuous distributions the two are equal with probability zero. So given what the MLP sees, the best rule takes one of two answers: the one-token Bayes answer $-2\mu\bar x$ on the line $x_t = \bar x$, and the running-mean rule $g^\star(\bar x)$ from the previous section everywhere off it (Appendix B). The switch between the two has zero width.

A ReLU network is continuous, so it cannot represent a zero-width jump. It can react when $x_t$ is close to $\bar x$, but then it also reacts to later positions whose current token happens to land near the mean.

![The t = 1 model]({{ "/assets/posts/mixture-model-t1.png" | relative_url }})

*Left: held-out contexts, colored by how close the current token is to the running mean. The 92% with $\lvert x_t - \bar x\rvert \geq 0.1$ sit on the running-mean rule; the 8% near the diagonal are pulled toward the one-token answer. Right: the blend weight $\lambda$ as the current token slides through the mean, at four fixed means.*

Since the best rule only ever gives one of those two answers, we measure where the network sits between them with $\lambda$:

$$\ell = \lambda\,(-2\mu\,\bar x) + (1 - \lambda)\,g^\star(\bar x),$$

so $\lambda = 0$ is the running-mean answer and $\lambda = 1$ is the one-token answer. The best rule has $\lambda = 1$ exactly on the diagonal and $\lambda = 0$ everywhere else. The network learns a tent instead, peaking at roughly 0.5 (depending on the mean) with a half-width of roughly 0.045. The tent helps on true first tokens and makes later predictions near the diagonal underconfident. A taller or narrower tent trades one error against the other, and training settles on a compromise.

The cost at later positions is small: at $t \geq 2$ this model is $1.62 \times 10^{-3}$ nats above Bayes, against $1.55 \times 10^{-3}$ for the model trained without the first token.

## Final thoughts

The one-dimensional problem is the simplest version of this story. In higher dimensions it gets more interesting, because the spread of the tokens around their running mean, $\lVert x_t - \bar x_t\rVert$, carries information about how many tokens there are. A model without positional encoding can use that to get closer to Bayes, and what it learns is harder to pin down.

I find in-context learning both obvious and a little magical. A transformer learns procedures, and when you train it on data where the posterior predictive is optimal, it often learns something very close to it. But a good loss does not tell you which procedure the model learned, so behavioral analysis alone is thin evidence. Architectural and optimization biases can stand between the model and the rule you had in mind, and counting is the example here. The no-BOS model did exactly what amortized Bayes predicts for the information it had. That information was less than we assumed, and the loss alone would never have told us.

*Müller, Samuel, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. "Transformers Can Do Bayesian Inference." ICLR, 2022.*

*Reuter, Arik, Tim G. J. Rudner, Vincent Fortuin, and David Rügamer. "Can Transformers Learn Full Bayesian Inference in Context?" ICML, 2025.*

## Appendix

### A. The best rule from the running mean

Without BOS the position $t$ is hidden, so we treat it as a latent variable with prior $\pi(t)$, uniform over the positions in the loss ($2, \dots, 31$ for the $t=1$-dropped model). Marginalizing,

$$P(A \mid \phi) \propto \sum_t \pi(t)\, p(\phi \mid t, A).$$

**Running mean only.** Given $t$ and the source, $\bar x_t \sim \mathcal N(\mp\mu, 1/t)$. Dropping constants shared by $A$ and $B$,

$$g^\star(\bar x) = \log \sum_t \sqrt t\; e^{-t(\bar x + \mu)^2/2} \;-\; \log \sum_t \sqrt t\; e^{-t(\bar x - \mu)^2/2}.$$

Near $\bar x = 0$ the large-$t$ terms dominate and $g^\star$ is steep.

**Running mean and current token.** Write the other $t - 1$ tokens' mean as $m = (t\bar x - x_t)/(t-1)$. Given $t$ and the source, $x_t \sim \mathcal N(\mp\mu, 1)$ and $m \sim \mathcal N(\mp\mu, 1/(t-1))$ independently, so changing variables from $(m, x_t)$ to $(\bar x, x_t)$,

$$p(\bar x, x_t \mid t, A) \propto \frac{t}{t-1}\,\sqrt{t-1}\;\exp\!\Big(-\frac{(x_t + \mu)^2}{2} - \frac{(t-1)(m + \mu)^2}{2}\Big),$$

and similarly for $B$. Summing over $t$ as above gives $g_2(\bar x, x_t)$. At $t \geq 2$ it differs from $g^\star(\bar x)$ only slightly; the figures use $g_2$ as the reference.

### B. The jump at $x_t = \bar x$

Now include $t = 1$ in the prior. At $t = 1$ the pair $(\bar x, x_t)$ always lies on the diagonal, so its distribution is $\mathcal N(x_1; \mp\mu, 1)$ along the line $x_t = \bar x$ and zero elsewhere: a point mass in the direction across the diagonal. For $t \ge 2$ the pair has an ordinary density, which assigns the diagonal probability zero.

So off the diagonal the $t = 1$ term contributes nothing and the best rule is $g_2(\bar x, x_t)$. On the diagonal the $t = 1$ term dominates any finite density, the posterior is certain that $t = 1$, and the best rule is the one-token answer $-2\mu\,\bar x$:

$$\ell^\star(\bar x, x_t) = \begin{cases} -2\mu\, \bar x & x_t = \bar x, \\ g_2(\bar x, x_t) & \text{otherwise.} \end{cases}$$

This is the zero-width jump. Any continuous network must replace it with something of finite width, which in practice is the tent.

