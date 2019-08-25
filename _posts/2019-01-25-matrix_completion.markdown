---
layout: post
title: A More Efficient Approach to Large Scale Matrix Completion Problems
subtitle:
---

This paper investigates a scalable optimization procedure to the low-rank matrix completion problem posed by Candes and Recht [2]. We identify the singular value decomposition as a computational bot- tleneck for large problem instances, and propose utilizing an approxi- mately computed SVD borne out of recent advances in random linear algebra. We then use this approximately computed SVD to implement some popular first order methods for solving the matrix completion problem. Our results indicate that these modified routines can easily handle very large data sets. In particular, we are able to recover 35 million entries from a 104 × 104 matrix in a matter of minutes.

[paper](/assets/matrix_completion.pdf)
