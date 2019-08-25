---
layout: post
title:  Making Sense of Random Forest Probabilities, a Kernel Perspective
coauthor: Abraham J. Wyner
subtitle:
---

A random forest is a popular tool for estimating probabilities in machine learning classification tasks. However, the means by which this is accomplished is unprincipled: one simply counts the fraction of trees in a forest that vote for a certain class. In this paper, we forge a connection between random forests and kernel regression. This places random forest probability estimation on more sound statistical footing. As part of our investigation, we develop a model for the proximity kernel and relate it to the geometry and sparsity of the estimation problem. We also provide intuition and recommendations for tuning a random forest to improve its probability estimates.

[paper](/assets/rf_probability.pdf) 
