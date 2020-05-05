# BayesianDPMixturevMF
This package implements a few Bayesian inference algorithms for von Mises Fisher based mixture models 

* MvMF_inference.m: a Gibbs sampler for finite mixture of vMFs
* iMvMF_inference.m: a Gibbs sampler for infinite mixture or Dirichlet Process mixture of vMFs (Chinese restaurant process representation)
* Collapsed_iMvMF_inference.m: a Gibbs sampler for infinite mixture or Dirichlet Process mixture of vMFs with the mean vectors integrated 
* Collapsed_iMCIvMF_inference.m: a Gibbs sampler for infinite mixture of conditionally independent vMFs with the mean vectors integrated 

To get started, see demo.mlx

I will rewrite the whole program and upload it soon ...

Copyright (C) 2018 Lei Fang, lf28 at st-andrews.ac.uk; Free to distribute, modify, adapt etc. 

Please cite the following paper if needed (also for more details)
L. Fang, J. Ye and S. Dobson, "Sensor-Based Human Activity Mining Using Dirichlet 
Process Mixtures of Directional Statistical Models," 2019 IEEE International Conference on 
Data Science and Advanced Analytics (DSAA), Washington, DC, USA, 2019, pp. 154-163.
