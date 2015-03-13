## PetRBF---A parallel O(N) algorithm for radial basis function interpolation ##

Continuing with our previous efforts to provide the scientific community with fast algorithms for radial basis functions, this code project provides a full parallel implementation of the method used in [pyrbf](http://code.google.com/p/pyrbf/).


**September 2009** -- at this time, we release a stable version of the test code, used to produce the first paper and scalability results.  A future release will offer a more streamlined interface.

|We distribute this code under the MIT License, giving potential users the greatest freedom possible. We do, however, request fellow scientists that if they use our codes in research, they kindly include us in the acknowledgement of their papers.  We do not request gratuitous citations;  only cite our articles if you deem it warranted.|
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Visit the [group's webpage](http://barbagroup.bu.edu/) for more codes.

### Publications ###

  * _"PetRBF—A parallel O(N) algorithm for radial basis function interpolation"_  by Rio Yokota, L A Barba, Matthew G Knepley.

> Preprint uploaded to [ArXiv](http://arxiv.org/abs/0909.5413v1) on 29 September 2009.
> Published as: _Comput. Meth. Appl. Mech. Eng._, **199**(25–28): 1793–1804 (May 2010)  http://dx.doi.org/10.1016/j.cma.2010.02.008

> We have developed a parallel algorithm for radial basis function (RBF) interpolation that exhibits O(N) complexity,requires O(N) storage, and scales excellently up to a thousand processes. The algorithm uses a GMRES iterative solver with a restricted additive Schwarz method (RASM) as a preconditioner and a fast matrix-vector algorithm. Previous fast RBF methods, --,achieving at most O(NlogN) complexity,--, were developed using multiquadric and polyharmonic basis functions. In contrast, the present method uses Gaussians with a small variance (a common choice in particle methods for fluid simulation, our main target application). The fast decay of the Gaussian basis function allows rapid convergence of the iterative solver even when the subdomains in the RASM are very small. The present method was implemented in parallel using the PETSc library (developer version). Numerical experiments demonstrate its capability in problems of RBF interpolation with more than 50 million data points, timing at 106 seconds (19 iterations for an error tolerance of 10^-15 on 1024 processors of a Blue Gene/L (700 MHz PowerPC processors). The parallel code is freely available in the open-source model.