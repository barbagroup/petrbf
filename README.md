# PetRBF
Many applications in computational science need to approximate a function based on finite data. When the data are in
a certain sense “scattered” in their domain, one very powerful technique is radial basis function (RBF) interpolation.
For many years, the wide applicability of RBF interpolation was hindered by its numerical difficulty and expense.
Indeed, in their mathematical expression, RBF methods produce an ill-conditioned linear system, for which a direct
solution becomes prohibitive for more than a few thousand data points.

We have developed a parallel algorithm for RBF interpolation that exhibits O(N) complexity, requires O(N) storage,
and scales excellently up to a thousand processes. The algorithm uses the GMRES iterative solver with a restricted
additive Schwarz method (RASM) as a preconditioner and a fast matrix-vector algorithm. Previous fast RBF methods
— achieving at most O(N log N) complexity — were developed using multiquadric and polyharmonic basis functions. In
contrast, the present method uses Gaussians with a small variance. The fast decay of the Gaussian basis function
allows rapid convergence of the iterative solver even when the subdomains in the RASM are very small. The method was
implemented in parallel using the PETSc library (developer version). Numerical experiments demonstrate its capability
in problems of RBF interpolation with more than 50 million data points, timing at 106 seconds (19 iterations for an
error tolerance of 10e−15) on 1024 processors of a Blue Gene/L (700 MHz PowerPC processors).

See the paper [PetRBF--A parallel O(N) algorithm for radial basis function interpolation](http://arxiv.org/abs/0909.5413)
