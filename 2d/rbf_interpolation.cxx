#include <fstream>
#include <petscksp.h>

#include "par.h"
#include "mpi_range.h"
#include "get_cluster.h"
#include "get_buffer.h"
#include "get_trunc.h"
#include "matmult.h"

/** RBF solver.
 *
 * Using collocation, it finds the weights (gi) for a set of RBF gaussian bases (xi, yi).
 *
 * Parameters:
 * xi, yi:       Coordinates of the gaussian bases.
 * gi:           Returns the solved weights for the gaussian bases.
 * ei:           Estimation of the field
 * wi:           Solution of the field at the bases locations.
 * sigma:        Sigma parameter of the gaussian.
 * nsigma_box:   Size of the inner box or 'local box'.
 * sigma buffer: Size of the buffer area.
 * sigma_trunc:  Truncation point for sigma.
 * its:          Returns solver teration data.
 */
PetscErrorCode rbf_interpolation(Vec xi, Vec yi, Vec gi, Vec ei, Vec wi,
  double sigma, int nsigma_box, int sigma_buffer, int sigma_trunc, int *its)
{
  int i,ic,id,ista,iend,*isort,ievent[10];
  std::ofstream fid0,fid1;
  PARTICLE particle;
  CLUSTER cluster;
  MPI2 mpi;
  BOTH both;
  both.p = &particle;
  both.c = &cluster;

  PetscErrorCode ierr;
  KSP ksp;
  PC pc;
  IS isx,isy,*is,*is_local;
  Mat M,P;
  Vec xx;
  PetscInt *idx;
  PetscScalar *xxx;
  VecScatter ctx;

  ierr = PetscLogEventRegister("InitVec",0,&ievent[0]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitCluster",0,&ievent[1]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitIS",0,&ievent[2]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitGhost",0,&ievent[3]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitMat",0,&ievent[4]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Post Processing",0,&ievent[5]);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(ievent[0],0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpi.nprocs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);CHKERRQ(ierr);
  cluster.file = 0;

  /*
    particle parameters
  */
  particle.sigma = sigma;
  ierr = VecMin(xi,&particle.xmin);CHKERRQ(ierr);
  ierr = VecMax(xi,&particle.xmax);CHKERRQ(ierr);
  ierr = VecMin(yi,&particle.ymin);CHKERRQ(ierr);
  ierr = VecMax(yi,&particle.ymax);CHKERRQ(ierr);

  /*
    cluster parameters
  */
  cluster.nsigma_box = nsigma_box;
  cluster.sigma_buffer = sigma_buffer;
  cluster.sigma_trunc = sigma_trunc;

  /*
    calculate problem size
  */
  ierr = VecGetSize(xi,&particle.ni);CHKERRQ(ierr);
  ierr = VecGetSize(xi,&particle.nj);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xi,&particle.ista,&particle.iend);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xi,&particle.jsta,&particle.jend);CHKERRQ(ierr);
  particle.nilocal = particle.iend-particle.ista;
  particle.njlocal = particle.jend-particle.jsta;

  ierr = PetscLogEventEnd(ievent[0],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[1],0,0,0,0);CHKERRQ(ierr);

  /*
    generate clusters
  */
  ierr = VecGetArray(xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecGetArray(yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecGetArray(xi,&particle.xjl);CHKERRQ(ierr);
  ierr = VecGetArray(yi,&particle.yjl);CHKERRQ(ierr);

  Get_cluster clusters;
  clusters.get_cluster(&particle,&cluster);

  ierr = VecRestoreArray(xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecRestoreArray(yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecRestoreArray(xi,&particle.xjl);CHKERRQ(ierr);
  ierr = VecRestoreArray(yi,&particle.yjl);CHKERRQ(ierr);
  isort = new int [particle.nilocal];

  ierr = PetscLogEventEnd(ievent[1],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[2],0,0,0,0);CHKERRQ(ierr);

  /*
    generate IS
  */
  ierr = ISCreateStride(PETSC_COMM_WORLD,particle.nilocal,particle.ista,1,&isx);CHKERRQ(ierr);
  ierr = ISDuplicate(isx,&isy);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&particle.i);CHKERRQ(ierr);
  ierr = VecSetSizes(particle.i,particle.nilocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(particle.i);CHKERRQ(ierr);
  ierr = VecScatterCreate(particle.ii,isx,particle.i,isy,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.ii,particle.i,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.ii,particle.i,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  ierr = ISDestroy(isx);CHKERRQ(ierr);
  ierr = ISDestroy(isy);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.i);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.i);CHKERRQ(ierr);
  ierr = VecGetArray(particle.i,&particle.il);CHKERRQ(ierr);
  for(i=0; i<particle.nilocal; i++) {
    isort[i] = particle.il[i];
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,particle.nilocal,isort,&isx);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.i,&particle.il);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ievent[2],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[3],0,0,0,0);CHKERRQ(ierr);

  /*
    generate ghost vectors
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&particle.xi);CHKERRQ(ierr);
  ierr = VecSetSizes(particle.xi,particle.nilocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(particle.xi);CHKERRQ(ierr);
  ierr = VecCreateGhost(PETSC_COMM_WORLD,particle.nilocal,PETSC_DECIDE,cluster.nighost,cluster.ighost,
    &particle.xi);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xi,&particle.yi);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xi,&particle.gi);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xi,&particle.ei);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xi,&particle.wi);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,particle.nilocal,particle.ista,1,&isy);CHKERRQ(ierr);
  ierr = VecScatterCreate(xi,isx,particle.xi,isy,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,xi,particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,xi,particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,yi,particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,yi,particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,gi,particle.gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,gi,particle.gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,ei,particle.ei,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,ei,particle.ei,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,wi,particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,wi,particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.xi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.xi);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.yi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.yi);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.gi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.gi);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.ei);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.ei);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.wi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.wi);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.ei,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.ei,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ievent[3],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[4],0,0,0,0);CHKERRQ(ierr);

  /*
    RBF interpolation
  */
  is = new IS [particle.nilocal];
  is_local = new IS [particle.nilocal];
  idx = new PetscInt [cluster.maxbuffer];

  ierr = VecCreate(PETSC_COMM_WORLD,&xx);CHKERRQ(ierr);
  ierr = VecSetSizes(xx,particle.nilocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,particle.nilocal,particle.nilocal,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(M,MATSHELL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(M);CHKERRQ(ierr);
  ierr = MatShellSetOperation(M,MATOP_MULT, (void (*)(void)) mymatmult);
  ierr = MatView(M,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatShellSetContext(M,&both);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr);
  ierr = MatSetSizes(P,particle.nilocal,particle.nilocal,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(P,MATSHELL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(P);CHKERRQ(ierr);
  ierr = MatShellSetOperation(P,MATOP_GET_SUBMATRICES, (void (*)(void)) mysubmat);
  ierr = MatShellSetContext(P,&both);CHKERRQ(ierr);

  ierr = VecGetArray(particle.xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.gi,&particle.gil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.ei,&particle.eil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.wi,&particle.wil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.xi,&particle.xjl);CHKERRQ(ierr);
  ierr = VecGetArray(particle.yi,&particle.yjl);CHKERRQ(ierr);
  ierr = VecGetArray(particle.gi,&particle.gjl);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCASMSetOverlap(pc,0);CHKERRQ(ierr);
  for (ic=cluster.icsta; ic<cluster.icend; ic++) {
    id = ic-cluster.icsta;
    Get_buffer buffer;
    buffer.get_buffer(&particle,&cluster,ic);
    ista = cluster.ista[ic];
    iend = cluster.iend[ic];
    if (ista <= iend) {
      for (i=0; i<cluster.npbufferi; i++) {
        idx[i] = cluster.idx[i];
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,cluster.npbufferi,idx,&is[id]);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,iend-ista+1,idx,&is_local[id]);CHKERRQ(ierr);
  }
  ierr = PCASMSetSortIndices(pc,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PCASMSetLocalSubdomains(pc,cluster.nclocal,is,is_local);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,M,P,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ievent[4],0,0,0,0);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,particle.ei,xx);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[5],0,0,0,0);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,its);CHKERRQ(ierr);

  ierr = VecRestoreArray(particle.xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.gi,&particle.gil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.ei,&particle.eil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.wi,&particle.wil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.xi,&particle.xjl);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.yi,&particle.yjl);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.gi,&particle.gjl);CHKERRQ(ierr);

  cluster.icsta = 0;
  cluster.icend = cluster.nclocal;
  ierr = VecGetArray(xx,&xxx);CHKERRQ(ierr);
  for(i=particle.ista; i<particle.iend; i++) {
    ierr = VecSetValues(particle.gi,1,&i,&xxx[i-particle.ista],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&xxx);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.gi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.gi);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterCreate(particle.xi,isy,xi,isx,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.xi,xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.xi,xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.yi,yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.yi,yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.gi,gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.gi,gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.ei,ei,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.ei,ei,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.wi,wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.wi,wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);

  for (ic=0; ic<cluster.nclocal; ic++) {
    ierr = ISDestroy(is[ic]);CHKERRQ(ierr);
    ierr = ISDestroy(is_local[ic]);CHKERRQ(ierr);
  }
  delete[] idx;
  delete[] is;
  delete[] is_local;
  delete[] isort;
  delete[] cluster.ista;
  delete[] cluster.iend;
  delete[] cluster.jsta;
  delete[] cluster.jend;
  delete[] cluster.ix;
  delete[] cluster.iy;
  delete[] cluster.xc;
  delete[] cluster.yc;
  delete[] cluster.ighost;
  delete[] cluster.jghost;
  delete[] cluster.ilocal;
  delete[] cluster.jlocal;
  delete[] cluster.idx;
  delete[] cluster.xib;
  delete[] cluster.yib;
  delete[] cluster.gib;
  delete[] cluster.eib;
  delete[] cluster.wib;
  delete[] cluster.xjt;
  delete[] cluster.yjt;
  delete[] cluster.gjt;

  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = ISDestroy(isx);CHKERRQ(ierr);
  ierr = ISDestroy(isy);CHKERRQ(ierr);
  ierr = VecDestroy(xx);CHKERRQ(ierr);
  ierr = VecDestroy(particle.i);CHKERRQ(ierr);
  ierr = VecDestroy(particle.ii);CHKERRQ(ierr);
  ierr = VecDestroy(particle.xi);CHKERRQ(ierr);
  ierr = VecDestroy(particle.yi);CHKERRQ(ierr);
  ierr = VecDestroy(particle.gi);CHKERRQ(ierr);
  ierr = VecDestroy(particle.ei);CHKERRQ(ierr);
  ierr = VecDestroy(particle.wi);CHKERRQ(ierr);
  ierr = MatDestroy(M);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ievent[5],0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
