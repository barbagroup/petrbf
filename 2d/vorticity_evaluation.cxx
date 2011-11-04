#include <fstream>
#include <iostream>
#include <petscksp.h>

#include "par.h"
#include "get_cluster.h"
#include "get_buffer.h"
#include "get_trunc.h"
#include "get_vorticity.h"

/** RBF gaussian interpolation.
 *
 * Interpolation from source points (xi,yi,wi) for the values (gj) at the evaluation
 * points (xj,yj).
 *
 * Parameters
 * xi, yi:       Coordinates of the evaluation points.
 * wi:           Variable for storing the evaluation.
 * xj, yj:       Coordinates of the source points.
 * gj:           Weight for the source points.
 * sigma:        Parameter of the gaussian.
 * nsigma_box:   Size of inner box, measured in sigma.
 * sigma_buffer: Size of the buffer, measured in sigma.
 * sigma_trunc:  Truncation distance for the gaussians, meassured in sigma.
 */
PetscErrorCode vorticity_evaluation(Vec xi, Vec yi, Vec wi, Vec xj, Vec yj, Vec gj,
  double sigma, int nsigma_box, int sigma_buffer, int sigma_trunc)
{
  int i,*isort,*jsort,ievent[10];
  PetscReal ximin,ximax,yimin,yimax,xjmin,xjmax,yjmin,yjmax;
  std::ofstream fid0,fid1;
  PARTICLE particle;
  CLUSTER cluster;
  MPI2 mpi;
  BOTH both;
  both.p = &particle;
  both.c = &cluster;

  PetscErrorCode ierr;
  IS isx,isy,jsx,jsy;
  VecScatter ctx;

  ierr = PetscLogEventRegister("InitVec",0,&ievent[0]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitCluster",0,&ievent[1]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitIS",0,&ievent[2]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitGhost",0,&ievent[3]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("InitRHS",0,&ievent[4]);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Post Processing",0,&ievent[5]);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(ievent[0],0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpi.nprocs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);CHKERRQ(ierr);
  cluster.file = 0;

  /*
    particle parameters
  */
  particle.sigma = sigma;
  ierr = VecMin(xi,PETSC_NULL,&ximin);CHKERRQ(ierr);
  ierr = VecMax(xi,PETSC_NULL,&ximax);CHKERRQ(ierr);
  ierr = VecMin(yi,PETSC_NULL,&yimin);CHKERRQ(ierr);
  ierr = VecMax(yi,PETSC_NULL,&yimax);CHKERRQ(ierr);
  ierr = VecMin(xj,PETSC_NULL,&xjmin);CHKERRQ(ierr);
  ierr = VecMax(xj,PETSC_NULL,&xjmax);CHKERRQ(ierr);
  ierr = VecMin(yj,PETSC_NULL,&yjmin);CHKERRQ(ierr);
  ierr = VecMax(yj,PETSC_NULL,&yjmax);CHKERRQ(ierr);
  particle.xmin = std::min(ximin,xjmin);
  particle.xmax = std::max(ximax,xjmax);
  particle.ymin = std::min(yimin,yjmin);
  particle.ymax = std::max(yimax,yjmax);

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
  ierr = VecGetSize(xj,&particle.nj);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xi,&particle.ista,&particle.iend);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xj,&particle.jsta,&particle.jend);CHKERRQ(ierr);
  particle.nilocal = particle.iend-particle.ista;
  particle.njlocal = particle.jend-particle.jsta;

  ierr = PetscLogEventEnd(ievent[0],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[1],0,0,0,0);CHKERRQ(ierr);

  /*
    generate clusters
  */
  ierr = VecGetArray(xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecGetArray(yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecGetArray(xj,&particle.xjl);CHKERRQ(ierr);
  ierr = VecGetArray(yj,&particle.yjl);CHKERRQ(ierr);

  Get_cluster clusters;
  clusters.get_cluster(&particle,&cluster);

  ierr = VecRestoreArray(xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecRestoreArray(yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecRestoreArray(xj,&particle.xjl);CHKERRQ(ierr);
  ierr = VecRestoreArray(yj,&particle.yjl);CHKERRQ(ierr);
  isort = new int [particle.nilocal];
  jsort = new int [particle.njlocal];

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
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.i);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.i);CHKERRQ(ierr);
  ierr = VecGetArray(particle.i,&particle.il);CHKERRQ(ierr);
  for(i=0; i<particle.nilocal; i++) {
    isort[i] = (int) particle.il[i];
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,particle.nilocal,isort,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.i,&particle.il);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_WORLD,particle.njlocal,particle.jsta,1,&jsx);CHKERRQ(ierr);
  ierr = ISDuplicate(jsx,&jsy);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&particle.j);CHKERRQ(ierr);
  ierr = VecSetSizes(particle.j,particle.njlocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(particle.j);CHKERRQ(ierr);
  ierr = VecScatterCreate(particle.jj,jsx,particle.j,jsy,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.jj,particle.j,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.jj,particle.j,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = ISDestroy(&jsx);CHKERRQ(ierr);
  ierr = ISDestroy(&jsy);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.j);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.j);CHKERRQ(ierr);
  ierr = VecGetArray(particle.j,&particle.jl);CHKERRQ(ierr);

  for(i=0; i<particle.njlocal; i++) {
    jsort[i] = (int) particle.jl[i];
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,particle.njlocal,jsort,PETSC_COPY_VALUES,&jsx);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.j,&particle.jl);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ievent[2],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[3],0,0,0,0);CHKERRQ(ierr);

  /*
    generate ghost vectors
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&particle.xi);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&particle.xj);CHKERRQ(ierr);
  ierr = VecSetSizes(particle.xi,particle.nilocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetSizes(particle.xj,particle.njlocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(particle.xi);CHKERRQ(ierr);
  ierr = VecSetFromOptions(particle.xj);CHKERRQ(ierr);
  ierr = VecCreateGhost(PETSC_COMM_WORLD,particle.nilocal,PETSC_DECIDE,cluster.nighost,cluster.ighost,&particle.xi);CHKERRQ(ierr);
  ierr = VecCreateGhost(PETSC_COMM_WORLD,particle.njlocal,PETSC_DECIDE,cluster.njghost,cluster.jghost,&particle.xj);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xi,&particle.yi);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xi,&particle.wi);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xj,&particle.yj);CHKERRQ(ierr);
  ierr = VecDuplicate(particle.xj,&particle.gj);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,particle.nilocal,particle.ista,1,&isy);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,particle.njlocal,particle.jsta,1,&jsy);CHKERRQ(ierr);
  ierr = VecScatterCreate(xi,isx,particle.xi,isy,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,xi,particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,xi,particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,yi,particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,yi,particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,wi,particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,wi,particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecScatterCreate(xj,jsx,particle.xj,jsy,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,xj,particle.xj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,xj,particle.xj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,yj,particle.yj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,yj,particle.yj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,gj,particle.gj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,gj,particle.gj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.xi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.xi);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.yi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.yi);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.wi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.wi);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.xj);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.xj);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.yj);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.yj);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(particle.gj);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle.gj);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.xj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.xj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.yj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.yj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle.gj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle.gj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(particle.xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.wi,&particle.wil);CHKERRQ(ierr);
  ierr = VecGetArray(particle.xj,&particle.xjl);CHKERRQ(ierr);
  ierr = VecGetArray(particle.yj,&particle.yjl);CHKERRQ(ierr);
  ierr = VecGetArray(particle.gj,&particle.gjl);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ievent[3],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[4],0,0,0,0);CHKERRQ(ierr);

  /*
    estimate vorticity field on particle from vortex strength
  */
  Get_vorticity vorticity;
  vorticity.get_vorticity(&particle,&cluster);

  ierr = VecRestoreArray(particle.xi,&particle.xil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.yi,&particle.yil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.wi,&particle.wil);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.xj,&particle.xjl);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.yj,&particle.yjl);CHKERRQ(ierr);
  ierr = VecRestoreArray(particle.gj,&particle.gjl);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ievent[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ievent[5],0,0,0,0);CHKERRQ(ierr);

  ierr = VecScatterCreate(particle.xi,isy,xi,isx,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.xi,xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.xi,xi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.yi,yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.yi,yi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.wi,wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.wi,wi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecScatterCreate(particle.xj,jsy,xj,jsx,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.xj,xj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.xj,xj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.yj,yj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.yj,yj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,particle.gj,gj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,particle.gj,gj,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  delete[] isort;
  delete[] jsort;
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
  delete[] cluster.wib;
  delete[] cluster.xjt;
  delete[] cluster.yjt;
  delete[] cluster.gjt;

  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = ISDestroy(&jsx);CHKERRQ(ierr);
  ierr = ISDestroy(&jsy);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.i);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.j);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.ii);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.jj);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.xi);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.yi);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.wi);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.xj);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.yj);CHKERRQ(ierr);
  ierr = VecDestroy(&particle.gj);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ievent[5],0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
