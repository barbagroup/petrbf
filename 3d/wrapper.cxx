#include <mpi.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <petscksp.h>
#include <petscis.h>
#include <petsclog.h>

#include "par.h"
#include "get_cluster.h"
#include "get_buffer.h"
#include "get_trunc.h"
#include "get_vorticity.h"

extern PetscErrorCode vorticity_evaluation(Vec,Vec,Vec,Vec,Vec,Vec,Vec,Vec,double,int,int,int);
extern PetscErrorCode rbf_interpolation(Vec,Vec,Vec,Vec,Vec,double,int,int,int,int*);

int RBFInterpolation(int argc, char **argv,
                     int ng, float *xg, float *yg, float *zg, float *wg,
                     int np, float *xp, float *yp, float *zp, float *gp,
                     float sigma, int nsigma_box, int sigma_buffer, int sigma_trunc)
{
  int i,its,ni,nj,ista,iend,jsta,jend;

  MPI2 mpi;

  PetscErrorCode ierr;
  PetscScalar *xid,*yid,*zid,*wid,*gid,*xjd,*yjd,*zjd,*gjd;
  Vec xi,yi,zi,wi,gi,xj,yj,zj,gj;

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  MPI_Comm_size(PETSC_COMM_WORLD,&mpi.nprocs);
  MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);

  /*
    calculate problem size
  */
  ni = ng;
  nj = np;

  /*
    generate particles
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&xi);CHKERRQ(ierr);
  ierr = VecSetSizes(xi,PETSC_DECIDE,ni);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xi);CHKERRQ(ierr);
  ierr = VecDuplicate(xi,&yi);CHKERRQ(ierr);
  ierr = VecDuplicate(xi,&zi);CHKERRQ(ierr);
  ierr = VecDuplicate(xi,&gi);CHKERRQ(ierr);
  ierr = VecDuplicate(xi,&wi);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xi,&ista,&iend);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&xj);CHKERRQ(ierr);
  ierr = VecSetSizes(xj,PETSC_DECIDE,nj);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xj);CHKERRQ(ierr);
  ierr = VecDuplicate(xj,&yj);CHKERRQ(ierr);
  ierr = VecDuplicate(xj,&zj);CHKERRQ(ierr);
  ierr = VecDuplicate(xj,&gj);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xj,&jsta,&jend);CHKERRQ(ierr);
  ierr = VecGetArray(xi,&xid);CHKERRQ(ierr);
  ierr = VecGetArray(yi,&yid);CHKERRQ(ierr);
  ierr = VecGetArray(zi,&zid);CHKERRQ(ierr);
  ierr = VecGetArray(wi,&wid);CHKERRQ(ierr);
  ierr = VecGetArray(gi,&gid);CHKERRQ(ierr);
  ierr = VecGetArray(xj,&xjd);CHKERRQ(ierr);
  ierr = VecGetArray(yj,&yjd);CHKERRQ(ierr);
  ierr = VecGetArray(zj,&zjd);CHKERRQ(ierr);
  ierr = VecGetArray(gj,&gjd);CHKERRQ(ierr);
  for(i=ista; i<iend; i++) {
    xid[i-ista] = xg[i-ista];
    yid[i-ista] = yg[i-ista];
    zid[i-ista] = zg[i-ista];
    gid[i-ista] = wg[i-ista];
    wid[i-ista] = wg[i-ista];
  }
  for(i=jsta; i<jend; i++) {
    xjd[i-jsta] = xp[i-jsta];
    yjd[i-jsta] = yp[i-jsta];
    zjd[i-jsta] = zp[i-jsta];
    gjd[i-jsta] = gp[i-jsta];
  }
  ierr = VecRestoreArray(xi,&xid);CHKERRQ(ierr);
  ierr = VecRestoreArray(yi,&yid);CHKERRQ(ierr);
  ierr = VecRestoreArray(zi,&zid);CHKERRQ(ierr);
  ierr = VecRestoreArray(wi,&wid);CHKERRQ(ierr);
  ierr = VecRestoreArray(gi,&gid);CHKERRQ(ierr);
  ierr = VecRestoreArray(xj,&xjd);CHKERRQ(ierr);
  ierr = VecRestoreArray(yj,&yjd);CHKERRQ(ierr);
  ierr = VecRestoreArray(zj,&zjd);CHKERRQ(ierr);
  ierr = VecRestoreArray(gj,&gjd);CHKERRQ(ierr);

  if( fabs(xg[0] - xp[0]) > epsf ) {
    vorticity_evaluation(xi,yi,zi,wi,xj,yj,zj,gj,sigma,nsigma_box,sigma_buffer,sigma_trunc);
  }
  rbf_interpolation(xi,yi,zi,gi,wi,sigma,nsigma_box,sigma_buffer,sigma_trunc,&its);

  ierr = VecGetArray(gi,&gid);CHKERRQ(ierr);
  for(i=ista; i<iend; i++) {
    wg[i-ista] = gid[i-ista];
  }
  ierr = VecRestoreArray(gi,&gid);CHKERRQ(ierr);

  return ierr;
}
