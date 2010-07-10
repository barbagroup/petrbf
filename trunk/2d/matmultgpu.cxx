#ifndef _MYMATMULT_FUNC
#define _MYMATMULT_FUNC

#include <fstream>
#include "par.h"
#include "get_buffer.h"
#include "get_trunc.h"

extern void gpumatmult(double*, double*, double*, double*, double*, double*, int*, int, double);

PetscErrorCode mymatmult(Mat A,Vec x,Vec y)
{
  int i,j,ic,il,ista,iend;
  int iblok,isize,im,jc,*offset;
  double *targetX,*targetY,*targetW,*sourceX,*sourceY,*sourceG;
  double dx,dy,w;
  PetscScalar *ax,*ay;
  PetscErrorCode ierr;
  BOTH *both;
  ierr = MatShellGetContext(A, (void **) &both);CHKERRQ(ierr);
  PARTICLE *particle = both->p;
  CLUSTER *cluster = both->c;

  offset = new int [cluster->n+1];
  targetX = new double [cluster->n*threadsPerBlock];
  targetY = new double [cluster->n*threadsPerBlock];
  targetW = new double [cluster->n*threadsPerBlock];
  sourceX = new double [cluster->n*cluster->maxtrunc];
  sourceY = new double [cluster->n*cluster->maxtrunc];
  sourceG = new double [cluster->n*cluster->maxtrunc];
  PetscFunctionBegin;
  ierr = VecGetArray(x,&ax);CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay);CHKERRQ(ierr);
  for (i=particle->ista; i<particle->iend; i++) {
    ierr = VecSetValues(particle->gi,1,&i,&ax[i-particle->ista],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(particle->gi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle->gi);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle->gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle->gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(particle->gi,&particle->gil);CHKERRQ(ierr);
  iblok = 0;
  jc = 0;
  for (ic=cluster->icsta; ic<cluster->icend; ic++) {
    Get_trunc trunc;
    trunc.get_trunc(particle,cluster,ic);
    ista = cluster->ista[ic];
    iend = cluster->iend[ic];
    isize = iend-ista+1;
    for (i=0; i<isize; i++) {
      im = iblok*threadsPerBlock+i;
      targetX[im] = particle->xil[i+ista];
      targetY[im] = particle->yil[i+ista];
    }
    for (i=isize; i<threadsPerBlock; i++) {
      im = iblok*threadsPerBlock+i;
      targetX[im] = 0;
      targetY[im] = 0;
    }
    offset[iblok] = jc;
    for (j=0; j<cluster->nptruncj; j++) {
      sourceX[jc] = cluster->xjt[j];
      sourceY[jc] = cluster->yjt[j];
      sourceG[jc] = cluster->gjt[j];
      jc++;
    }
    iblok++;
  }
  offset[iblok] = jc;

  gpumatmult(targetX,targetY,targetW,sourceX,sourceY,sourceG,offset,iblok,particle->sigma);

  iblok = 0;
  for (ic=cluster->icsta; ic<cluster->icend; ic++) {
    Get_trunc trunc;
    trunc.get_trunc(particle,cluster,ic);
    ista = cluster->ista[ic];
    iend = cluster->iend[ic];
    isize = iend-ista+1;
    for (i=0; i<isize; i++) {
      il = cluster->ilocal[i+ista];
      im = iblok*threadsPerBlock+i;
      ay[il-particle->ista] = targetW[im];
    }
    iblok++;
  }
  ierr = VecRestoreArray(particle->gi,&particle->gil);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&ax);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&ay);CHKERRQ(ierr);
  delete[] offset;
  delete[] targetX;
  delete[] targetY;
  delete[] targetW;
  delete[] sourceX;
  delete[] sourceY;
  delete[] sourceG;
  PetscFunctionReturn(0);
}

PetscErrorCode mysubmat(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  int i,ic,id,j,ista,iend;
  double dx,dy;
  PetscInt *idx;
  PetscScalar *A;
  PetscErrorCode ierr;
  BOTH *both;
  ierr = MatShellGetContext(mat, (void **) &both);CHKERRQ(ierr);
  PARTICLE *particle = both->p;
  CLUSTER *cluster = both->c;

  idx = new PetscInt [cluster->maxbuffer];
  A = new PetscScalar [cluster->maxbuffer*cluster->maxbuffer];

  PetscFunctionBegin;
  ierr = PetscMalloc(n * sizeof(Mat*), submat);CHKERRQ(ierr);
  ierr = VecGetArray(particle->gi,&particle->gil);CHKERRQ(ierr);
  for (ic = cluster->icsta; ic < cluster->icend; ic++) {
    id = ic-cluster->icsta;
    Get_buffer buffer;
    buffer.get_buffer(particle,cluster,ic);
    ierr = MatCreate(PETSC_COMM_SELF,&(*submat)[id]);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix((*submat)[id], "sub_");CHKERRQ(ierr);
    ierr = MatSetSizes((*submat)[id],cluster->npbufferi,cluster->npbufferi,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr
);
    ierr = MatSetFromOptions((*submat)[id]);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation((*submat)[id],cluster->npbufferi,PETSC_NULL);CHKERRQ(ierr);
    ista = cluster->ista[ic];
    iend = cluster->iend[ic];
    if (ista <= iend) {
      for (i=0; i<cluster->npbufferi; i++) {
        for (j=0; j<cluster->npbufferi; j++) {
          dx = cluster->xib[i]-cluster->xib[j];
          dy = cluster->yib[i]-cluster->yib[j];
          A[i*cluster->npbufferi+j] = exp(-(dx*dx+dy*dy)/(2*particle->sigma*particle->sigma))/
            (2*M_PI*particle->sigma*particle->sigma);
        }
        idx[i] = i;
      }
    }
    ierr = MatSetValues((*submat)[id],cluster->npbufferi,idx,cluster->npbufferi,idx,A,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin((*submat)[id],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd((*submat)[id],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(particle->gi,&particle->gil);CHKERRQ(ierr);
  delete[] A;
  delete[] idx;

  PetscFunctionReturn(0);
}

#endif

