#ifndef _MYMATMULT_FUNC
#define _MYMATMULT_FUNC

PetscErrorCode mymatmult(Mat A,Vec x,Vec y)
{
  int i,j,ic,il,ista,iend;
  double dx,dy,w;
  PetscScalar *ax,*ay;
  PetscErrorCode ierr;
  BOTH *both;
  ierr = MatShellGetContext(A, (void **) &both);CHKERRQ(ierr);
  PARTICLE *particle = both->p;
  CLUSTER *cluster = both->c;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&ax);CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay);CHKERRQ(ierr);
  for(i=particle->ista; i<particle->iend; i++) {
    ierr = VecSetValues(particle->gi,1,&i,&ax[i-particle->ista],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(particle->gi);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(particle->gi);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(particle->gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(particle->gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(particle->gi,&particle->gil);CHKERRQ(ierr);
  for (ic=cluster->icsta; ic<cluster->icend; ic++) {
    Get_trunc trunc;
    trunc.get_trunc(particle,cluster,ic);
    ista = cluster->ista[ic];
    iend = cluster->iend[ic];
    for (i=ista; i<=iend; i++) {
      il = cluster->ilocal[i];
      w = 0;
      for (j=0; j<cluster->nptruncj; j++) {
        dx = particle->xil[i]-cluster->xjt[j];
        dy = particle->yil[i]-cluster->yjt[j];
        w += cluster->gjt[j]*exp(-(dx*dx+dy*dy)/(2*particle->sigma*particle->sigma))/
          (2*M_PI*particle->sigma*particle->sigma);
      }
      ay[il-particle->ista] = w;
    }
    /* Counted 1 for exp() */
    ierr = PetscLogFlops((iend-ista)*cluster->nptruncj*15);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(particle->gi,&particle->gil);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&ax);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&ay);CHKERRQ(ierr);
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
  if (scall == MAT_REUSE_MATRIX) {SETERRQ(PETSC_ERR_SUP, "Cannot handle submatrix reuse yet");}
  ierr = PetscMalloc(n * sizeof(Mat*), submat);CHKERRQ(ierr);
  ierr = VecGetArray(particle->gi,&particle->gil);CHKERRQ(ierr);
  for(ic = cluster->icsta; ic < cluster->icend; ic++) {
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