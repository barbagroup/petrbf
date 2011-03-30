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
#include "mpi_range.h"
#include "get_cluster.h"
#include "get_buffer.h"
#include "get_trunc.h"
#include "get_vorticity.h"
#include "matmult.h"

#include "vorticity_evaluation.cxx"
#include "rbf_interpolation.cxx"

int main(int argc,char **argv)
{
  int i,its,nsigma_box,sigma_buffer,sigma_trunc,ni,nj,ista,iend,nlocal;
  double sigma,overlap,h,*xd,*yd,*gd,*ed,*wd,t,err,errd;
  clock_t tic,toc;
  tic = std::clock();

  std::ifstream fid;
  std::ofstream fid0,fid1;
  MPI2 mpi;

  PetscErrorCode ierr;
  Vec x,y,g,e,w;

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  MPI_Comm_size(PETSC_COMM_WORLD,&mpi.nprocs);
  MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);

  /*
    particle parameters
  */
  sigma = 0.007;
  overlap = atof(argv[1]);
  h = overlap*sigma;

  /*
    cluster parameters
  */
  nsigma_box = atoi(argv[2]);
  sigma_buffer = (int)nsigma_box*atof(argv[3]);
  if (overlap < 0.8+epsf) {
    sigma_trunc = nsigma_box+6;
  } else {
    sigma_trunc = nsigma_box+4;
  }

  /*
    calculate problem size
  */
  ni = 5346;
  if(mpi.myrank==0) {
    printf("||---------------------------------------\n");
    printf("|| number of particles        : %d      \n",ni);
    printf("|| std of Gaussian (sigma)    : %f      \n",sigma);
    printf("|| overlap ratio (h/sigma)    : %f      \n",overlap);
    printf("|| non-overlapping subdomain  : %d sigma\n",nsigma_box);
    printf("|| overlapping subdomain      : %d sigma\n",(int)fmin(sigma_buffer,floor(2/sigma)));
    printf("|| entire domain              : %d sigma\n",(int)floor(2/sigma));
    printf("||---------------------------------------\n");
  }
  nj = ni;

  /*
    generate particles
  */
  xd = new double [ni];
  yd = new double [ni];
  gd = new double [ni];
  ed = new double [ni];
  wd = new double [ni];
  fid.open("cdata");
  for (i=0; i<ni; i++) {
    fid >> xd[i];
    fid >> yd[i];
    fid >> gd[i];
    fid >> ed[i];
    wd[i] = ed[i];
  }
  fid.close();
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,ni);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&g);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&e);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&ista,&iend);CHKERRQ(ierr);
  nlocal = iend-ista;
  for(i=ista; i<iend; i++) {
    ierr = VecSetValues(x,1,&i,&xd[i],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(y,1,&i,&yd[i],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(g,1,&i,&gd[i],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(e,1,&i,&ed[i],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(w,1,&i,&wd[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(g);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(g);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(e);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(e);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(w);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(w);CHKERRQ(ierr);

//  vorticity_evaluation(x,y,w,x,y,g,sigma,nsigma_box,sigma_buffer,sigma_trunc);
  rbf_interpolation(x,y,g,e,sigma,nsigma_box,sigma_buffer,sigma_trunc,&its);
  vorticity_evaluation(x,y,w,x,y,g,sigma,nsigma_box,sigma_buffer,sigma_trunc);

  /*
    calculate the L2 norm error
  */
  ierr = VecAXPY(w,-1,e);CHKERRQ(ierr);
  ierr = VecNorm(e,NORM_2,&err);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&errd);CHKERRQ(ierr);
  err = log(errd/err)/log(10.0);

  toc = std::clock();
  t = (double)(toc-tic)/ (double)CLOCKS_PER_SEC;
  if (mpi.myrank == 0) {
    char file[13];
    sprintf(file,"%d.dat",mpi.nprocs);
    fid0.open(file);
    fid0 << t << std::endl << its;
    fid0.close();
    printf("error : %g\n",err);
    printf("time  : %g\n",t);
  }

  delete[] xd;
  delete[] yd;
  delete[] gd;
  delete[] ed;
  delete[] wd;
  PetscFinalize();
}
