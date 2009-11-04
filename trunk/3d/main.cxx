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
  int i,its,nsigma_box,sigma_buffer,sigma_trunc,nx,ny,nz,ni,nj,ista,iend,nlocal;
  double sigma,overlap,h,xmin,xmax,ymin,ymax,zmin,zmax,xd,yd,zd,gd,ed,wd,t,err,errd;
  clock_t tic,toc;
  tic = std::clock();

  std::ofstream fid0,fid1;
  PARAMETER parameter;
  MPI2 mpi;

  PetscErrorCode ierr;
  Vec x,y,z,g,e,w;

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  MPI_Comm_size(PETSC_COMM_WORLD,&mpi.nprocs);
  MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);

  /*
    physical parameters
  */
  parameter.vis = 0.1;
  parameter.t = 1;

  /*
    particle parameters
  */
  sigma = 0.1;
  overlap = atof(argv[1]);
  h = overlap*sigma;
  xmin = -1;
  xmax = 1;
  ymin = -1;
  ymax = 1;
  zmin = -1;
  zmax = 1;

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
  nx = (int)floor((xmax-xmin+epsf)/h)+1;
  ny = (int)floor((ymax-ymin+epsf)/h)+1;
  nz = (int)floor((zmax-zmin+epsf)/h)+1;
  ni = nx*ny*nz;
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
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,ni);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&z);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&g);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&e);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&ista,&iend);CHKERRQ(ierr);
  nlocal = iend-ista;
  for(i=ista; i<iend; i++) {
    xd = xmin+floor((i%(nx*ny))/ny)*h;
    yd = ymin+(i%ny)*h;
    zd = zmin+floor(i/(nx*ny))*h;
//    ed = (float)rand()/RAND_MAX;
//    xd = xd*nx/(nx+1)+ed*h/2;
//    ed = (float)rand()/RAND_MAX;
//    yd = yd*ny/(ny+1)+ed*h/2;
    ed = exp(-(xd*xd+yd*yd+zd*zd)/(4*parameter.vis*parameter.t))/(M_PI*4*parameter.vis*parameter.t);
    wd = ed;
    gd = ed*h*h;
    ierr = VecSetValues(x,1,&i,&xd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(y,1,&i,&yd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(z,1,&i,&zd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(g,1,&i,&gd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(e,1,&i,&ed,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(w,1,&i,&wd,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(z);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(g);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(g);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(e);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(e);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(w);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(w);CHKERRQ(ierr);

  vorticity_evaluation(x,y,z,w,x,y,z,g,sigma,nsigma_box,sigma_buffer,sigma_trunc,&its);
  rbf_interpolation(x,y,z,g,e,w,sigma,nsigma_box,sigma_buffer,sigma_trunc,&its);
  vorticity_evaluation(x,y,z,w,x,y,z,g,sigma,nsigma_box,sigma_buffer,sigma_trunc,&its);

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
    if (1.0-overlap < epsf) {
//      sprintf(file,"%s-%s-%s.dat",argv[1],argv[2],argv[3]);
    } else {
//      sprintf(file,"0%s-%s-%s.dat",argv[1],argv[2],argv[3]);
    }
    sprintf(file,"%d.dat",mpi.nprocs);
    fid0.open(file);
    fid0 << t << std::endl << its;
    fid0.close();
    printf("error : %g\n",err);
    printf("time  : %g\n",t);
  }

  PetscFinalize();
}
