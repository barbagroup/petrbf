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
  int i,its,nsigma_box,sigma_buffer,sigma_trunc,nx,ny,nxg,nyg,ni,nj,ng,ista,iend,istag,iendg,nlocal;
  double sigma,overlap,h,hg,xmin,xmax,ymin,ymax,xd,yd,gd,ed,wd,t,err,errd;
  clock_t tic,toc;
  tic = std::clock();

  std::ofstream fid0,fid1;
  PARAMETER parameter;
  MPI2 mpi;

  PetscErrorCode ierr;
  PetscViewer viewer;
  Vec x,y,g,e,w,xg,yg,wg,we;

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  MPI_Comm_size(PETSC_COMM_WORLD,&mpi.nprocs);
  MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);

  /*
    physical parameters
  */
  parameter.vis = 0.01;
  parameter.t = 1;

  /*
    particle parameters
  */
  h = 0.02;
  overlap = atof(argv[1]);
  sigma = h/overlap;
  hg = h/2;
  xmin = -4;
  xmax = 4;
  ymin = -4;
  ymax = 4;

  /*
    cluster parameters
  */
  nsigma_box = atoi(argv[2]);
  sigma_buffer = (int)nsigma_box*atof(argv[3]);
  if (overlap < 0.8+epsf) {
    sigma_trunc = nsigma_box+16;
  } else {
    sigma_trunc = nsigma_box+14;
  }

  /*
    calculate problem size
  */
  nx = (int)floor((xmax-xmin+epsf)/h)+1;
  ny = (int)floor((ymax-ymin+epsf)/h)+1;
  nxg = (int)floor((xmax-xmin+epsf)/hg)+1;
  nyg = (int)floor((ymax-ymin+epsf)/hg)+1;
  ni = nx*ny;
  ng = nxg*nyg;
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
  ierr = VecDuplicate(x,&g);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&e);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&xg);CHKERRQ(ierr);
  ierr = VecSetSizes(xg,PETSC_DECIDE,ng);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xg);CHKERRQ(ierr);
  ierr = VecDuplicate(xg,&yg);CHKERRQ(ierr);
  ierr = VecDuplicate(xg,&wg);CHKERRQ(ierr);
  ierr = VecDuplicate(xg,&we);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&ista,&iend);CHKERRQ(ierr);
  nlocal = iend-ista;
  for(i=ista; i<iend; i++) {
    xd = xmin+floor(i/ny)*h;
    yd = ymin+(i%ny)*h;
    ed = 0.75*exp(-((9*xd-2)*(9*xd-2)+(9*yd-2)*(9*yd))/4)
      +0.75*exp(-((9*xd+1)*(9*xd+1))/49-((9*yd+1)*(9*yd+1))/10)
      +0.5*exp(-((9*xd-7)*(9*xd-7)+(9*yd-3)*(9*yd-3))/4)
      -0.2*exp(-(9*xd-4)*(9*xd-4)-(9*yd-7)*(9*yd-7));
//    ed = exp(-(xd*xd+yd*yd)/(4*parameter.vis*parameter.t))/(M_PI*4*parameter.vis*parameter.t);
    wd = ed;
    gd = ed*h*h;
    ierr = VecSetValues(x,1,&i,&xd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(y,1,&i,&yd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(g,1,&i,&gd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(e,1,&i,&ed,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(w,1,&i,&wd,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecGetOwnershipRange(xg,&istag,&iendg);CHKERRQ(ierr);
  for(i=istag; i<iendg; i++) {
    xd = xmin+floor(i/nyg)*hg;
    yd = ymin+(i%nyg)*hg;
    wd = 0.75*exp(-((9*xd-2)*(9*xd-2)+(9*yd-2)*(9*yd))/4)
      +0.75*exp(-((9*xd+1)*(9*xd+1))/49-((9*yd+1)*(9*yd+1))/10)
      +0.5*exp(-((9*xd-7)*(9*xd-7)+(9*yd-3)*(9*yd-3))/4)
      -0.2*exp(-(9*xd-4)*(9*xd-4)-(9*yd-7)*(9*yd-7));
//    wd = exp(-(xd*xd+yd*yd)/(4*parameter.vis*parameter.t))/(M_PI*4*parameter.vis*parameter.t);
    ierr = VecSetValues(xg,1,&i,&xd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(yg,1,&i,&yd,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(we,1,&i,&wd,INSERT_VALUES);CHKERRQ(ierr);
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
  ierr = VecAssemblyBegin(xg);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xg);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(yg);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(yg);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(we);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(we);CHKERRQ(ierr);

//  vorticity_evaluation(x,y,w,x,y,g,sigma,nsigma_box,sigma_buffer,sigma_trunc);
  rbf_interpolation(x,y,g,e,sigma,nsigma_box,sigma_buffer,sigma_trunc,&its);
  vorticity_evaluation(xg,yg,wg,x,y,g,sigma,nsigma_box,sigma_buffer,sigma_trunc);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"we.dat",&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = VecView(we,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"wg.dat",&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = VecView(wg,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  /*
    calculate the L2 norm error
  */
  ierr = VecAXPY(wg,-1,we);CHKERRQ(ierr);
  ierr = VecNorm(we,NORM_2,&err);CHKERRQ(ierr);
  ierr = VecNorm(wg,NORM_2,&errd);CHKERRQ(ierr);
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
