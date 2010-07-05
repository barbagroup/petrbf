#ifndef _GETCLUSTER_CLASS
#define _GETCLUSTER_CLASS

#include "par.h"

extern void mpi_range(MPI2*);

class Get_cluster
{
  int n,ic,id,io,ip,ix,iy,ista,iend,ix_cluster,iy_cluster,j,jc,jd,jsta,jend,jx,jy,jx_min,jx_max,jy_min,jy_max;
  int icall,ncall,nilocal,njlocal,*iplocal,*jplocal,*ipglobal,*jpglobal,*ipoffset,*jpoffset,*idghost;
  double sort;
  MPI2 mpi;
public:
  void get_cluster(PARTICLE *particle,CLUSTER *cluster)
  {
    MPI_Comm_size(PETSC_COMM_WORLD,&mpi.nprocs);
    MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);

    cluster->neighbor_buffer = (int) ceil((cluster->sigma_buffer - cluster->nsigma_box + epsf) / 2 / cluster->nsigma_box);
    cluster->neighbor_trunc  = (int) ceil((cluster->sigma_trunc - cluster->nsigma_box + epsf) / 2 / cluster->nsigma_box);
    cluster->neighbor_ghost  = std::max(cluster->neighbor_buffer, cluster->neighbor_trunc);

    // Calculate cluster size
    cluster->xmin       = particle->xmin-epsf;
    cluster->xmax       = particle->xmax+epsf;
    cluster->ymin       = particle->ymin-epsf;
    cluster->ymax       = particle->ymax+epsf;
    cluster->box_length = cluster->nsigma_box*particle->sigma+epsf;

    // Calculate number of clusters in each direction
    cluster->nx = (int)ceil((cluster->xmax - cluster->xmin)/cluster->box_length);
    cluster->ny = (int)ceil((cluster->ymax - cluster->ymin)/cluster->box_length);
    cluster->n  = cluster->nx*cluster->ny;

    
    // Allocate arrays
    cluster->ista = new int [cluster->n];
    cluster->iend = new int [cluster->n];
    cluster->jsta = new int [cluster->n];
    cluster->jend = new int [cluster->n];
    cluster->ix   = new int [cluster->n];
    cluster->iy   = new int [cluster->n];
    cluster->xc   = new double [cluster->n];
    cluster->yc   = new double [cluster->n];

    iplocal  = new int [cluster->n];
    jplocal  = new int [cluster->n];
    ipglobal = new int [cluster->n];
    jpglobal = new int [cluster->n];
    ipoffset = new int [cluster->n];
    jpoffset = new int [cluster->n];
    idghost  = new int [cluster->n];

    
    // Calculate the x, y index and coordinates of the center
    ic = -1;
    for (ix = 0; ix < cluster->nx; ix++) {
      for (iy = 0; iy < cluster->ny; iy++) {
        ic++;
        cluster->ix[ic]   = ix;
        cluster->iy[ic]   = iy;
        cluster->xc[ic]   = cluster->xmin+(ix+0.5)*cluster->box_length;
        cluster->yc[ic]   = cluster->ymin+(iy+0.5)*cluster->box_length;
        cluster->ista[ic] = 0;
        cluster->iend[ic] = -1;
        cluster->jsta[ic] = 0;
        cluster->jend[ic] = -1;
        iplocal[ic]       = 0;
        jplocal[ic]       = 0;
        ipoffset[ic]      = 0;
        jpoffset[ic]      = 0;
      }
    }

    /*
      assign cluster number to particles
    */
    for (ip = 0; ip < particle->nilocal; ip++) {
      ix_cluster = (int)floor((particle->xil[ip] - cluster->xmin) / cluster->box_length);
      iy_cluster = (int)floor((particle->yil[ip] - cluster->ymin) / cluster->box_length);
      ic         = ix_cluster * cluster->ny + iy_cluster;
      iplocal[ic]++;
    }
    for (ip = 0; ip < particle->njlocal; ip++) {
      ix_cluster = (int)floor((particle->xjl[ip] - cluster->xmin) / cluster->box_length);
      iy_cluster = (int)floor((particle->yjl[ip] - cluster->ymin) / cluster->box_length);
      ic         = ix_cluster * cluster->ny + iy_cluster;
      jplocal[ic]++;
    }

    /*
      communicate and find global box offset (cluster->ista) and local box offset (ipoffset)
    */
    MPI_Exscan(iplocal, ipoffset, cluster->n, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Exscan(jplocal, jpoffset, cluster->n, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(iplocal, ipglobal, cluster->n, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(jplocal, jpglobal, cluster->n, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    id = 0;
    jd = 0;
    for (ic = 0; ic < cluster->n; ic++) {
      ipoffset[ic]      += id;
      jpoffset[ic]      += jd;
      cluster->ista[ic]  = id;
      cluster->jsta[ic]  = jd;
      id                += ipglobal[ic];
      jd                += jpglobal[ic];
      cluster->iend[ic]  = id-1;
      cluster->jend[ic]  = jd-1;
    }

    mpi.nsta = 0;
    mpi.nend = cluster->n-1;
    mpi_range(&mpi);
    cluster->icsta = mpi.ista;
    cluster->icend = mpi.iend;

    VecCreate(PETSC_COMM_WORLD,&particle->ii);
    VecCreate(PETSC_COMM_WORLD,&particle->jj);
    VecSetSizes(particle->ii,particle->nilocal,PETSC_DETERMINE);
    VecSetSizes(particle->jj,particle->njlocal,PETSC_DETERMINE);
    VecSetFromOptions(particle->ii);
    VecSetFromOptions(particle->jj);

    for (ip = 0; ip < particle->nilocal; ip++) {
      ix_cluster = (int)floor((particle->xil[ip] - cluster->xmin) / cluster->box_length);
      iy_cluster = (int)floor((particle->yil[ip] - cluster->ymin) / cluster->box_length);
      ic         = ix_cluster * cluster->ny + iy_cluster;
      sort       = ip + particle->ista;
      VecSetValues(particle->ii,1,&ipoffset[ic],&sort,INSERT_VALUES);
      ipoffset[ic]++;
    }
    for (ip = 0; ip < particle->njlocal; ip++) {
      ix_cluster = (int)floor((particle->xjl[ip] - cluster->xmin) / cluster->box_length);
      iy_cluster = (int)floor((particle->yjl[ip] - cluster->ymin) / cluster->box_length);
      ic         = ix_cluster * cluster->ny + iy_cluster;
      sort       = ip + particle->jsta;
      VecSetValues(particle->jj, 1, &jpoffset[ic], &sort, INSERT_VALUES);
      jpoffset[ic]++;
    }
    VecAssemblyBegin(particle->ii);
    VecAssemblyEnd(particle->ii);
    VecAssemblyBegin(particle->jj);
    VecAssemblyEnd(particle->jj);

    particle->ista    = cluster->ista[cluster->icsta];
    particle->jsta    = cluster->jsta[cluster->icsta];
    particle->iend    = cluster->iend[cluster->icend-1]+1;
    particle->jend    = cluster->jend[cluster->icend-1]+1;
    particle->nilocal = particle->iend-particle->ista;
    particle->njlocal = particle->jend-particle->jsta;

   /*
     determine size and create buffer & trunc temp arrays
   */
    cluster->niperbox = 0;
    cluster->njperbox = 0;
    for (ic = 0; ic < cluster->n; ic++) {
      if(cluster->iend[ic] - cluster->ista[ic] + 1 > cluster->niperbox) {
        cluster->niperbox = cluster->iend[ic] - cluster->ista[ic] + 1;
      }
      if(cluster->jend[ic] - cluster->jsta[ic] + 1 > cluster->njperbox) {
        cluster->njperbox = cluster->jend[ic] - cluster->jsta[ic] + 1;
      }
    }

    cluster->maxbuffer = cluster->niperbox * (2 * cluster->neighbor_buffer + 1) * (2 * cluster->neighbor_buffer + 1);
    cluster->maxtrunc  = cluster->njperbox * (2 * cluster->neighbor_trunc + 1) * (2 * cluster->neighbor_trunc + 1);

    for (ic = 0; ic < cluster->n; ic++) {
      idghost[ic] = 0;
    }
    for (ic = cluster->icsta; ic < cluster->icend; ic++) {
      idghost[ic] = 1;
    }
    cluster->ncghost = 0;
    for (ic = cluster->icsta; ic < cluster->icend; ic++) {
      ix     = cluster->ix[ic];
      iy     = cluster->iy[ic];
      jx_min = std::max(0, ix - cluster->neighbor_ghost);
      jx_max = std::min(cluster->nx - 1, ix + cluster->neighbor_ghost);
      jy_min = std::max(0, iy - cluster->neighbor_ghost);
      jy_max = std::min(cluster->ny - 1, iy + cluster->neighbor_ghost);
      for (jx = jx_min; jx <= jx_max; jx++) {
        for (jy = jy_min; jy <= jy_max; jy++) {
          jc = jx * cluster->ny + jy;
          if (idghost[jc] == 0) {
            idghost[jc] = 2;
            cluster->ncghost++;
          }
        }
      }
    }
    cluster->nclocal  = cluster->icend - cluster->icsta;
    cluster->maxghost = std::max(cluster->niperbox, cluster->njperbox) * cluster->ncghost;
    cluster->maxlocal = std::max(cluster->niperbox, cluster->njperbox) * (cluster->nclocal + cluster->ncghost);
    cluster->ighost   = new int [cluster->maxghost];
    cluster->ilocal   = new int [cluster->maxlocal];
    cluster->jghost   = new int [cluster->maxghost];
    cluster->jlocal   = new int [cluster->maxlocal];

    /*
      local cluster indexing
    */
    nilocal = 0;
    for (ic = 0; ic < cluster->n; ic++) {
      if (idghost[ic] == 1) {
        ista              = cluster->ista[ic];
        iend              = cluster->iend[ic];
        cluster->ista[ic] = nilocal;
        for (j = ista; j <= iend; j++) {
          cluster->ilocal[nilocal] = j;
          nilocal++;
        }
        cluster->iend[ic] = nilocal-1;
      }
    }
    cluster->nighost = 0;
    for (ic = 0; ic < cluster->n; ic++) {
      if (idghost[ic] == 2) {
        ista              = cluster->ista[ic];
        iend              = cluster->iend[ic];
        cluster->ista[ic] = nilocal;
        for (j = ista; j <= iend; j++) {
          cluster->ighost[cluster->nighost] = j;
          cluster->nighost++;
          cluster->ilocal[nilocal] = j;
          nilocal++;
        }
        cluster->iend[ic] = nilocal-1;
      }
    }
    njlocal = 0;
    for (ic = 0; ic < cluster->n; ic++) {
      if (idghost[ic] == 1) {
        jsta              = cluster->jsta[ic];
        jend              = cluster->jend[ic];
        cluster->jsta[ic] = njlocal;
        for (j = jsta; j <= jend; j++) {
          cluster->jlocal[njlocal] = j;
          njlocal++;
        }
        cluster->jend[ic] = njlocal-1;
      }
    }
    cluster->njghost = 0;
    for (ic=0; ic<cluster->n; ic++) {
      if (idghost[ic] == 2) {
        jsta              = cluster->jsta[ic];
        jend              = cluster->jend[ic];
        cluster->jsta[ic] = njlocal;
        for (j = jsta; j <= jend; j++) {
          cluster->jghost[cluster->njghost] = j;
          cluster->njghost++;
          cluster->jlocal[njlocal] = j;
          njlocal++;
        }
        cluster->jend[ic] = njlocal-1;
      }
    }

    delete[] iplocal;
    delete[] jplocal;
    delete[] ipglobal;
    delete[] jpglobal;
    delete[] ipoffset;
    delete[] jpoffset;
    delete[] idghost;

    cluster->idx = new int [cluster->maxbuffer];
    cluster->xib = new double [cluster->maxbuffer];
    cluster->yib = new double [cluster->maxbuffer];
    cluster->gib = new double [cluster->maxbuffer];
    cluster->eib = new double [cluster->maxbuffer];
    cluster->wib = new double [cluster->maxbuffer];
    cluster->xjt = new double [cluster->maxtrunc];
    cluster->yjt = new double [cluster->maxtrunc];
    cluster->gjt = new double [cluster->maxtrunc];
  }
};

#endif

