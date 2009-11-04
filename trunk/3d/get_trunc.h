class Get_trunc
{
  int i,il,ista,iend,ix,iy,iz,j,jc,jsta,jend,jx,jy,jz,jx_min,jx_max,jy_min,jy_max,jz_min,jz_max;
  double xc,yc,zc,xi,yi,zi,ei,wi,xj,yj,zj,gj;
public:
  void get_trunc(PARTICLE *particle, CLUSTER *cluster, int ic)
  {
    MPI2 mpi;
    MPI_Comm_rank(PETSC_COMM_WORLD,&mpi.myrank);
 
    cluster->trunc_length = cluster->sigma_trunc*particle->sigma/2+epsf;

    ista = cluster->ista[ic];
    iend = cluster->iend[ic];
    if (ista <= iend) {
      xc = cluster->xc[ic];
      yc = cluster->yc[ic];
      zc = cluster->zc[ic];
      ix = cluster->ix[ic];
      iy = cluster->iy[ic];
      iz = cluster->iz[ic];
      jx_min = std::max(0,ix-cluster->neighbor_trunc);
      jx_max = std::min(cluster->nx-1,ix+cluster->neighbor_trunc);
      jy_min = std::max(0,iy-cluster->neighbor_trunc);
      jy_max = std::min(cluster->ny-1,iy+cluster->neighbor_trunc);
      jz_min = std::max(0,iz-cluster->neighbor_trunc);
      jz_max = std::min(cluster->nz-1,iz+cluster->neighbor_trunc);

  /*
    put all particles in the center box into the corresponding cell structure
  */
      jsta = cluster->jsta[ic];
      jend = cluster->jend[ic];
      i = -1;
      for (j=jsta; j<=jend; j++) {
        i++;
        cluster->xjt[i] = particle->xjl[j];
        cluster->yjt[i] = particle->yjl[j];
        cluster->zjt[i] = particle->zjl[j];
        cluster->gjt[i] = particle->gjl[j];
      }

  /*
    loop through all neighbors
  */
      for (jx=jx_min; jx<=jx_max; jx++) {
        for (jy=jy_min; jy<=jy_max; jy++) {
          for (jz=jz_min; jz<=jz_max; jz++) {
            if (ix != jx || iy != jy || iz != jz) {
              jc = jz*cluster->nx*cluster->ny+jx*cluster->ny+jy;
              jsta = cluster->jsta[jc];
              jend = cluster->jend[jc];
 
  /*
    select from the particles in the neighbor boxes, the ones that belong in the truncated zone
  */
              if (jsta <= jend) {
                for (j=jsta; j<=jend; j++) {
                  xj = particle->xjl[j];
                  yj = particle->yjl[j];
                  zj = particle->zjl[j];
                  gj = particle->gjl[j];
 
  /*
    add all particles in the neighbor boxes into the corresponding cell structure
  */
                  if (fabs(xj-xc) < cluster->trunc_length &&
                      fabs(yj-yc) < cluster->trunc_length &&
                      fabs(zj-zc) < cluster->trunc_length) {
                    i++;
                    cluster->xjt[i] = xj;
                    cluster->yjt[i] = yj;
                    cluster->zjt[i] = zj;
                    cluster->gjt[i] = gj;
                  }
                }
              }
            }
          }
        }
      }
      cluster->nptruncj = i+1;
    }
    else {
      cluster->nptruncj = 0;
    }
    if (cluster->file == 1) {
      std::ofstream fid;
      fid.open("trunc.dat", std::ios::app);
      fid << cluster->nptruncj << " ";
      for (i=0; i<cluster->nptruncj; i++) fid << cluster->xjt[i] << " ";
      for (i=0; i<cluster->nptruncj; i++) fid << cluster->yjt[i] << " ";
      for (i=0; i<cluster->nptruncj; i++) fid << cluster->zjt[i] << " ";
      fid.close();
    }
  }
};
