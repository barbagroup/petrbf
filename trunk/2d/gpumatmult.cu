extern const int threadsPerBlock;

void gpumatmult(double *targetX, double *targetY, double *targetW,
                double *sourceX, double *sourceY, double *sourceG,
                int *offset, int iblok, double sigma)
{
  int ic,i,im,j;
  double dx,dy,w;

  for (ic=0; ic<iblok; ic++) {
    for (i=0; i<threadsPerBlock; i++) {
      im = ic*threadsPerBlock+i;
      w = 0;
      for (j=offset[ic]; j<offset[ic+1]; j++) {
        dx = targetX[im]-sourceX[j];
        dy = targetY[im]-sourceY[j];
        w += sourceG[j]*exp(-(dx*dx+dy*dy)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
      }
      targetW[im] = w;
    }
  }
}
