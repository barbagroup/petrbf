#ifndef _PETRBF_SUPP
#define _PETRBF_SUPP
/**
 * Support functions for RBF manipulation.
 */
#include "petscvec.h"

/** 
 * Computes the normalized log10 error of two vectors and stores the results.
 * W = log10(abs(x - y))
 */
int VecNormalizedErrorWXY(Vec W, Vec X, Vec Y){
  PetscInt  i, nlocal;
  PetscReal *array;
  PetscReal vecNorm;
  
  (W == Y) ? VecAXPY(W, -1.0, X) : VecWAXPY(W, -1.0, X, Y);
  VecAbs(W);
  VecNormalize(W, &vecNorm);
  
  VecGetLocalSize(W, &nlocal);
  VecGetArray(W, &array);
  for (i=0; i<nlocal; i++) {
    array[i] = log10(array[i]);
  }
  VecRestoreArray(W, &array);  
  
  return 0;
}

/** 
 * Computes the log10 difference of two vectors and stores the results.
 * W = log10(abs(x - y))
 */
int VecErrorWXY(Vec W, Vec X, Vec Y){
  PetscInt  i, nlocal;
  PetscReal *array;
  
  VecWAXPY(W, -1.0, X, Y);
  VecAbs(W);
  
  VecGetLocalSize(W, &nlocal);
  VecGetArray(W, &array);
  for (i=0; i<nlocal; i++) {
    array[i] = log10(array[i]);
  }
  VecRestoreArray(W, &array);  
  
  return 0;
}

#endif