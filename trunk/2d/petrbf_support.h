/** Support functions for RBF manipulation.
 *
 */
#include "petscvec.h"

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