/**
 * Simple interpolation example.
 * Given a source set of gaussian points, a new interpolator is created for a new
 * set of points.
 *
 * Files are created with data of the computations. The files are in PETSc binary format.
 * List of files:
 * data/meshEval.dat    Field evaluated at a mesh that results from the interpolation with the new set.
 * data/meshSource.dat  Field evaluated at a mesh that results from the interpolation with the original set.
 * data/meshError.dat   Log10 error of the field, when comparing the results from both sets.
 * data/set1.dat        Original set of gaussian points.
 * data/set2.dat        New set of gaussian points.
 */
#include "petscvec.h"
#include "vorticity_evaluation.cxx"
#include "rbf_interpolation.cxx"

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

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscRandom    rctx;
  PetscViewer    viewer;
  
  try {
    PetscInt    i, j;
    ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL); CHKERRXX(ierr);
    
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    PetscRandomSetType(rctx,PETSCRAND);
    
    /* I) Experimental setup */
    
    // Source points parameters
    PetscInt Nsource = 1000; // number of source points
    PetscReal constW = 1.0;  // constant weight assigned to each source point
    PetscInt Meval   = 4000; // number of evaluation points
    PetscReal sigma  = 0.02;
    // RBF interpolation parameters
    PetscInt  its;  // number of iterations, output of solver
    PetscReal nsigma_box   = 10;
    PetscReal sigma_buffer = 2 * nsigma_box;
    PetscReal sigma_trunc  = nsigma_box + 10;
    // Lattice parameters
    PetscInt numNodesX = 100;
    PetscInt numNodesY = 100;
    PetscInt numNodes  = numNodesX * numNodesY;
    
    
    //// Create set of Nsource points: coordinates are (x1, y1) and known weight w1
    Vec sourceX, sourceY, sourceW; // position X, position Y, weight, value at position
    VecCreate(PETSC_COMM_WORLD, &sourceX);
    VecSetSizes(sourceX, PETSC_DECIDE, Nsource);
    VecSetFromOptions(sourceX);
    VecDuplicate(sourceX, &sourceY);
    VecDuplicate(sourceX, &sourceW);
    // Random set of source points
    VecSetRandom(sourceX, rctx);
    VecSetRandom(sourceY, rctx);
    // Set points weights equal to a constant
    VecSet(sourceW, constW);
    
    
    //// New set of Meval points: coordinates are given by (x2, y2) and g2 is unknown
    Vec evalX, evalY, evalW, evalV; // position X, position Y, weight, value at position
    VecCreate(PETSC_COMM_WORLD, &evalX);
    VecSetSizes(evalX, PETSC_DECIDE, Meval);
    VecSetFromOptions(evalX);
    VecDuplicate(evalX, &evalY);
    VecDuplicate(evalX, &evalW);
    VecDuplicate(evalX, &evalV);
    VecZeroEntries(evalW);
    VecZeroEntries(evalV);
    // Random set of evaluation points
    VecSetRandom(evalX, rctx);
    VecSetRandom(evalY, rctx);
    
    
    //// Create a set of points (latticeX, latticeY) on a lattice
    Vec         latticeX, latticeY;
    Vec         latticeW_source; // weight values from evaluation with source-set
    Vec         latticeW_eval;   // weight values from evaluation with evaluation-set
    Vec         latticeW_err;    // weight errors between source-set vs eval-set evaluations
    PetscInt    rangeLow, nlocal;
    PetscScalar *arrayX, *arrayY;
    VecCreate(PETSC_COMM_WORLD, &latticeX);
    VecSetSizes(latticeX, PETSC_DECIDE, numNodes);
    VecSetFromOptions(latticeX);
    VecDuplicate(latticeX, &latticeY);
    VecDuplicate(latticeX, &latticeW_source);
    VecDuplicate(latticeX, &latticeW_eval);
    VecDuplicate(latticeX, &latticeW_err);
    VecZeroEntries(latticeW_source);
    VecZeroEntries(latticeW_eval);
    VecZeroEntries(latticeW_err);
    // Setting up the lattice points
    VecGetOwnershipRange(latticeX, &rangeLow, PETSC_NULL);
    VecGetLocalSize(latticeX, &nlocal);
    VecGetArray(latticeX, &arrayX);
    VecGetArray(latticeY, &arrayY);
    for (i=0, j=rangeLow; i<nlocal; i++, j++) {
      arrayX[i] = ((PetscReal)(j % numNodesX)) / numNodesX;
      arrayY[i] = floor(j / numNodesX) / numNodesX;
    }
    VecRestoreArray(latticeX, &arrayX);
    VecRestoreArray(latticeY, &arrayY);    
    
    
    /* II) Evaluate at new points locations */
    vorticity_evaluation(evalX, evalY, evalV, sourceX, sourceY, sourceW, sigma, nsigma_box, sigma_buffer, sigma_trunc);
    
    
    /* III) Obtain weights at the new points */
    rbf_interpolation(evalX, evalY, evalW, evalV, evalV, sigma, nsigma_box, sigma_buffer, sigma_trunc, &its);
    
    
    /* IV) Compare new interpolation evaluating at the original source set */
    vorticity_evaluation(latticeX, latticeY, latticeW_source, sourceX, sourceY, sourceW, sigma, nsigma_box, sigma_buffer, sigma_trunc);
    vorticity_evaluation(latticeX, latticeY, latticeW_eval,   evalX,   evalY,   evalW,   sigma, nsigma_box, sigma_buffer, sigma_trunc);
    
    VecErrorWXY(latticeW_err, latticeW_source, latticeW_eval);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/set1.dat", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = VecView(sourceX, viewer); CHKERRQ(ierr);
    ierr = VecView(sourceY, viewer); CHKERRQ(ierr);
    ierr = VecView(sourceW, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/set2.dat", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = VecView(evalX, viewer); CHKERRQ(ierr);
    ierr = VecView(evalY, viewer); CHKERRQ(ierr);
    ierr = VecView(evalW, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    
    
    /* V) Write output */
    // Save using PETSC binary format
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/meshSource.dat", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = VecView(latticeX, viewer); CHKERRQ(ierr);
    ierr = VecView(latticeY, viewer); CHKERRQ(ierr);
    ierr = VecView(latticeW_source, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/meshEval.dat", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = VecView(latticeX, viewer); CHKERRQ(ierr);
    ierr = VecView(latticeY, viewer); CHKERRQ(ierr);
    ierr = VecView(latticeW_eval, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/meshError.dat", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = VecView(latticeX, viewer); CHKERRQ(ierr);
    ierr = VecView(latticeY, viewer); CHKERRQ(ierr);
    ierr = VecView(latticeW_err, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
    
    /* Free resources */
    ierr = VecDestroy (sourceX);  CHKERRQ(ierr);
    ierr = VecDestroy (sourceY);  CHKERRQ(ierr);
    ierr = VecDestroy (sourceW);  CHKERRQ(ierr);
    ierr = VecDestroy (evalX);    CHKERRQ(ierr);
    ierr = VecDestroy (evalY);    CHKERRQ(ierr);
    ierr = VecDestroy (evalW);    CHKERRQ(ierr);
    ierr = VecDestroy (evalV);    CHKERRQ(ierr);
    ierr = VecDestroy (latticeX); CHKERRQ(ierr);
    ierr = VecDestroy (latticeY); CHKERRQ(ierr);
    ierr = VecDestroy (latticeW_source); CHKERRQ(ierr);
    ierr = VecDestroy (latticeW_eval);   CHKERRQ(ierr);
    ierr = PetscRandomDestroy(rctx);     CHKERRQ(ierr);
  } catch(PETSc::Exception e) {
    std::cerr << "ERROR: " << e << std::endl;
  }
  ierr = PetscFinalize(); CHKERRXX(ierr);
  return 0;
}
