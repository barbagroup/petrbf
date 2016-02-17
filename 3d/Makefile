NP=1
LOCDIR=3d
LIB = -L$(CUDA_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/lib -lcudart -lGL -lGLU -lcutil_x86_64 -lstdc++

cpu: main.o rbf_interpolation.o vorticity_evaluation.o mpi_range.o matmult.o
	${CLINKER} -o main $? ${PETSC_SNES_LIB}
	mpirun -n ${NP} ./main 1.0 3 1.5 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall
gpu: main.o rbf_interpolation.o vorticity_evaluation.o mpi_range.o matmultgpu.o gpumatmult.o
	${CLINKER} -o main $? ${PETSC_SNES_LIB} $(LIB)
	mpirun -n ${NP} ./main 1.0 3 1.5 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall

static: wrapper.o rbf_interpolation.o vorticity_evaluation.o mpi_range.o matmultgpu.o gpumatmult.o
	ar ruv librbf.a $?
	ranlib librbf.a

purge:
	$(RM) *.dat *.o main
save:
	make purge
	tar zcvf ../../rbf.tgz ../../rbf

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
