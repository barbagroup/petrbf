NP=1
LOCDIR=2d
LIB = -L$(CUDA_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/lib -lcudart -lGL -lGLU -lcutil_x86_64 -lstdc++ -ldl -lm

cpu: main.o rbf_interpolation.o vorticity_evaluation.o mpi_range.o matmult.o
	${CLINKER} -o main $? ${PETSC_SNES_LIB}
	mpirun -n $(NP) ./main .9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall -log_summary
gpu: main.o rbf_interpolation.o vorticity_evaluation.o mpi_range.o matmultgpu.o gpumatmult.o
	${CLINKER} -o main $? ${PETSC_SNES_LIB} $(LIB)
	mpirun -n $(NP) ./main .8 7 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall -log_summary

main2: main2.o
	${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	mpirun -n $(NP) ./main2 .9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall
	$(RM) *.o
cylinder: cylinder.o
	${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	mpirun -n $(NP) ./cylinder .9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall
purge:
	$(RM) *.dat *.o main main2 cylinder
save:
	make purge
	tar zcvf ../../rbf.tgz ../../rbf

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
