main: main.o
	${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	$(PETSC_DIR)/$(PETSC_ARCH)/bin/mpiexec -np $(NP) ./main .9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall

purge:
	$(RM) *.dat *.o main cylinder
save:
	make purge
	tar zcvf ../rbf.tgz ../rbf

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
