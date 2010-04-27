NP=1
LOCDIR=2d

main: main.o
	${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	$(PETSC_DIR)/$(PETSC_ARCH)/bin/mpiexec -n $(NP) ./main .9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall
main2: main2.o
	${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	$(PETSC_DIR)/$(PETSC_ARCH)/bin/mpiexec -n $(NP) ./main2 .9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall
	$(RM) *.o
cylinder: cylinder.o
	${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	$(PETSC_DIR)/$(PETSC_ARCH)/bin/mpiexec -n $(NP) ./cylinder .9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall
simple: simple.o
	${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	$(PETSC_DIR)/$(PETSC_ARCH)/bin/mpiexec -n $(NP) ./simple -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000
rsimple:
	$(PETSC_DIR)/$(PETSC_ARCH)/bin/mpiexec -n $(NP) ./simple -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-13 -ksp_max_it 1000 -vecscatter_alltoall
purge:
	$(RM) *.dat *.o main cylinder simple
save:
	make purge
	tar zcvf ../rbf.tgz ../rbf

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
