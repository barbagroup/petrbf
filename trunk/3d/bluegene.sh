# @ job_type = parallel
# @ environment = COPY_ALL
# @ executable = /bgl/BlueLight/ppcfloor/bglsys/bin/mpirun
# @ arguments = -verbose 1 -np $NP -cwd $HOME/rbf -exe $HOME/rbf/main -args ".9 5 1.9 -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_rtol 1e-13 -ksp_max_it 1000 -ksp_monitor -log_summary -vecscatter_alltoall"
# @ wall_clock_limit = 30:00
# @ input = /dev/null
# @ output = $(jobid).out
# @ error = $(jobid).err
# @ notification = never
# @ queue
