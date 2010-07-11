export OVERLAP=8
export B_DOMAIN=3
export D_DOMAIN=15
rm stop
while [ $OVERLAP -ge 8 ]
do
a=0.1
export O=`echo $OVERLAP \* $a | bc`
export B=$B_DOMAIN
export D=`echo $D_DOMAIN \* $a | bc`
echo "||-----------------------------------------------------------------"
if [ $OVERLAP -eq 10 ]; then
echo "||  overlap :  $O    B domain :  $B    D/B ratio :  $D   compile"
else
echo "||  overlap :  $O     B domain :  $B    D/B ratio :  $D   compile"
fi
echo "||-----------------------------------------------------------------"
make gpuc
# llsubmit ../bluegene.sh
# qsub -V ../start.sh
$PETSC_DIR/$PETSC_ARCH/bin/mpiexec -np 1 ./main $O $B $D -pc_type asm -sub_pc_type lu -sub_mat_type dense -ksp_monitor -ksp_rtol 1e-5 -ksp_max_it 100 -vecscatter_alltoall
export D_DOMAIN=`expr $D_DOMAIN + 2`
if [ $D_DOMAIN -gt 23 ]; then
export D_DOMAIN=15
export B_DOMAIN=`expr $B_DOMAIN + 1`
fi
if [ $B_DOMAIN -gt 7 ]; then
export B_DOMAIN=3
export OVERLAP=`expr $OVERLAP - 1`
fi
if [ -e "stop" ]; then
echo "||-----------------------------------------------------------------"
echo "||                        canceled by user                         "
echo "||-----------------------------------------------------------------"
export OVERLAP=7
fi
done
