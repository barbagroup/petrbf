export NP=1
while [ $NP -le 1024 ]
do
    echo $NP
    llsubmit bluegene.sh
    export NP=`echo $NP \* 2 | bc`
done
