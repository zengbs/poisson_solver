#!/bin/bash


rm Potential 

NUM_NX=1024
FileName="Output__ComputingSpeed_vs_NumThreads_NUM_NX_${NUM_NX}"
printf "# NUM_NX = %d\n" $NUM_NX  > $FileName
echo "# [1] cells/sec [2] number of threads" >> $FileName


for Threads in {1..16}
do
  g++ main.c -fopenmp -DNUM_NX=$NUM_NX -DNUM_THREADS=$Threads
  ./a.out
  ComputingSpeed=`sed -n 1p Potential  | awk '{print $2}'`
  NumThreads=`sed -n 2p Potential  | awk '{print $4}'`
  echo "$ComputingSpeed    $NumThreads" >> $FileName
  rm a.out 
done
