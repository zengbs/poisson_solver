mpic++  main.c -fopenmp  && mpirun  -np 2 --bind-to socket --report-bindings ./a.out
