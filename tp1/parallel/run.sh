#!/bin/bash
#sudo ./main $1 $2 $3


mpirun --hostfile hostfile -np 24 parallel $1 $2 $3