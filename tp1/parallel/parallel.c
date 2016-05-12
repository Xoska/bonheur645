//
// Created by Agustin Suarez on 2016-05-05.
//

#include "mpi.h"
#include <stdio.h>

void main(int argc, char *argv[]){
    int err;
    int np = 16;
    int myid = 0;
    err = MPI_init(&argc, &argv);

    if(err != MPI_SUCCESS){
        printf("ERROR");
        exit(1);
    }

    MPI_Comm_Size(MPI_COMM_WORLD, &np);
    MPI_Comm_Rank(MPI_COMM_WROLD, &myid);

    printf("Hello World! Je suis le processus no. %d de %d processus\n", myid, np);
    MPI_Finalize();
}