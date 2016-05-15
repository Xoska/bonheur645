//
// Created by Agustin Suarez on 2016-05-05.
//

#include "mpi.h"
#include <stdio.h>

void main(int argc, char *argv[]){
    int err;
    int np;
    int myid;
    err = MPI_Init(&argc, &argv);

    if(err != MPI_SUCCESS){
        printf("ERROR");
        exit(1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    long long myMatrix[8][8];

    if(myid == 0){
        printf("Actually, this is a test %d %d\n", myid, np);


        /*Tests*/
//        int i, j;
//        for(i = 0; i < 8; i++){
//            for (j = 0; j < 8; j++){
//                myMatrix[i][j] = 4;
//            }
//        }

        printMatrix(myMatrix);

        //On scatter  ou broadcast les elements de la matrice a chaque processeur sauf 0
        MPI_Scatter(myMatrix, 2, MPI_INT, myMatrix, 2, MPI_INT, 0, MPI_COMM_WORLD);

        printf("done\n");

    } else {
        printf("Hello World! Je suis le processus no. %d de %d processus, mymatrix: %d\n", myid, np, myMatrix[0][1]);
        //Perform calculations
        //myid donne le id du processeur. peut etre utilisee pour gerer le row et la colonne
        // mpigather

    }

    MPI_Finalize();
}

//void printMatrix(long** matrix){
//    int i,j;
//
//    for (i = 0; i < sizeof(matrix); i++){
//        for (j = 0; j < sizeof(matrix[i]); j++){
//            printf("%d |", matrix[i][j]);
//        }
//        printf("\n");
//    }
//}