//
// Created by Agustin Suarez on 2016-05-05.
//

#include "mpi.h"
#include <stdio.h>
#include "math.h"

//void printMatrix(int** matrix);

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

    int myMatrix[8][8];

    //if(myid == 0){
        //printf("Actually, this is a test %d %d\n", myid, np);


        /*Tests*/
        if(myid ==0){
            int m, n;
            for(m = 0; m < 8; m++){
                for (n = 0; n < 8; n++){
                    myMatrix[m][n] = 4;
                }
            }
        }

        //printMatrix(myMatrix);

        //On scatter ou broadcast les elements de la matrice a chaque processeur sauf 0
        //MPI_Bcast(myMatrix, 1, MPI_INT, myMatrix, 2, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(myMatrix, 64, MPI_INT, 0, MPI_COMM_WORLD);

        int i = myid/8;
        int j = myid%8;
        int iterations = 2;
        int l;
        for(l = 1; l <= iterations; l++){
            myMatrix[i][j] += (i+j)*l;
        }
        printf("Cell (%d,%d) has a value of %i\n", i, j, myMatrix[i][j]);

        printf("done\n");

    //} else {
        //printf("Salut mon ID c'est: %d, j'ai recu les valeurs suivantes: %d\n", myid, myMatrix[0][0]);

//        int i = myid/8;
//        int j = myid%8;
//        int iterations = 1;
//        int l;
//        for(l = 0; l < iterations; l++){
//            myMatrix[i][j] += (i+j)*l;
//        }
//        printf("Cell %d: %d has a value of %i\n", i, j, myMatrix[i][j]);


        //Perform calculations
        //myid donne le id du processeur. peut etre utilisee pour gerer le row et la colonne
        // mpigather

   // }

    MPI_Finalize();
}

//void printMatrix(int matrix){
//    int i,j;
//
//    for (i = 0; i < sizeof(matrix); i++){
//        for (j = 0; j < sizeof(matrix[i]); j++){
//            printf("%d |", matrix[i][j]);
//        }
//        printf("\n");
//    }
//}