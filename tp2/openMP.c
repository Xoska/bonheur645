#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#define SIZE 10

struct TransformationProperties {
    int sizeX;
    int sizeY;
    int iterations;
    int startValue;
    int problemToSolve;
};

int** createArray(struct TransformationProperties properties) {

    int** array;
    int i, j;

    array = (int**) malloc(properties.sizeX * sizeof(int*));
    for (i = 0; i < properties.sizeX; i++) {

        array[i] = (int*) malloc(properties.sizeY * sizeof(int));

        for (j = 0; j < properties.sizeY; j++) {

            array[i][j] = properties.startValue;
        }
    }

    return array;
}

void destroyArray(int** array)
{
    free(*array);
    free(array);
}

struct TransformationProperties buildProperties(char *argv[]) {

    struct TransformationProperties properties;

    properties.sizeX = SIZE;
    properties.sizeY = SIZE;
    properties.problemToSolve = atoi(argv[1]);
    properties.startValue = atoi(argv[2]);
    properties.iterations = atoi(argv[3]);

    return properties;
}

int** solveProblemOne(int** array, struct TransformationProperties properties) {

    int i, j, k;

    #pragma omp parallel
    {
        for (k = 1; k <= properties.iterations; ++k) {

            for (i = 0; i < properties.sizeX; i++) {

                for (j = 0; j < properties.sizeY; j++) {

                    usleep(500);

                    array[i][j] = array[i][j] + i + j;
                }
            }
        }
    }

    return array;
}

int** solveProblemTwo(int** array, struct TransformationProperties properties) {

    int i, j, k;

    for (k = 1; k <= properties.iterations; ++k) {

        for (i = 0; i < properties.sizeX; i++) {

            for (j = 0; j < properties.sizeY; j++) {

                usleep(1000);

                if (j >= 9) {

                    array[i][j] = array[i][j] + i;
                }
                else {

                    array[i][j] = array[i][j] + array[i][j + 1];
                }
            }
        }
    }

    return array;
}

void printTime(clock_t timeDiff) {

    long time = timeDiff * 1000 / CLOCKS_PER_SEC;

    printf("Time taken : %ld seconds %ld milliseconds.\n", time / 1000, time % 1000);
}

void printArray(int** array, struct TransformationProperties properties) {

    printf("Problem %d. \n", properties.problemToSolve);

    int i, j;

    for (i = 0; i < properties.sizeX; i++) {

        for (j = 0; j < properties.sizeY; j++) {

            printf("%-10d|", array[i][j]);
        }

        printf("\n");
    }

    printf("\n");
}

void printResults(int** result,
                  struct TransformationProperties properties, clock_t timeDiff) {

    printTime(timeDiff);
    printArray(result, properties);
}

void processProcedure(struct TransformationProperties properties) {

    printf("\n\nStarting process...\n");
    clock_t startSolvingProblems = clock();

    int** array = createArray(properties);

    if (properties.problemToSolve == 1) {

        array = solveProblemOne(array, properties);
    }
    else if (properties.problemToSolve == 2) {

        array = solveProblemTwo(array, properties);
    }

    printResults(array, properties, clock() - startSolvingProblems);

    destroyArray(array);
}

void init() {

    omp_set_num_threads(64);
}

int main(int argc, char *argv[]) {

    if ( argc == 4 ) {

        struct TransformationProperties properties = buildProperties(argv);

        init();

        int rang, nprocs;
        int toto = 0;
        #pragma omp parallel shared(toto)
        {
            rang = omp_get_thread_num();
            nprocs = omp_get_num_threads();
            printf("Bonjour, je suis %d (parmi %d threads)\n", rang, nprocs);
            toto++;
        }

        printf("Bonjour, je suis toto : %d\n", toto);


      //  processProcedure(properties);
    }
    else {

        printf("Missing arguments.\n");
    }

    return 0;
}