#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define SIZE 8

struct TransformationProperties {
    int sizeX;
    int sizeY;
    int iterations;
    long startValue;
    int problemToSolve;
};

long** createArray(struct TransformationProperties properties) {

    long** array;
    int i, j;

    array = (long**) malloc(properties.sizeX * sizeof(long*));
    for (i = 0; i < properties.sizeX; i++) {

        array[i] = (long*) malloc(properties.sizeY * sizeof(long));

        for (j = 0; j < properties.sizeY; j++) {

            array[i][j] = properties.startValue;
        }
    }

    return array;
}

void destroyArray(long** array)
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

long** solveProblemOne(long** array, struct TransformationProperties properties) {

    int i, j, k;

    for (k = 1; k <= properties.iterations; ++k) {

        for (i = 0; i < properties.sizeX; i++) {

            for (j = 0; j < properties.sizeY; j++) {

                usleep(1000);

                array[i][j] = array[i][j] + (i + j) * k;
            }
        }
    }

    return array;
}

long** solveProblemTwo(long** array, struct TransformationProperties properties) {

    int i, j, k;

    for (k = 1; k <= properties.iterations; ++k) {

        for (i = 0; i < properties.sizeX; i++) {

            for (j = 0; j < properties.sizeY; j++) {

                usleep(1000);

                if (j > 0) {

                    array[i][j] = array[i][j] + array[i][j - 1] * k;
                }
                else {

                    array[i][j] = array[i][j] + (i * k);
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

void printArray(long** array, struct TransformationProperties properties) {

    printf("Problem %d. \n", properties.problemToSolve);

    int i, j;

    for (i = 0; i < properties.sizeX; i++) {

        for (j = 0; j < properties.sizeY; j++) {

            printf("%ld|", array[i][j]);
        }

        printf("\n");
    }

    printf("\n");
}

void printResults(char* typeProcess, long** result,
                  struct TransformationProperties properties, clock_t timeDiff) {

    printf("Problems solve type : %s. \n", typeProcess);

    printTime(timeDiff);
    printArray(result, properties);
}

void processSequentialProcedure(struct TransformationProperties properties) {

    printf("\n\nStarting sequential process...\n");
    clock_t startSolvingProblems = clock();

    long** array = createArray(properties);

    if (properties.problemToSolve == 1) {

        array = solveProblemOne(array, properties);
    }
    else if (properties.problemToSolve == 2) {

        array = solveProblemTwo(array, properties);
    }

    printResults("Sequential", array, properties, clock() - startSolvingProblems);

    destroyArray(array);
}

int main(int argc, char *argv[]) {

    if ( argc == 4 ) {

        struct TransformationProperties properties = buildProperties(argv);

        processSequentialProcedure(properties);
    }
    else {

        printf("Missing arguments.\n");
    }
}