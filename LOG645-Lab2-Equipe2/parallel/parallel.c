#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#define SIZE 10

struct TransformationProperties {
    int sizeX;
    int sizeY;
    int dataSize;
    int iterations;
    int startValue;
    int problemToSolve;
};

typedef struct Cell {
    int posX;
    int posY;
    int value;
} cell;

struct Cell buildCell(int posX, int posY, int value) {

    struct Cell cell;

    cell.posX = posX;
    cell.posY = posY;
    cell.value = value;

    return cell;
}

struct Cell* createCells(struct TransformationProperties properties) {

    struct Cell* cells = (struct Cell*) malloc(sizeof(struct Cell) * (properties.sizeX * properties.sizeY));

    int i = 0;
    int posX, posY;

    for (posX = 0; posX < properties.sizeX; posX++) {

        for (posY = 0; posY < properties.sizeY; posY++) {

            if (properties.problemToSolve == 1) {

                cells[i++] = buildCell(posX, posY, properties.startValue);
            }
            else if (properties.problemToSolve == 2) {

                cells[i++] = buildCell(posY, posX, properties.startValue);
            }
        }
    }

    return cells;
}

void printCells(struct Cell* cells, struct TransformationProperties properties, double elapsedTime) {

    printf("Time taken : %f ms.\n", elapsedTime);
    printf("Results : \n");

    int i;
    for (i = 0; i < properties.dataSize; ++i) {

        if (i > 0 && i % properties.sizeX == 0) {

            printf("\n");
        }

        printf("%-10d|", cells[i].value);
    }

    printf("\n\n");
}

void freeMemory(struct Cell* cells) {

    free(cells);
}

struct TransformationProperties buildProperties(char *argv[]) {

    struct TransformationProperties properties;

    properties.sizeX = SIZE;
    properties.sizeY = SIZE;
    properties.dataSize = properties.sizeX * properties.sizeY;
    properties.problemToSolve = atoi(argv[1]);
    properties.startValue = atoi(argv[2]);
    properties.iterations = atoi(argv[3]);

    return properties;
}

int processProblemOneCore(struct Cell cell) {

    usleep(500);

    return cell.posX + cell.posY;
}

struct Cell* processProblemOneSequential(struct Cell* cells, struct TransformationProperties properties) {

    int i, k;
    for (k = 1; k <= properties.iterations; ++k) {

        for (i = 0; i < properties.dataSize; i++) {

            cells[i].value += processProblemOneCore(cells[i]);
        }
    }

    return cells;
}

struct Cell* processProblemOneParallel(struct Cell* cells, struct TransformationProperties properties) {

    int i, k;
    for (k = 1; k <= properties.iterations; ++k) {

        #pragma omp parallel for shared(cells) schedule(static, 1)
        for (i = 0; i < properties.dataSize; i++) {

            cells[i].value += processProblemOneCore(cells[i]);
        }
    }

    return cells;
}

int processProblemTwoCore(struct Cell* cells, int i, int sizeX) {

    usleep(500);

    if (cells[i].posX >= sizeX - 1) {

        return cells[i].posY;
    }
    else {

        return cells[i + 1].value;
    }
}

struct Cell* processProblemTwoSequential(struct Cell* cells, struct TransformationProperties properties) {

    int i, k;
    for (k = 1; k <= properties.iterations; ++k) {

        for (i = properties.dataSize - 1; i >= 0; i--) {

            cells[i].value += processProblemTwoCore(cells, i, properties.sizeX);
        }
    }

    return cells;
}

struct Cell* processProblemTwoParallel(struct Cell* cells, struct TransformationProperties properties) {

    int i, k;
    for (k = 1; k <= properties.iterations; ++k) {

        #pragma omp parallel for shared(cells) schedule(static, properties.sizeX)
        for (i = properties.dataSize - 1; i >= 0; i--) {

            cells[i].value += processProblemTwoCore(cells, i, properties.sizeX);
        }
    }

    return cells;
}

void printProblemInitialization(int problemToSolve, char* solvingType) {

    printf("\nProblem %d solving type : %s.\nStarting process...\n\n", problemToSolve, solvingType);
}

struct Cell* processProblemSequential(struct Cell* cells, struct TransformationProperties properties) {

    printProblemInitialization(properties.problemToSolve, "Sequential");

    if (properties.problemToSolve == 1) {

        return processProblemOneSequential(cells, properties);
    }
    else if (properties.problemToSolve == 2) {

        return processProblemTwoSequential(cells, properties);
    }

    return cells;
}

struct Cell* processProblemParallel(struct Cell* cells, struct TransformationProperties properties) {

    printProblemInitialization(properties.problemToSolve, "Parallel OpenMP");

    if (properties.problemToSolve == 1) {

        return processProblemOneParallel(cells, properties);
    }
    else if (properties.problemToSolve == 2) {

        return processProblemTwoParallel(cells, properties);
    }

    return cells;
}

double getTimeDifferenceMS(struct timeval startTime, struct timeval endTime) {

    double elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000.0;
    elapsedTime += (endTime.tv_usec - startTime.tv_usec) / 1000.0;

    return elapsedTime;
}

double processProblemCore(struct TransformationProperties properties, int processType) {

    struct Cell* cells = createCells(properties);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);

    if (processType == 1) {

        cells = processProblemSequential(cells, properties);
    }
    else if (processType == 2) {

        cells = processProblemParallel(cells, properties);
    }

    gettimeofday(&endTime, NULL);
    double elapsedTime = getTimeDifferenceMS(startTime, endTime) ;

    printCells(cells, properties, elapsedTime);
    freeMemory(cells);

    return elapsedTime;
}

void printInitialization(struct TransformationProperties properties) {

    printf("\nProperties : \n");
    printf("    Problem to solve : %d\n", properties.problemToSolve);
    printf("    Starting value : %d\n", properties.startValue);
    printf("    Iterations : %d\n", properties.iterations);
    printf("    Matrix %d x %d\n\n", properties.sizeX, properties.sizeY);
    printf("----------------------------------------------------\n");
}

void compareProcesses(double timeSequential, double timeParallel) {

    double acceleration = timeSequential / timeParallel;

    printf("----------------------------------------------------\n\n");
    printf("Acceleration = Time sequential / Time parallel\n");
    printf("             = %f ms / %f ms\n", timeSequential, timeParallel);
    printf("             = %f\n", acceleration);
}

void initOMP() {

    omp_set_num_threads(64);
}

int main(int argc, char *argv[]) {

    if ( argc == 4 ) {

        struct TransformationProperties properties = buildProperties(argv);
        initOMP();

        if (properties.problemToSolve == 1 || properties.problemToSolve == 2) {

            printInitialization(properties);

            double timeSequential = processProblemCore(properties, 1);
            double timeParallel = processProblemCore(properties, 2);

            compareProcesses(timeSequential, timeParallel);

            printf("\n");
        }
        else {

            printf("Invalid arguments. Problem to solve must be 1 or 2.\n");
        }
    }
    else {

        printf("Missing arguments : [Problem to solve] [Starting value] [Iterations].\n");
    }

    return 0;
}