#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
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

void printCells(struct Cell* cells, struct TransformationProperties properties, struct timeval startTime) {

    double elapsedTime;
    struct timeval endTime;

    gettimeofday(&endTime, NULL);

    elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000.0;
    elapsedTime += (endTime.tv_usec - startTime.tv_usec) / 1000.0;

    printf("Time taken : %f milliseconds.\n", elapsedTime);

    int i, dataSize = properties.sizeX * properties.sizeY;

    for (i = 0; i < dataSize; ++i) {

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
    properties.problemToSolve = atoi(argv[1]);
    properties.startValue = atoi(argv[2]);
    properties.iterations = atoi(argv[3]);

    return properties;
}

struct Cell* processProblemOne(struct Cell* cells, struct TransformationProperties properties) {

    int i, k, dataSize = properties.sizeX * properties.sizeY;

    for (k = 1; k <= properties.iterations; ++k) {

        #pragma omp parallel for shared(cells) schedule(static, 1)
        for (i = 0; i < dataSize; i++) {

            usleep(500);

            cells[i].value += (cells[i].posX + cells[i].posY);
        }
    }

    return cells;
}

struct Cell* processProblemTwo(struct Cell* cells, struct TransformationProperties properties) {

    int i, k, dataSize = properties.sizeX * properties.sizeY;

    for (k = 1; k <= properties.iterations; ++k) {

        #pragma omp parallel for shared(cells) schedule(static, properties.sizeX)
        for (i = dataSize - 1; i >= 0; i--) {

            usleep(500);

            if (cells[i].posX >= properties.sizeX - 1) {

                cells[i].value += cells[i].posY;
            }
            else {

                cells[i].value += cells[i + 1].value;
            }
        }
    }

    return cells;
}

void processProblemCore(struct Cell* cells, struct TransformationProperties properties, struct timeval startTime) {

    if (properties.problemToSolve == 1) {

        cells = processProblemOne(cells, properties);
    }
    else if (properties.problemToSolve == 2) {

        cells = processProblemTwo(cells, properties);
    }

    printCells(cells, properties, startTime);

    freeMemory(cells);
}

void initOMP() {

    omp_set_num_threads(64);
}

int main(int argc, char *argv[]) {

    if ( argc == 4 ) {

        printf("\nProblems solve type : Parallel openMP.\nStarting process...\n\n");

        struct timeval startTime;
        gettimeofday(&startTime, NULL);

        struct TransformationProperties properties = buildProperties(argv);
        struct Cell* cells = createCells(properties);
        initOMP();

        if (properties.problemToSolve == 1 || properties.problemToSolve == 2) {

            processProblemCore(cells, properties, startTime);
        }
        else {

            printf("Invalid arguments.\n");
        }
    }
    else {

        printf("Missing arguments.\n");
    }

    return 0;
}