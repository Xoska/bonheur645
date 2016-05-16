#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <stddef.h>
#include <stdbool.h>

#define SIZE 8
#define CELL_STRUCTURE_SIZE 3

struct TransformationProperties {
    int sizeX;
    int sizeY;
    int iterations;
    long startValue;
    int problemToSolve;
};

struct MPIParams {
    int worldRank;
    int worldSize;
    int dataSize;
    struct Cell* initData;
    int subsetSize;
};

typedef struct Cell {
    int posX;
    int posY;
    int value;
} cell;

MPI_Datatype mpiCellType;

struct TransformationProperties buildProperties(char *argv[]) {

    struct TransformationProperties properties;

    properties.sizeX = SIZE;
    properties.sizeY = SIZE;
    properties.problemToSolve = atoi(argv[1]);
    properties.startValue = atoi(argv[2]);
    properties.iterations = atoi(argv[3]);

    return properties;
}

struct Cell buildCell(int posX, int posY, int value) {

    struct Cell cell;

    cell.posX = posX;
    cell.posY = posY;
    cell.value = value;

    return cell;
}

void printCells(struct Cell* cells, int size) {

    int i;

    for (i = 0; i < size; ++i) {

        printf("Cell[%d][%d] = %d\n", cells[i].posX, cells[i].posY, cells[i].value);
    }
}

struct Cell* createCells(struct TransformationProperties properties) {

    struct Cell* cells = (struct Cell*) malloc(sizeof(struct Cell) * (properties.sizeX * properties.sizeY));

    int i = 0;
    int posX, posY;

    for (posX = 0; posX < properties.sizeX; posX++) {

        for (posY = 0; posY < properties.sizeY; posY++) {

            cells[i++] = buildCell(posX, posY, properties.startValue);
        }
    }

    return cells;
}

struct Cell computeProblemOne(struct Cell cell, int iterations) {

    int valueAtIterationZero = cell.value;

    int k;
    for (k = 1; k <= iterations; ++k) {

        cell.value = valueAtIterationZero + (cell.posX + cell.posY) * k;
        valueAtIterationZero = cell.value;
    }

    return cell;
}

struct Cell* computeCells(struct Cell* cells, int subsetSize, struct TransformationProperties properties) {

    int i;
    for (i = 0; i < subsetSize; ++i) {

        cells[i] = computeProblemOne(cells[i], properties.iterations);
    }

    return cells;
}


void createCellStructure() {

    const int countStructureCell = CELL_STRUCTURE_SIZE;

    int blockLengths[CELL_STRUCTURE_SIZE] = {1, 1, 1};
    MPI_Datatype types[CELL_STRUCTURE_SIZE] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[CELL_STRUCTURE_SIZE];

    offsets[0] = offsetof(cell, posX);
    offsets[1] = offsetof(cell, posY);
    offsets[2] = offsetof(cell, value);

    MPI_Type_create_struct(countStructureCell, blockLengths, offsets, types, &mpiCellType);
    MPI_Type_commit(&mpiCellType);
}

struct MPIParams initMPI(int argc, char *argv[], struct TransformationProperties properties) {

    struct MPIParams mpiParams;

    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    mpiParams.worldRank = world_rank;

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    mpiParams.worldSize = world_size;

    createCellStructure();

    mpiParams.dataSize = properties.sizeX * properties.sizeY;
    mpiParams.subsetSize = mpiParams.dataSize / mpiParams.worldSize;

    mpiParams.initData = NULL;

    if (mpiParams.worldRank == 0) {

        mpiParams.initData = createCells(properties);
    }

    return mpiParams;
}

bool isRootProcess(int worldRank) {

    return worldRank == 0;
}

void processParallelProblemOne(int argc, char *argv[], struct TransformationProperties properties) {

    struct MPIParams mpiParams = initMPI(argc, argv, properties);

    struct Cell* subsetCells = (struct Cell*)malloc(sizeof(struct Cell) * mpiParams.subsetSize);

    MPI_Scatter(mpiParams.initData, mpiParams.subsetSize, mpiCellType,
                subsetCells, mpiParams.subsetSize, mpiCellType,
                0, MPI_COMM_WORLD);

    struct Cell* subProcessedCells = computeCells(subsetCells, mpiParams.subsetSize, properties);

    struct Cell* processedCells = NULL;

    if (isRootProcess(mpiParams.worldRank)) {

        processedCells = (struct Cell*) malloc(sizeof(struct Cell) * mpiParams.dataSize);
    }

    MPI_Gather(&subProcessedCells[0], mpiParams.subsetSize, mpiCellType,
               processedCells, mpiParams.subsetSize, mpiCellType, 0, MPI_COMM_WORLD);

    if (isRootProcess(mpiParams.worldRank)) {

        printCells(processedCells, mpiParams.dataSize);

        free(mpiParams.initData);
        free(processedCells);
    }

    free(subProcessedCells);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

void processParallelProblemTwo(int argc, char *argv[], struct TransformationProperties properties) {


}

int main(int argc, char *argv[]) {

    if (argc == 4) {

        struct TransformationProperties properties = buildProperties(argv);

        if (properties.problemToSolve == 1) {

            processParallelProblemOne(argc, argv, properties);
        }
        else if (properties.problemToSolve == 2) {

            processParallelProblemTwo(argc, argv, properties);
        }
    }
    else {

        printf("Missing arguments.\n");
    }
}