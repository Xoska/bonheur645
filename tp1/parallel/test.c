#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

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
    MPI_Comm topology;
    int topologyRank;
    int topologySize;
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

struct Cell* computeProblemOne(struct Cell* cells, int subsetSize, struct TransformationProperties properties) {

    int i, k;

    for (i = 0; i < subsetSize; ++i) {

        int valueAtIterationZero = cells[i].value;

        for (k = 1; k <= properties.iterations; ++k) {

            cells[i].value = valueAtIterationZero + (cells[i].posX + cells[i].posY) * k;
            valueAtIterationZero = cells[i].value;
        }
    }

    return cells;
}

struct Cell* computeProblemTwo(struct Cell* cells, int subsetSize, int initIteration, int maxIterations) {

    int k, j;

    for (k = initIteration; k <= maxIterations; ++k) {

        struct Cell* previousIterationCells = malloc(subsetSize * sizeof(struct Cell));
        memcpy(previousIterationCells, cells, subsetSize * sizeof(struct Cell));

        for (j = 0; j < subsetSize; ++j) {

            if (j > 0) {

                cells[j].value = previousIterationCells[j].value + cells[j - 1].value * k;
            }
            else {

                cells[j].value = previousIterationCells[j].value + (cells[j].posX * k);
            }
        }

        free(previousIterationCells);
    }

    return cells;
}

struct Cell* computeProblemTwoDispatch(struct MPIParams mpiParams, struct Cell* cells, int subsetSize) {

    int senderRank = mpiParams.worldRank + subsetSize;

    MPI_Status status;
    MPI_Request	send_request;

    cells = computeProblemTwo(cells, subsetSize, 1, 1);

    MPI_Isend(&cells[0], subsetSize, mpiCellType, senderRank, mpiParams.worldRank, MPI_COMM_WORLD, &send_request);
    MPI_Wait(&send_request, &status);

    MPI_Recv(&cells[0], subsetSize, mpiCellType, senderRank, senderRank, MPI_COMM_WORLD, &status);

    return cells;
}

struct Cell* computeProblem(struct MPIParams mpiParams, struct Cell* cells,
                            int subsetSize, struct TransformationProperties properties) {

    if (properties.problemToSolve == 1) {

        return computeProblemOne(cells, subsetSize, properties);
    }
    else if (properties.problemToSolve == 2) {

        return computeProblemTwoDispatch(mpiParams, cells, subsetSize);
    }
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
    mpiParams.initData = NULL;

    if (mpiParams.worldRank == 0) {

        mpiParams.initData = createCells(properties);
    }

    return mpiParams;
}

bool isRootProcess(int worldRank) {

    return worldRank == 0;
}

void freeMemory(struct MPIParams mpiParams, struct Cell* processedCells, struct Cell* subProcessedCells)  {

    if (isRootProcess(mpiParams.worldRank)) {

        free(mpiParams.initData);
        free(processedCells);
    }

    free(subProcessedCells);
}

void report(struct MPIParams mpiParams, struct Cell* processedCells) {

    if (isRootProcess(mpiParams.worldRank)) {

        printCells(processedCells, mpiParams.dataSize);
    }
}

void endProcess(struct MPIParams mpiParams, struct Cell* processedCells, struct Cell* subProcessedCells) {
    report(mpiParams, processedCells);
    freeMemory(mpiParams, processedCells, subProcessedCells);

    MPI_Barrier(mpiParams.topology);
    MPI_Finalize();
}

void processProblemCore(struct MPIParams mpiParams, struct TransformationProperties properties, int subsetSize, bool canScatter) {

    if (canScatter) {

        struct Cell* subsetCells = (struct Cell*)malloc(sizeof(struct Cell) * subsetSize);

        MPI_Scatter(mpiParams.initData, subsetSize, mpiCellType, subsetCells, subsetSize, mpiCellType, 0, mpiParams.topology);

        struct Cell* subProcessedCells = computeProblem(mpiParams, subsetCells, subsetSize, properties);
        struct Cell* processedCells = NULL;

        if (isRootProcess(mpiParams.worldRank)) {

            processedCells = (struct Cell*) malloc(sizeof(struct Cell) * mpiParams.dataSize);
        }

        MPI_Gather(&subProcessedCells[0], subsetSize, mpiCellType, processedCells, subsetSize, mpiCellType, 0, mpiParams.topology);

        endProcess(mpiParams, processedCells, subProcessedCells);
    }
    else {

        if (mpiParams.worldRank < subsetSize * 2) {

            MPI_Status status;
            MPI_Request send_request;
            int senderRank = mpiParams.worldRank - subsetSize;
            struct Cell* subsetCells = (struct Cell*)malloc(sizeof(struct Cell) * subsetSize);

            MPI_Recv(&subsetCells[0], subsetSize, mpiCellType, senderRank, senderRank, MPI_COMM_WORLD, &status);

            struct Cell* subProcessedCells = computeProblemTwo(subsetCells, subsetSize, 2, properties.iterations);

            MPI_Isend(&subsetCells[0], subsetSize, mpiCellType, senderRank, mpiParams.worldRank, MPI_COMM_WORLD, &send_request);


        }

        MPI_Finalize();
    }
}

void processParallelProblemOne(struct MPIParams mpiParams, struct TransformationProperties properties) {

    int subsetSize = mpiParams.dataSize / mpiParams.worldSize;

    mpiParams.topology = MPI_COMM_WORLD;
    mpiParams.topologyRank = mpiParams.worldRank;
    mpiParams.topologySize = mpiParams.worldSize;

    processProblemCore(mpiParams, properties, subsetSize, true);
}

void processParallelProblemTwo(struct MPIParams mpiParams, struct TransformationProperties properties) {

    int subsetSize = properties.sizeY;

    MPI_Comm topology;

    int processorsNeeded = mpiParams.dataSize / subsetSize;
    MPI_Comm_split(MPI_COMM_WORLD, mpiParams.worldRank < processorsNeeded, mpiParams.worldRank, &topology) ;
    mpiParams.topology = topology;

    int topology_rank;
    MPI_Comm_rank(topology, &topology_rank);
    mpiParams.topologyRank = topology_rank;

    int topology_size;
    MPI_Comm_size(topology, &topology_size);
    mpiParams.topologySize = topology_size;

    bool canScatter = mpiParams.topologySize <= subsetSize;

    processProblemCore(mpiParams, properties, subsetSize, canScatter);
}

int main(int argc, char *argv[]) {

    if (argc == 4) {

        struct TransformationProperties properties = buildProperties(argv);
        struct MPIParams mpiParams = initMPI(argc, argv, properties);

        if (properties.problemToSolve == 1) {

            processParallelProblemOne(mpiParams, properties);
        }
        else if (properties.problemToSolve == 2 && mpiParams.worldSize >= (properties.sizeY * 2)) {

            processParallelProblemTwo(mpiParams, properties);
        }
        else {

            printf("Invalid arguments.\n");
        }
    }
    else {

        printf("Missing arguments.\n");
    }
}