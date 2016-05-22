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

void printCells(struct Cell* cells, struct TransformationProperties properties, clock_t timeDiff) {

    printf("Problems solve type : Parallel. \n");

    long time = timeDiff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken : %ld seconds %ld milliseconds.\n", time / 1000, time % 1000);

    int i, dataSize = properties.sizeX * properties.sizeY;

    for (i = 0; i < dataSize; ++i) {

        if (i > 0 && i % properties.sizeX == 0) {

            printf("\n");
        }

        printf("%-10d|", cells[i].value);
    }

    printf("\n\n");
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

struct Cell* computeProblemOne(struct Cell* cells, int subsetSize, int maxIterations) {

    int i, k;

    for (i = 0; i < subsetSize; ++i) {

        for (k = 1; k <= maxIterations; ++k) {

            cells[i].value = cells[i].value + (cells[i].posX + cells[i].posY) * k;
        }
    }

    return cells;
}

struct Cell* computeProblemTwo(struct Cell* cells, int subsetSize, int iteration) {

    int j;

    for (j = 0; j < subsetSize; ++j) {

        if (j > 0) {

            cells[j].value = cells[j].value + cells[j - 1].value * iteration;
        }
        else {

            cells[j].value = cells[j].value + (cells[j].posY * iteration);
        }
    }

    return cells;
}

struct Cell* computeProblemTwoBalancerOne(struct MPIParams mpiParams, struct Cell* cells,
                                          int subsetSize, int maxIteration) {

    MPI_Status status;
    MPI_Request	send_request;
    int senderRank = mpiParams.worldRank + subsetSize;
    int iteration = 1;

    while (iteration <= maxIteration) {

        MPI_Isend(&cells[0], subsetSize, mpiCellType, senderRank, mpiParams.worldRank, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, &status);

        MPI_Recv(&cells[0], subsetSize, mpiCellType, senderRank, senderRank, MPI_COMM_WORLD, &status);

        if (++iteration <= maxIteration) {

            cells = computeProblemTwo(cells, subsetSize, iteration);
            iteration++;
        }
    }

    return cells;
}

void computeProblemTwoBalancerTwo(struct MPIParams mpiParams, int subsetSize, int maxIterations) {

    if (maxIterations > 0) {

        int iteration = 0;
        MPI_Status status;
        MPI_Request send_request;
        int senderRank = mpiParams.worldRank - subsetSize;
        struct Cell* subsetCells = (struct Cell*)malloc(sizeof(struct Cell) * subsetSize);

        do {

            MPI_Recv(&subsetCells[0], subsetSize, mpiCellType, senderRank, senderRank, MPI_COMM_WORLD, &status);

            iteration++;
            struct Cell* subProcessedCells = computeProblemTwo(subsetCells, subsetSize, iteration);

            MPI_Isend(&subProcessedCells[0], subsetSize, mpiCellType, senderRank, mpiParams.worldRank, MPI_COMM_WORLD, &send_request);
            MPI_Wait(&send_request, &status);
        }
        while (++iteration < maxIterations);
    }
}

struct Cell* computeProblem(struct MPIParams mpiParams, struct Cell* cells,
                            int subsetSize, struct TransformationProperties properties) {

    if (properties.problemToSolve == 1) {

        return computeProblemOne(cells, subsetSize, properties.iterations);
    }
    else {

        return computeProblemTwoBalancerOne(mpiParams, cells, subsetSize, properties.iterations);
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

void report(struct MPIParams mpiParams, struct TransformationProperties properties,
            struct Cell* processedCells, clock_t time) {

    if (isRootProcess(mpiParams.worldRank)) {

        printCells(processedCells, properties, clock() - time);
    }
}

void processProblemCore(struct MPIParams mpiParams, struct TransformationProperties properties,
                        int subsetSize, bool canScatter, clock_t time) {

    if (canScatter) {

        struct Cell* subsetCells = (struct Cell*)malloc(sizeof(struct Cell) * subsetSize);

        MPI_Scatter(mpiParams.initData, subsetSize, mpiCellType, subsetCells, subsetSize, mpiCellType, 0, mpiParams.topology);

        struct Cell* subProcessedCells = computeProblem(mpiParams, subsetCells, subsetSize, properties);
        struct Cell* processedCells = NULL;

        if (isRootProcess(mpiParams.worldRank)) {

            processedCells = (struct Cell*) malloc(sizeof(struct Cell) * mpiParams.dataSize);
        }

        MPI_Gather(&subProcessedCells[0], subsetSize, mpiCellType, processedCells, subsetSize, mpiCellType, 0, mpiParams.topology);

        report(mpiParams, properties, processedCells, time);
        freeMemory(mpiParams, processedCells, subProcessedCells);

        MPI_Barrier(mpiParams.topology);
        MPI_Finalize();
    }
    else {

        if (mpiParams.worldRank < subsetSize * 2) {

            computeProblemTwoBalancerTwo(mpiParams, subsetSize, properties.iterations);
        }

        MPI_Finalize();
    }
}

void processParallelProblemOne(struct MPIParams mpiParams, struct TransformationProperties properties, clock_t time) {

    int subsetSize = mpiParams.dataSize / mpiParams.worldSize;

    mpiParams.topology = MPI_COMM_WORLD;
    mpiParams.topologyRank = mpiParams.worldRank;
    mpiParams.topologySize = mpiParams.worldSize;

    processProblemCore(mpiParams, properties, subsetSize, true, time);
}

void processParallelProblemTwo(struct MPIParams mpiParams, struct TransformationProperties properties, clock_t time) {

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

    bool canScatter = mpiParams.worldRank < processorsNeeded;

    processProblemCore(mpiParams, properties, subsetSize, canScatter, time);
}

int main(int argc, char *argv[]) {

    if (argc == 4) {

        clock_t startSolvingProblems;
        struct TransformationProperties properties = buildProperties(argv);
        struct MPIParams mpiParams = initMPI(argc, argv, properties);

        if (mpiParams.worldRank == 0) {

            printf("\nStarting parallel process...\n");
            startSolvingProblems = clock();
        }

        if (properties.problemToSolve == 1) {

            processParallelProblemOne(mpiParams, properties, startSolvingProblems);
        }
        else if (properties.problemToSolve == 2 && mpiParams.worldSize >= (properties.sizeY * 2)) {

            processParallelProblemTwo(mpiParams, properties, startSolvingProblems);
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