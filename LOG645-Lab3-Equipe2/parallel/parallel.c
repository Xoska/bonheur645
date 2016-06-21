#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define CELL_STRUCTURE_SIZE 7

struct TransformationProperties {
    int sizeX;
    int sizeY;
    int iterations;
    int discretizedTime;
    int subDivisionSize;
    int nbProc;
};

struct Topology {
    MPI_Comm comm;
    int rank;
    int size;
    int dataSize;
};

struct MPIParams {
    struct Cell* initData;
    Topology world;
    Topology main;
    Topology leftOver;
};

typedef struct Cell {
    int posX;
    int posY;
    int value;
    int valueMinusPosXOne;
    int valuePlusPosXOne;
    int valueMinusPosYOne;
    int valuePlusPosYOne;
} cell;

MPI_Datatype mpiCellType;

struct TransformationProperties buildProperties(char *argv[]) {

    struct TransformationProperties properties;

    properties.sizeX = atoi(argv[1]);
    properties.sizeY = atoi(argv[2]);
    properties.iterations = atoi(argv[3]);
    properties.discretizedTime = atoi(argv[4]);
    properties.subDivisionSize = atoi(argv[5]);
    properties.nbProc = atoi(argv[6]);

    return properties;
}

double getTimeDifferenceMS(struct timeval startTime, struct timeval endTime) {

    double elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000.0;
    elapsedTime += (endTime.tv_usec - startTime.tv_usec) / 1000.0;

    return elapsedTime;
}

struct Cell addAdjacentValuesToCell(struct Cell cell, int valueMinusPosXOne, int valuePlusPosXOne,
                                   int valueMinusPosYOne, int valuePlusPosYOne) {

    cell.valueMinusPosXOne = valueMinusPosXOne;
    cell.valuePlusPosXOne = valuePlusPosXOne;
    cell.valueMinusPosYOne = valueMinusPosYOne;
    cell.valuePlusPosYOne = valuePlusPosYOne;

    return cell;
}

struct Cell buildCell(int posX, int posY, int value) {

    struct Cell cell;

    cell.posX = posX;
    cell.posY = posY;
    cell.value = value;

    return cell;
}

bool isCellPadding(int i, int sizeX, int sizeY) {

    bool isPaddingHorizontal = i % sizeY == 0 || i % sizeY == sizeY - 1;
    bool isPaddingVertical = i % sizeX == 0 || i % sizeX == sizeX - 1;

    return isPaddingHorizontal || isPaddingVertical;
}

void printWithFormat(int value) {

    printf("%-10d|", value);
}

void printCells(struct Cell* cells, struct TransformationProperties properties, double elapsedTime) {

    printf("Problems solve type : Parallel with MPI. \n");

    printf("Time taken : %f ms.\n", elapsedTime);
    printf("Results : \n");

    int i, dataSize = properties.sizeX * properties.sizeY;

    for (i = 0; i < dataSize; ++i) {

        if (i > 0 && i % properties.sizeX == 0) {

            printf("\n");
        }

        if (isCellPadding(i, properties.sizeX, properties.sizeY)) {

            printWithFormat(0);
        }
        else {

            printWithFormat(cells[i].value);
        }
    }

    printf("\n\n");
}

struct Cell* addAdjacentValuesToCells(int innerSizeX, int innerSizeY, struct Cell* cells, int sizeCells) {

    int i;

    for (i = 0; i < size; ++i) {

        cells[i].valueMinusPosXOne = cells[i].posX > 1 ? cells[i - 1] : 0;
        cells[i].valuePlusPosXOne = cells[i].posX < innerSizeX ? cells[i + 1] : 0;
        cells[i].valueMinusPosYOne = cells[i].posY > 1 ? cells[i - innerSizeX] : 0;
        cells[i].valuePlusPosYOne = cells[i].posY < innerSizeY ? cells[i + realSizeX]: 0;
    }

    return cells;
}

struct Cell* createCells(struct TransformationProperties properties) {

    struct Cell* cells = (struct Cell*) malloc(sizeof(struct Cell) * (properties.sizeX * properties.sizeY));

    int i = 0;
    int posX, posY;

    for (posX = 1; posX < properties.sizeX - 1; posX++) {

        for (posY = 1; posY < properties.sizeY - 1; posY++) {

            int value = (posX * (properties.sizeX - posX - 1)) * (posY * (properties.sizeY - posY - 1));
            cells[i++] = buildCell(posX, posY, value);
        }
    }

    return cells;
}

struct Cell* computeProblem(struct Cell* cells, int subsetSize, int iteration, struct TransformationProperties properties) {

    int i;

    for (i = 0; i < subsetSize; ++i) {

        cells[i].value = (1 - 4 * properties.discretizedTime / pow(properties.subDivisionSize, 2))
                         * cells[i].value
                         + (properties.discretizedTime / pow(properties.subDivisionSize, 2))
                         * (cells[i].valueMinusPosXOne + cells[i].valuePlusPosXOne
                            + cells[i].valueMinusPosYOne + cells[i].valuePlusPosYOne);
    }

    return cells;
}

void createCellStructure() {

    const int countStructureCell = CELL_STRUCTURE_SIZE;

    int blockLengths[CELL_STRUCTURE_SIZE] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[CELL_STRUCTURE_SIZE] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[CELL_STRUCTURE_SIZE];

    offsets[0] = offsetof(cell, posX);
    offsets[1] = offsetof(cell, posY);
    offsets[2] = offsetof(cell, value);
    offsets[3] = offsetof(cell, valueMinusPosXOne);
    offsets[4] = offsetof(cell, valuePlusPosXOne);
    offsets[5] = offsetof(cell, valueMinusPosYOne);
    offsets[6] = offsetof(cell, valuePlusPosYOne);

    MPI_Type_create_struct(countStructureCell, blockLengths, offsets, types, &mpiCellType);
    MPI_Type_commit(&mpiCellType);
}

int getCommRank(MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(worldTopology.comm, &rank);

    return rank;
}

int getCommSize(MPI_Comm comm) {

    int size;
    MPI_Comm_size(worldTopology.comm, &size);

    return size;
}

struct Topology initWorldTopology(struct TransformationProperties properties) {

    struct Topology worldTopology;

    worldTopology.comm = MPI_COMM_WORLD;
    worldTopology.rank = getCommRank(worldTopology.comm);
    worldTopology.size = getCommSize(worldTopology.comm);
    worldTopology.dataSize = properties.sizeX * properties.sizeY - ((2 * properties.sizeX + 2 * properties.sizeY) - 4);

    return worldTopology;
}

struct Topology initMainTopology(struct Topology worldTopology, int nbLeftOvers) {

    struct Topology mainTopology;

    MPI_Comm comm;
    MPI_Comm_split(worldTopology.comm, worldTopology.rank < worldTopology.dataSize, worldTopology.rank, &comm) ;
    mainTopology.comm = comm;

    mainTopology.rank = getCommRank(comm);
    mainTopology.size = getCommSize(comm);
    mainTopology.dataSize = worldTopology.dataSize - nbLeftOvers;

    return mainTopology;
}

struct Topology initLeftOverTopology(struct Topology worldTopology, int nbLeftOvers) {

    struct Topology leftOverTopology;

    MPI_Comm comm;
    MPI_Comm_split(worldTopology.comm, worldTopology.rank < nbLeftOvers, worldTopology.rank, &comm) ;
    leftOverTopology.comm = comm;

    leftOverTopology.rank = getCommRank(comm);
    leftOverTopology.size = getCommSize(comm);
    leftOverTopology.dataSize = nbLeftOvers;

    return leftOverTopology;
}

struct MPIParams initMPI(int argc, char *argv[], struct TransformationProperties properties) {

    struct MPIParams mpiParams;

    MPI_Init(&argc, &argv);

    createCellStructure();

    mpiParams.world = initWorldTopology(properties);

    int nbLeftOvers = mpiParams.world.dataSize % properties.nbProc;

    mpiParams.main = initMainTopology(mpiParams.world, nbLeftOvers);
    mpiParams.leftOver = initLeftOverTopology(mpiParams.world, nbLeftOvers);

    if (mpiParams.world.rank == 0) {

        mpiParams.initData = createCells(properties);
    }

    return mpiParams;
}

bool isRootProcess(int worldRank) {

    return worldRank == 0;
}

struct Cell* initCellsOfSize(int size) {

    return (struct Cell*)malloc(sizeof(struct Cell) * size);
}


struct Cell* scatterByTopologyCore(struct TransformationProperties properties, struct Topology topology,
                                   struct Cell* cells, int subsetSize, int iteration) {

    struct Cell* subsetCells = initCellsOfSize(subsetSize);
    MPI_Scatter(cells, subsetSize, mpiCellType, subsetCells, subsetSize, mpiCellType, 0, topology.comm);

    struct Cell* subProcessedCells = computeProblem(subsetCells, subsetSize, iteration, properties);
    struct Cell* processedCells = NULL;

    if (isRootProcess(topology.rank)) {

        processedCells = initCellsOfSize(topology.dataSize);
    }

    MPI_Gather(&subProcessedCells[0], subsetSize, mpiCellType, processedCells, subsetSize, mpiCellType, 0, mpiParams.topology);

    free(subProcessedCells);

    return processedCells;
}

struct Cell* scatterMainCells(struct MPIParams mpiParams, struct Cell* data,
                              struct TransformationProperties properties, int i) {

    mainCells = malloc(mpiParams.main.dataSize * sizeof(int));
    memcpy(mainCells, data, mpiParams.main.dataSize * sizeof(int));

    int subsetSizeMain = mpiParams.main.dataSize / properties.nbProc;

    return scatterByTopologyCore(properties, mpiParams.main, mainCells, subsetSizeMain, i);
}

struct Cell* scatterLeftOverCells(struct MPIParams mpiParams, struct Cell* data,
                                  struct TransformationProperties properties, int i) {

    leftOverCells = malloc(mpiParams.leftOver.dataSize * sizeof(int));
    memcpy(leftOverCells, data + mpiParams.main.dataSize, mpiParams.leftOver.dataSize * sizeof(int));

    int subsetSizeLeftOver = 1;

    return scatterByTopologyCore(properties, mpiParams.leftOver, leftOverCells, subsetSizeLeftOver, i);
}

struct Cell* concatenateCells(int size, struct Cell* mainCells, int mainCellsSize,
                              struct Cell* leftoverCells, int leftoverCellsSize) {

    struct Cell* data = malloc(sizeof(struct Cell) * mainCellsSize + leftoverCellsSize);

    memcpy(data, mainCells, mainCellsSize * sizeof(struct Cell));

    if (leftoverCellsSize > 0) {

        memcpy(data + mainCellsSize, leftoverCells, leftoverCellsSize * sizeof(struct Cell));
    }

    return data;
}

double processProblemCore(struct MPIParams mpiParams, struct TransformationProperties properties, struct timeval startTime) {

    struct Cell* data = mpiParams.initData;

    int i, innerSizeX = properties.sizeX - 2, innerSizeY = properties.sizeY - 2;
    int innerSize = innerSizeX * innerSizeY;
    struct timeval endTime;

    for (i = 1; i < properties.iterations; ++i) {

        data = addAdjacentValuesToCells(innerSizeX, innerSizeY, data, innerSize);

        struct Cell* mainCells = NULL;

        if (mpiParams.world.rank < mpiParams.main.dataSize) {

            scatterMainCells(mpiParams, data, properties, i);
        }

        struct Cell* leftOverCells = NULL;

        if (worldTopology.rank < mpiParams.leftOver.dataSize) {

            scatterLeftOverCells(mpiParams, data, properties, i);
        }

        data = concatenateCells(mainCells, leftOverCells);

        free(mainCells);
        free(leftOverCells);
    }

    if (isRootProcess(mpiParams.world.rank)) {

        gettimeofday(&endTime, NULL);
        double elapsedTime = getTimeDifferenceMS(startTime, endTime) ;

        printCells(data, properties, elapsedTime);
    }

    free(data);

    return elapsedTime;
}

double processParallel(struct TransformationProperties properties) {

    struct timeval startTime;
    struct MPIParams mpiParams = initMPI(argc, argv, properties);

    if (mpiParams.worldRank == 0) {

        printf("\nStarting parallel process with MPI...\n");
        gettimeofday(&startTime, NULL);
    }

    double elapsedTIme = processParallel(properties);

    MPI_Finalize();

    return elapsedTIme;
}

int main(int argc, char *argv[]) {

    if (argc == 6) {

        struct TransformationProperties properties = buildProperties(argv);

        double timeParallel = processParallel(properties);
    }
    else {

        printf("Missing arguments.\n");
    }

    return 0;
}