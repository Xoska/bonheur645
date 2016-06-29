#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define CELL_STRUCTURE_SIZE 7

const int WAIT_TIME = 5;

struct TransformationProperties {
    int sizeX;
    int sizeY;
    int iterations;
    double discretizedTime;
    double subDivisionSize;
};

struct Topology {
    MPI_Comm comm;
    int rank;
    int size;
    int dataSize;
};

struct MPIParams {
    struct Cell* initData;
    struct Topology world;
    struct Topology main;
    struct Topology leftOver;
};

typedef struct Cell {
    int posX;
    int posY;
    double value;
    double valueMinusPosXOne;
    double valuePlusPosXOne;
    double valueMinusPosYOne;
    double valuePlusPosYOne;
} cell;

MPI_Datatype mpiCellType;

struct TransformationProperties buildProperties(char *argv[]) {

    struct TransformationProperties properties;

    properties.sizeX = atoi(argv[1]);
    properties.sizeY = atoi(argv[2]);
    properties.iterations = atoi(argv[3]);
    sscanf(argv[4], "%lf", &properties.discretizedTime);
    sscanf(argv[5], "%lf", &properties.subDivisionSize);

    return properties;
}

bool isRootProcess(int rank) {

    return rank == 0;
}

double getTimeDifferenceMS(struct timeval startTime, struct timeval endTime) {

    double elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000.0;
    elapsedTime += (endTime.tv_usec - startTime.tv_usec) / 1000.0;

    return elapsedTime;
}

int powSquare(double value) {

    return value * value;
}

void printWithFormat(float value) {

    printf("%-10.2f|", value);
}

void printProblemInitialization(char* solvingType) {

    printf("\nProblem solving type : %s.\nStarting process...\n\n", solvingType);
}

double** createArraySequential(struct TransformationProperties properties) {

    double** array;
    int i, j;

    array = (double**) malloc(properties.sizeX * sizeof(double*));
    for (i = 0; i < properties.sizeX; i++) {

        array[i] = (double*) malloc(properties.sizeY * sizeof(double));

        for (j = 0; j < properties.sizeY; j++) {

            array[i][j] = (i * (properties.sizeX - i - 1)) * (j * (properties.sizeY - j - 1));
        }
    }

    return array;
}

double** copyMatrix(struct TransformationProperties properties, double** matrix) {

    double** newMatrix;
    int i, j;

    newMatrix = (double**) malloc(properties.sizeX * sizeof(double*));
    for (i = 0; i < properties.sizeX; i++) {

        newMatrix[i] = (double*) malloc(properties.sizeY * sizeof(double));

        for (j = 0; j < properties.sizeY; j++) {

            newMatrix[i][j] = matrix[i][j];
        }
    }

    return newMatrix;
}

void printSequentialData(struct TransformationProperties properties, double** data, double elapsedTime) {

    printf("Time taken : %f ms.\n", elapsedTime);
    printf("Results : \n");

    int i, j;

    for (i = 0; i < properties.sizeX; ++i) {

        for (j = 0; j < properties.sizeY; ++j) {

            printWithFormat(data[i][j]);
        }

        printf("\n");
    }

    printf("\n\n");
}

double processSequential(struct TransformationProperties properties) {

    printProblemInitialization("Sequential");

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);

    double** processedData = createArraySequential(properties);
    double value;

    int i, j, k;
    for (k = 1; k < properties.iterations; ++k) {

        double** originalData = copyMatrix(properties, processedData);

        for (i = 0; i < properties.sizeX; ++i) {

            for (j = 0; j < properties.sizeY; ++j) {

                usleep(WAIT_TIME);

                value = 0;

                if (i > 0 && j > 0 && i < properties.sizeX - 1 && j < properties.sizeY - 1) {

                    value = (1 - 4 * properties.discretizedTime / powSquare(properties.subDivisionSize))
                            * originalData[i][j]
                            + (properties.discretizedTime / powSquare(properties.subDivisionSize))
                            * (originalData[i - 1][j] + originalData[i + 1][j] + originalData[i][j - 1] + originalData[i][j + 1]);
                }

                processedData[i][j] = value;
            }
        }
    }

    gettimeofday(&endTime, NULL);
    double elapsedTime = getTimeDifferenceMS(startTime, endTime);

    printSequentialData(properties, processedData, elapsedTime);

    return elapsedTime;
}

void printInitialization(struct TransformationProperties properties) {

    printf("\nProperties : \n");
    printf("    Size X : %d\n", properties.sizeX);
    printf("    Size Y : %d\n", properties.sizeY);
    printf("    Iterations : %d\n", properties.iterations);
    printf("    Discretized time : %f\n", properties.discretizedTime);
    printf("    Subdivision size : %f\n", properties.subDivisionSize);
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

struct Cell buildCell(int posX, int posY, double value) {

    struct Cell cell;

    cell.posX = posX;
    cell.posY = posY;
    cell.value = value;
    cell.valueMinusPosXOne = 0.00;
    cell.valuePlusPosXOne = 0.00;
    cell.valueMinusPosYOne = 0.00;
    cell.valuePlusPosYOne = 0.00;

    return cell;
}

bool isCellPadding(int i, int sizeY, int dataSize) {

    bool isPaddingHorizontal = i < sizeY || i > dataSize - sizeY;
    bool isPaddingVertical = i % sizeY == 0 || i % sizeY == sizeY - 1;

    return isPaddingHorizontal || isPaddingVertical;
}

void printCells(struct Cell* cells, struct TransformationProperties properties, double elapsedTime) {

    printf("Time taken : %f ms.\n", elapsedTime);
    printf("Results : \n");

    int i, j = 0, fullSize = properties.sizeX * properties.sizeY;

    for (i = 0; i < fullSize; ++i) {

        if (i > 0 && i % properties.sizeY == 0) {

            printf("\n");
        }

        if (isCellPadding(i, properties.sizeY, fullSize)) {

            printWithFormat(0);
        }
        else {
            printWithFormat(cells[j++].value);
        }
    }

    printf("\n\n");
}

struct Cell* addAdjacentValuesToCells(int innerSizeY, struct Cell* cells, int sizeCells) {

    int i;

    for (i = 0; i < sizeCells; ++i) {

        if (i % innerSizeY != 0) {

            cells[i].valueMinusPosXOne = cells[i - 1].value;
        }

        if (i % innerSizeY != innerSizeY - 1) {

            cells[i].valuePlusPosXOne = cells[i + 1].value;
        }

        if (i >= innerSizeY) {

            cells[i].valueMinusPosYOne = cells[i - innerSizeY].value;
        }

        if (i < sizeCells - innerSizeY) {

            cells[i].valuePlusPosYOne = cells[i + innerSizeY].value;
        }
    }

    return cells;
}

struct Cell* createCells(struct TransformationProperties properties) {

    struct Cell* cells = (struct Cell*) malloc(sizeof(struct Cell) * (properties.sizeX * properties.sizeY));

    int i = 0;
    int posX, posY;
    double value;

    for (posX = 1; posX < properties.sizeX - 1; posX++) {

        for (posY = 1; posY < properties.sizeY - 1; posY++) {

            value = (posX * (properties.sizeX - posX - 1)) * (posY * (properties.sizeY - posY - 1));
            cells[i++] = buildCell(posX, posY, value);
        }
    }

    return cells;
}

struct Cell* computeProblem(struct Cell* cells, int subsetSize, int iteration, struct TransformationProperties properties) {

    int i;

    for (i = 0; i < subsetSize; ++i) {

        usleep(WAIT_TIME);

        cells[i].value = (1 - 4 * properties.discretizedTime / powSquare(properties.subDivisionSize))
                * cells[i].value
                + (properties.discretizedTime / powSquare(properties.subDivisionSize))
                  * (cells[i].valueMinusPosXOne + cells[i].valuePlusPosXOne
                     + cells[i].valueMinusPosYOne + cells[i].valuePlusPosYOne);
/*
        printf("(1 - 4 * %f / (%f*%f)) * %f + (%f / (%f*%f)) * (%f + %f + %f + %f) = %f\n",
               properties.discretizedTime, properties.subDivisionSize, properties.subDivisionSize, cells[i].value,
               properties.discretizedTime, properties.subDivisionSize, properties.subDivisionSize,
               cells[i].valueMinusPosXOne, cells[i].valuePlusPosXOne, cells[i].valueMinusPosYOne, cells[i].valuePlusPosYOne, value);
               */
    }

    return cells;
}

void createCellStructure() {

    const int countStructureCell = CELL_STRUCTURE_SIZE;

    int blockLengths[CELL_STRUCTURE_SIZE] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[CELL_STRUCTURE_SIZE] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
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
    MPI_Comm_rank(comm, &rank);

    return rank;
}

int getCommSize(MPI_Comm comm) {

    int size;
    MPI_Comm_size(comm, &size);

    return size;
}

struct Topology initWorldTopology(struct TransformationProperties properties) {

    struct Topology worldTopology;

    worldTopology.comm = MPI_COMM_WORLD;
    worldTopology.rank = getCommRank(worldTopology.comm);
    worldTopology.size = getCommSize(worldTopology.comm);
    worldTopology.dataSize = (properties.sizeX - 2) * (properties.sizeY - 2);

    return worldTopology;
}

struct Topology initMainTopology(struct Topology worldTopology, int nbLeftOvers) {

    struct Topology mainTopology;

    MPI_Comm comm;
    MPI_Comm_split(worldTopology.comm, worldTopology.rank < worldTopology.dataSize - nbLeftOvers, worldTopology.rank, &comm) ;
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

    int nbLeftOvers = 0;

    if (mpiParams.world.dataSize >= mpiParams.world.size) {

        nbLeftOvers = mpiParams.world.dataSize % mpiParams.world.size;
    }

    mpiParams.main = initMainTopology(mpiParams.world, nbLeftOvers);
    mpiParams.leftOver = initLeftOverTopology(mpiParams.world, nbLeftOvers);

    if (isRootProcess(mpiParams.world.rank)) {

        mpiParams.initData = createCells(properties);
    }

    return mpiParams;
}

struct Cell* initCellsOfSize(int size) {

    return (struct Cell*) malloc(sizeof(struct Cell) * size);
}


struct Cell* extractCells(struct Cell* cells, int posStart, int posEnd) {

    int length = posEnd - posStart, i, pos = 0;
    struct Cell* newCells = initCellsOfSize(length);

    for (i = posStart; i < posEnd; ++i) {

        newCells[pos++] = cells[i];
    }

    return newCells;
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

    MPI_Gather(&subProcessedCells[0], subsetSize, mpiCellType, processedCells, subsetSize, mpiCellType, 0, topology.comm);

    free(subProcessedCells);

    return processedCells;
}

struct Cell* scatterMainCells(struct Topology main, struct Cell* mainCells,
                              struct TransformationProperties properties, int i) {

    int subsetSizeMain = main.dataSize / main.size;

    return scatterByTopologyCore(properties, main, mainCells, subsetSizeMain, i);
}

struct Cell* scatterLeftOverCells(struct Topology leftOver, struct Cell* leftOverCells,
                                  struct TransformationProperties properties, int i) {

    int subsetSizeLeftOver = 1;

    return scatterByTopologyCore(properties, leftOver, leftOverCells, subsetSizeLeftOver, i);
}

struct Cell* concatenateCells(struct Cell* mainCells, int mainCellsSize,
                              struct Cell* leftoverCells, int leftoverCellsSize) {

    struct Cell* data = initCellsOfSize(mainCellsSize + leftoverCellsSize);
    int pos = 0, i;

    for (i = 0; i < mainCellsSize; ++i) {

        data[pos++] = mainCells[i];
    }

    if (leftoverCellsSize > 0) {

        for (i = 0; i < leftoverCellsSize; ++i) {

            data[pos++] = leftoverCells[i];
        }
    }

    return data;
}

void endProcess(struct Cell* data, struct TransformationProperties properties, struct timeval startTime) {

    struct timeval endTime;
    gettimeofday(&endTime, NULL);
    double timeParallel = getTimeDifferenceMS(startTime, endTime);

    printCells(data, properties, timeParallel);
    free(data);

    double timeSequential = processSequential(properties);

    compareProcesses(timeParallel, timeSequential);
}
/*
void printShits(struct Cell* cells, int size, char* title) {

    printf("Voici le tableau intermediaire de : %s", title);

    int i;

    for (i = 0; i < size; ++i) {

        printf("Valeur %d : %f\n", i, cells[i].value);
    }

    printf("\n\n");
}
*/
void processProblemCore(struct MPIParams mpiParams, struct TransformationProperties properties, struct timeval startTime) {

    struct Cell* data = NULL;
    int i, innerSizeY, innerSize;
    bool hasLeftOver;

    if (isRootProcess(mpiParams.world.rank)) {

        innerSizeY = properties.sizeY - 2;
        innerSize = (properties.sizeX - 2) * innerSizeY;
        hasLeftOver = mpiParams.leftOver.dataSize > 0;

        data = mpiParams.initData;
    }

    for (i = 1; i < properties.iterations; ++i) {

        struct Cell* mainCells = NULL;
        struct Cell* leftOverCells = NULL;

        if (isRootProcess(mpiParams.world.rank)) {

            data = addAdjacentValuesToCells(innerSizeY, data, innerSize);

            if (!hasLeftOver) {

                mainCells = data;
            }
            else {

                mainCells = extractCells(data, 0, mpiParams.main.dataSize);
                leftOverCells = extractCells(data, mpiParams.main.dataSize, mpiParams.world.size);
            }
        }

        if (mpiParams.world.rank < mpiParams.main.dataSize) {

 //           printShits(mainCells, mpiParams.main.dataSize, "maincells avant scatter");
            mainCells = scatterMainCells(mpiParams.main, mainCells, properties, i);
  //          printShits(mainCells, mpiParams.main.dataSize, "maincells apres scatter");
        }

        if (mpiParams.world.rank < mpiParams.leftOver.dataSize) {

            leftOverCells = scatterLeftOverCells(mpiParams.leftOver, leftOverCells, properties, i);
        }

        MPI_Barrier(mpiParams.main.comm);
        MPI_Barrier(mpiParams.leftOver.comm);

        if (isRootProcess(mpiParams.world.rank)) {

            data = concatenateCells(mainCells, mpiParams.main.dataSize, leftOverCells, mpiParams.leftOver.dataSize);
        }
    }

    if (isRootProcess(mpiParams.world.rank)) {

        endProcess(data, properties, startTime);
    }
}

void processParallel(struct TransformationProperties properties, int argc, char *argv[]) {

    struct timeval startTime;
    struct MPIParams mpiParams = initMPI(argc, argv, properties);

    if (isRootProcess(mpiParams.world.rank)) {

        printInitialization(properties);
        printProblemInitialization("Parallel MPI");
        gettimeofday(&startTime, NULL);
    }

    processProblemCore(mpiParams, properties, startTime);

    MPI_Finalize();
}

int main(int argc, char *argv[]) {

    if (argc == 6) {

        struct TransformationProperties properties = buildProperties(argv);

        processParallel(properties, argc, argv);
    }
    else {

        printf("Missing arguments.\n");
    }

    return 0;
}