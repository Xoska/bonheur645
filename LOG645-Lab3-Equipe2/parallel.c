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

/**
 * Constante pour le nombre de block du nouveau type MPI.
 */
#define CELL_STRUCTURE_SIZE 7

/**
 * Constante pour l'attente de 5 micro-secondes lors d'un traitement.
 */
const int WAIT_TIME = 5;

/**
 * Structure comportant toutes les informations entrées par l'utilisateurs.
 * Il s'agit des variables d'exécution.
 */
struct TransformationProperties {
    int sizeX;
    int sizeY;
    int iterations;
    double discretizedTime;
    double subDivisionSize;
};

/**
 * Chaque topologie, que ce soit world ou un groupe custom, est toujours associée à
 * une communicateur et elles ont un rang, une taille et la taille de ses données associées.
 */
struct Topology {
    MPI_Comm comm;
    int rank;
    int size;
    int dataSize;
};

/**
 * Structure comportant toutes les topologies nécessaire à l'exécution et les données d'initialisation.
 */
struct MPIParams {
    struct Cell* initData;
    struct Topology world;
    struct Topology main;
    struct Topology leftOver;
};

/**
 * Structure comportant toutes les informations associé à une cellule. Chaque processeurs recevront
 * un ensemble de cellules dont chacune contient les valeurs nécessaires pour le calcul parallèle.
 * Cette structure correspond au nouveau type MPI.
 */
typedef struct Cell {
    int posX;
    int posY;
    double value;
    double valueMinusPosXOne;
    double valuePlusPosXOne;
    double valueMinusPosYOne;
    double valuePlusPosYOne;
} cell;

/**
 * Nouveau type MPI pour le passage des cellules dans les messages.
 */
MPI_Datatype mpiCellType;

/**
 * S'occupe d'initialiser la structure des propriétés d'exécutions entrées par l'utilisateur.
 * Donc, avec les données entrées à la ligne de commande, on retourne les variables d'exécution.
 */
struct TransformationProperties buildProperties(char *argv[]) {

    struct TransformationProperties properties;

    properties.sizeX = atoi(argv[1]);
    properties.sizeY = atoi(argv[2]);
    properties.iterations = atoi(argv[3]);
    sscanf(argv[4], "%lf", &properties.discretizedTime);
    sscanf(argv[5], "%lf", &properties.subDivisionSize);

    return properties;
}

/**
 * Détermine si le rang passé en paramètre est root, c'est-à-dire un rang de zéro.
 */
bool isRootProcess(int rank) {

    return rank == 0;
}

/**
 * Retourne la différence entre 2 temps de type timeval. On passe un temps initial et
 * un temps final, et cela nous donne la quantité de temps pour passer du temps initial
 * au final.
 */
double getTimeDifferenceMS(struct timeval startTime, struct timeval endTime) {

    double elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000.0;
    elapsedTime += (endTime.tv_usec - startTime.tv_usec) / 1000.0;

    return elapsedTime;
}

/**
 * Permet de mettre au carré une valeur
 */
int powSquare(double value) {

    return value * value;
}

/**
 * Permet d'imprimer une valeur dans un format acceptable pour la présentation.
 */
void printWithFormat(float value) {

    printf("%-10.2f|", value);
}

/**
 * Permet de présenter l'initialisation d'un type de problème. Donc, le paramètre peut être soit
 * séquentiel ou parallèle.
 */
void printProblemInitialization(char* solvingType) {

    printf("\nProblem solving type : %s.\nStarting process...\n\n", solvingType);
}

/**
 * Permet de créer un tableau en double dimension et d'initialiser ses valeurs en fonction
 * du problème de transfert de chaleur en 2D d'une plaque chauffante.
 */
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

/**
 * Permet de copier toutes les valeurs d'une matrice vers une autre matrice,
 * Utile pour faire abstraction des pointeurs et garder une copie d'une matrice.
 */
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

/**
 * Permet de présenter les résultats suite à la résolution du problème de transfert
 * de chaleur en 2D d'une plaque chauffante séquentiellement. Contrairement à la
 * fonction utilisée pour la résolution parallèle, cette fonction a besoin d'une matrice
 * et non d'un tableau.
 */
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

/**
 * Nécessaire pour faire le traitement du problème de transfert de chaleur en 2D d'une plaque chauffante
 * séquentiellement. Donc, contrairement à la résolution parallèle, il s'agit surtout  d'utiliser plusieurs
 * boucles imbriquées. Tout d'abord, on initialise les valeurs et le temps, puis on résout le problème
 * naîvement en ne distribuant pas la charge en utilisant simplement des boucles. Puis, on présente
 * les données.
 */
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

/**
 * Permet de présente les variables d'exécutions entrées a l'utilisateur dans un
 * format acceptable.
 */
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

/**
 * Permet de comparer les résultats de l'exécution parallele et séquentiel
 * en calculant l'accélération.
 */
void compareProcesses(double timeSequential, double timeParallel) {

    double acceleration = timeSequential / timeParallel;

    printf("----------------------------------------------------\n\n");
    printf("Acceleration = Time sequential / Time parallel\n");
    printf("             = %f ms / %f ms\n", timeSequential, timeParallel);
    printf("             = %f\n", acceleration);
}

/**
 * Permet d'initiliser une cellule a partir d'une structure. Par défaut,
 * le valeurs adjacentes sont égales a zéro jusqu'a la preuve que la cellule n'est
 * pas située a moins de 2 épaisseurs en bordure du tableau.
 */
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

/**
 * Permet de vérifier si, a partir d'un tableau a dimension unique, une cellule
 * est logiquement situé sur la bordure du tableau. C'est utile par calculer rapidement
 * une valeur puisque nous sachons qu'une donnée sur une bordure est toujours égale a zéro,
 */
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

    compareProcesses(timeSequential, timeParallel);
}

void processProblemCore(struct MPIParams mpiParams, struct TransformationProperties properties, struct timeval startTime) {

    struct Cell* data = NULL;
    int i, innerSizeY, innerSize;

    if (isRootProcess(mpiParams.world.rank)) {

        innerSizeY = properties.sizeY - 2;
        innerSize = (properties.sizeX - 2) * innerSizeY;

        data = mpiParams.initData;
    }

    for (i = 1; i < properties.iterations; ++i) {

        struct Cell* mainCells = NULL;
        struct Cell* leftOverCells = NULL;

        if (isRootProcess(mpiParams.world.rank)) {

            data = addAdjacentValuesToCells(innerSizeY, data, innerSize);

            if (mpiParams.leftOver.dataSize == 0) {

                mainCells = data;
            }
            else {

                mainCells = extractCells(data, 0, mpiParams.main.dataSize);
                leftOverCells = extractCells(data, mpiParams.main.dataSize, mpiParams.world.dataSize);
            }
        }

        if (mpiParams.world.rank < mpiParams.main.dataSize) {

            mainCells = scatterMainCells(mpiParams.main, mainCells, properties, i);
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