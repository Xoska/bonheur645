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

/**
 * Permet de présenter les résultats suite à la résolution du problème de transfert
 * de chaleur en 2D d'une plaque chauffante parallèlement. Contrairement à la
 * fonction utilisée pour la résolution séquentielle, cette fonction a besoin d'un tableau
 * à dimension unique de cellules et non une matrice.
 */
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

/**
 * Permet de peupler les valeurs adjacentes nécessaires à une cellule pour la résolution
 * du problème de transfert de chaleur en 2D d'une plaque chauffante. Par défaut, la valeur
 * adjacente est zéro, mais elle peut changée si la position de la cellule de base dans la matrice
 * logique est située à plus de 2 épaisseurs. Pour maximiser la performance, on veut effectuer
 * le moins de traitement possible, donc on calculera la valeur adjacente juste si nécessaire.
 */
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

/**
 * Permet d'initiliser le tableau de cellules à partir d'une matrice. Chaque cellules auront leur position
 * et valeur initiale en fonction du problème à résoudre. Ainsi, on transforme une matrice à un tableau à
 * une seule dimension pour faciliter la répartition des cellules à travers les processeurs. Par contre, une
 * cellule conserve sa position logique dans une matrice.
 */
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

/**
 * Permet de résoudre le problème de transfert de chaleur en 2D d'une plaque chauffante pour la
 * résolution parallèle. Chaque processeurs auront un ensemble de cellules à traiter dépendamment
 * de leur nombre dans le groupe et la taille des données.
 */
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

/**
 * Permet de créer la nouvelle structure MPI pour le passage de variables ayant la structure Cell
 */
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

/**
 * Permet de récuperer le rang d'un processeur dans son communicateur.
 */
int getCommRank(MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(comm, &rank);

    return rank;
}

/**
 * Permet de récupérer la taille de son groupe de processeurs selon son communicateur
 */
int getCommSize(MPI_Comm comm) {

    int size;
    MPI_Comm_size(comm, &size);

    return size;
}

/**
 * Permet d'initialiser la topologie globale du système. Ainsi, on récupère communicateur global, sont rang,
 * sa taille et la taille de ses données associées à traiter éventuellement. Pour calculer la taille des données,
 * on compte le nombre de cellule dans la matrice initiale après avoir enlever une couche d'épaisseur. On enlève
 * une couche d'épaisseur, car il est inutile de calculer des valeurs qui seront toujours zéro.
 */
struct Topology initWorldTopology(struct TransformationProperties properties) {

    struct Topology worldTopology;

    worldTopology.comm = MPI_COMM_WORLD;
    worldTopology.rank = getCommRank(worldTopology.comm);
    worldTopology.size = getCommSize(worldTopology.comm);
    worldTopology.dataSize = (properties.sizeX - 2) * (properties.sizeY - 2);

    return worldTopology;
}

/**
 * Permet d'initialiser un groupe de processeur dont son nombre sera maximiser en fonction du nombre
 * de cellules à traiter. Il sera toujours possible de diviser entièrement le nombre de données à
 * traiter de ce groupe par son nombre de processeur alloué.
 */
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

/**
 * Permet d'initialiser un groupe de processeur correspondant au restants de division du nombre de
 * cellules total à traiter par le nombre de processeur. Ainsi, On se crer 2 groupes distincts
 * pour la gestion des restes. Son nombre sera généralement beaucoup plus petit que l'autre
 * groupe principal. Par exemple, pour 24 cellules et 5 processeurs, le groupe principale
 * aura 20 cellules réparties du 5 processeurs et ce groupe aura 4 cellules réparties sur
 * 4 processeurs.
 */
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

/**
 * Permet d'initialiser toutes les topologies, de calculer le nombre de restant et d'initliser
 * les données à traiter. Il s'agit de l'initialisation de tous les paramètres nécessaires
 * pour l'utilisation de MPI au cours de l'exécution du système.
 */
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

/**
 * Permet d'initialiser un tableau à une dimension à une taille variable.
 */
struct Cell* initCellsOfSize(int size) {

    return (struct Cell*) malloc(sizeof(struct Cell) * size);
}

/**
 * Permet d'extraire des cellules d'un tableau à une dimension en fonction d'une position de
 * départ et une position de fin. Par exemple, si on a 24 cellules, cette fonction sera utilisée
 * pour séparer le tableau principal des données pour le groupe principal et le groupe des restants,
 * donc elle sera appelée d'abord pour séparer le tableau de 0 à 20, puis de 21 à 24 pour les restants.
 *
 */
struct Cell* extractCells(struct Cell* cells, int posStart, int posEnd) {

    int length = posEnd - posStart, i, pos = 0;
    struct Cell* newCells = initCellsOfSize(length);

    for (i = posStart; i < posEnd; ++i) {

        newCells[pos++] = cells[i];
    }

    return newCells;
}

/**
 * Il s'agit du coeur du traitement parallèle. En fonction du groupe en cours, le tableau de cellules
 * à traiter sera répartie également au travers de son groupe pour que chacun de ses processeurs résout
 * le problème, puis les cellules seront récupérées par le processeur racine. En fonction du tableau,
 * la charge sera répartie également au travers des processeurs de son groupe pour traiter les données,
 * puis le processeur racine du groupe récupère les données. Donc, cette fonction sera utilisée pour le
 * groupe principal et le groupe des restants.
 */
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

/**
 * Permet d'appeler la fonction de traitement parallèle et de calculer le nombre de cellules
 * attribué à chacun des processeur du groupe principal. Comme on peut le voir, la division
 * sera toujours entière.
 */
struct Cell* scatterMainCells(struct Topology main, struct Cell* mainCells,
                              struct TransformationProperties properties, int i) {

    int subsetSizeMain = main.dataSize / main.size;

    return scatterByTopologyCore(properties, main, mainCells, subsetSizeMain, i);
}

/**
 * Permet d'appeler la fonction de traitement parallèle et de calculer le nombre de cellules
 * attribué à chacun des processeur du groupe des restants. Comme on peut le voir, les processeurs
 * de ce groupe n'auront jamais plus que 1 élément.
 */
struct Cell* scatterLeftOverCells(struct Topology leftOver, struct Cell* leftOverCells,
                                  struct TransformationProperties properties, int i) {

    int subsetSizeLeftOver = 1;

    return scatterByTopologyCore(properties, leftOver, leftOverCells, subsetSizeLeftOver, i);
}

/**
 * Permet de récupérer les cellules traitées par le groupe principal et le groupe des restants
 * pour les ramener dans un seul tableau. Donc, on passe le tableau principal et le tableau
 * des restants pour sortir avec un seul tableau de leurs cellules. Cela est nécessaire
 * pour calculer éventuellement les cellules adjacente des données.
 */
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

/**
 * Permet de terminer le traitement parallèle. On calcule le temps final, on présente les données puis on
 * calcule le temps séquentiel, Ensuite, une comparaison des temps est présentée.
 */
void endProcess(struct Cell* data, struct TransformationProperties properties, struct timeval startTime) {

    struct timeval endTime;
    gettimeofday(&endTime, NULL);
    double timeParallel = getTimeDifferenceMS(startTime, endTime);

    printCells(data, properties, timeParallel);
    free(data);

    double timeSequential = processSequential(properties);

    compareProcesses(timeSequential, timeParallel);
}

/**
 * Il s'agit du coeur pour le traitement parallèle du programme. Cette fonction s'occupera d'abord
 * d'initialiser les différentes variables pour éviter de les re-calculer pendant le cours de
 * l'exécution parallèle, Ensuite, pour chaque pas de temps, le processeur racine divisera en 2
 * le tableau des données, soit une partie pour les données principales et une partie pour les restes.
 * Lorsque les 2 parties ont terminé leur traitement, le processseur racine s'occupe de fusionner
 * les tableaux et de reconstruire la valeur adjacente de chaque cellule.
 */
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

/**
 * Point d'entré du programme, On initialise les variables d'environnement du système
 * et on initialise les variables nécessaires pour l'exécution parallèle.
 */
int main(int argc, char *argv[]) {

    if (argc == 6) {

        struct TransformationProperties properties = buildProperties(argv);

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
    else {

        printf("Missing arguments.\n");
    }

    return 0;
}