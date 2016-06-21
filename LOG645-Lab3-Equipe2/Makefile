PROG = parallel
CFLAGS = -g -fPIC -m64 -Wall
LFLAGS= -fPIC -m64 -Wall
CC = mpicc

all: $(PROG)

parallel.o: parallel.c
	$(CC) $(CFLAGS) -c parallel.c

parallel : parallel.o
	$(CC) $(LFLAGS) parallel.o -o parallel

run:
	mpirun --hostfile hostfile -np ${NP} parallel ${ARGS}

ps:
	ps -fu $$USER

clean:
	/bin/rm -f *~
	/bin/rm -f *.o

test-mpi:
	mpirun --hostfile hostfile -np 16 parallel
