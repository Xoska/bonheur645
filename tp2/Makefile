PROG = parallel
CFLAGS = -g -fPIC -m64 -Wall -fopenmp
LFLAGS= -fPIC -m64 -Wall -fopenmp
CC = gcc

all: $(PROG)

parallel.o: parallel.c
	$(CC) $(CFLAGS) -c parallel.c

parallel : parallel.o
	$(CC) $(LFLAGS) parallel.o -o parallel

run:
	./parallel ${ARGS}

ps:
	ps -fu $$USER

clean:
	/bin/rm -f *~
	/bin/rm -f *.o
