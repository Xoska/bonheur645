PROG = sequential
CFLAGS = -g -fPIC -m64 -Wall
LFLAGS= -fPIC -m64 -Wall
CC = gcc

all: $(PROG)

sequential.o: sequential.c
	$(CC) $(CFLAGS) -c sequential.c

sequential : sequential.o
	$(CC) $(LFLAGS) sequential.o -o sequential

run:
	./sequential ${ARGS}

ps:
	ps -fu $$USER

clean:
	/bin/rm -f *~
	/bin/rm -f *.o
