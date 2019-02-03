main: main.c myfunc.c
	mpicc -g -Wall -fopenmp -o main main.c myfunc.c -lm

clean:
	-rm main

exec:
	mpiexec -f machines -n 4 ./main
