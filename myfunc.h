#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>

#define PIXEL_SIZE 1

#define V 1 //for filter
#define DIMENSION_Y 1920
#define DIMENSION_X 2520


// #define IMAGE = "image.raw"

float** get_randImage(int sqrt_comm_sz);
float** get_Image(int count, int sqrt_comm_sz);
void write_Image(MPI_Datatype type, int my_rank, int sqrt_comm_sz, float** I);

float** make_filter();
void printImage(float** I, int sqrt_comm_sz, int my_rank);
int innerConvolution(float** I, float** h, float** N, int sqrt_comm_sz);
int outerConvolution(float** I, float** h, float** N, int sqrt_comm_sz, int part);
void printFilter(float** h);
float** initNewArray(int sqrt_comm_sz);

void build_mpi_type(int count, MPI_Datatype* line, float** I, int flag);
void make_neighbors(int* N, int* NW, int* W, int* SW, int* S, int* SE, int* E, int* NE, int sqrt_comm_sz, int my_rank);
