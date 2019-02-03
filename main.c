#include "myfunc.h"

int main(){

    int comm_sz;
    int my_rank;
    int n;
    double start, finish, elapsed, local_elapsed;
    MPI_Datatype column, row;
    MPI_Request request1[8], request2[8];
    MPI_Status status;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    const int sqrt_comm_sz = sqrt(comm_sz);

    // Define neighbors
    int N, NW, W, SW, S, SE, E, NE;
    make_neighbors(&N, &NW, &W, &SW, &S, &SE, &E, &NE, sqrt_comm_sz, my_rank);

    // Make the filter
    float** h = make_filter();
    // printFilter(h);

    // Get the image
    float** I = get_randImage(sqrt_comm_sz);
    // float** I = get_Image(my_rank, sqrt_comm_sz);

    // Make space for new array
    float** NI = initNewArray(sqrt_comm_sz);

    // Find the number of the rows/columns of the process
    n = (DIMENSION_Y / sqrt_comm_sz)*PIXEL_SIZE;
    build_mpi_type(n, &row, I, 'r');

    n = DIMENSION_X / sqrt_comm_sz;
    build_mpi_type(n, &column, I, 'c');

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    int k = 0;
    int local_change, change = 1;
    while(change && k<100){
        int flag[9] = {0};



        MPI_Isend(&I[1][1], 1, row, N, 0, MPI_COMM_WORLD, &request1[0]);
        MPI_Isend(&I[DIMENSION_X/sqrt_comm_sz][1], 1, row, S, 0, MPI_COMM_WORLD, &request1[1]);
        MPI_Isend(&I[1][DIMENSION_Y/sqrt_comm_sz], 1, column, E, 0, MPI_COMM_WORLD, &request1[2]);
        MPI_Isend(&I[1][1], 1, column, W, 0, MPI_COMM_WORLD, &request1[3]);
        MPI_Isend(&I[1][1], PIXEL_SIZE, MPI_FLOAT, NW, 0, MPI_COMM_WORLD, &request1[4]);
        MPI_Isend(&I[1][DIMENSION_Y/sqrt_comm_sz], PIXEL_SIZE, MPI_FLOAT, NE, 0, MPI_COMM_WORLD, &request1[5]);
        MPI_Isend(&I[DIMENSION_X/sqrt_comm_sz][1], PIXEL_SIZE, MPI_FLOAT, SW, 0, MPI_COMM_WORLD, &request1[6]);
        MPI_Isend(&I[DIMENSION_X/sqrt_comm_sz][DIMENSION_Y/sqrt_comm_sz], PIXEL_SIZE, MPI_FLOAT, SE, 0, MPI_COMM_WORLD, &request1[7]);

        MPI_Irecv(&I[0][1], 1, row, N, 0, MPI_COMM_WORLD, &request2[0]);
        MPI_Irecv(&I[DIMENSION_X/sqrt_comm_sz + 1][1], 1, row, S, 0, MPI_COMM_WORLD, &request2[1]);
        MPI_Irecv(&I[1][DIMENSION_Y/sqrt_comm_sz + 1], 1, column, E, 0, MPI_COMM_WORLD, &request2[2]);
        MPI_Irecv(&I[1][0], 1, column, W, 0, MPI_COMM_WORLD, &request2[3]);
        MPI_Irecv(&I[0][0], PIXEL_SIZE, MPI_FLOAT, NW, 0, MPI_COMM_WORLD, &request2[4]);
        MPI_Irecv(&I[DIMENSION_X/sqrt_comm_sz +1][0], PIXEL_SIZE, MPI_FLOAT, SW, 0, MPI_COMM_WORLD, &request2[5]);
        MPI_Irecv(&I[0][DIMENSION_Y/sqrt_comm_sz +1], PIXEL_SIZE, MPI_FLOAT, NE, 0, MPI_COMM_WORLD, &request2[6]);
        MPI_Irecv(&I[DIMENSION_X/sqrt_comm_sz + 1][DIMENSION_Y/sqrt_comm_sz +1], PIXEL_SIZE, MPI_FLOAT, SE, 0, MPI_COMM_WORLD, &request2[7]);


        local_change = 0;
        // Make the Convolution for the inner box
        local_change += innerConvolution(I, h, NI, sqrt_comm_sz);

        while(flag[8] != 8){

            if(!flag[0]){
                MPI_Test(&request2[0], &flag[0], &status);
                // Make the convolution for the North row
                if(flag[0]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 0);
                    flag[8]++;
                }
            }
            if(!flag[1]){
                MPI_Test(&request2[1], &flag[1], &status);
                // Make the convolution for the South row
                if(flag[1]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 1);
                    flag[8]++;
                }
            }
            if(!flag[2]){
                MPI_Test(&request2[2], &flag[2], &status);
                // Make the convolution for the East row
                if(flag[2]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 2);
                    flag[8]++;
                }
            }
            if(!flag[3]){
                MPI_Test(&request2[3], &flag[3], &status);
                // Make the convolution for the West row
                if(flag[3]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 3);
                    flag[8]++;
                }
            }
            if(!flag[4] && flag[0] && flag[3]){
                MPI_Test(&request2[4], &flag[4], &status);
                // Make the convolution for the NorthWest row
                if(flag[4]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 4);
                    flag[8]++;
                }
            }
            if(!flag[5] && flag[1] && flag[3]){
                MPI_Test(&request2[5], &flag[5], &status);
                // Make the convolution for the SouthWest row
                if(flag[5]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 5);
                    flag[8]++;
                }
            }
            if(!flag[6] && flag[0] && flag[2]){
                MPI_Test(&request2[6], &flag[6], &status);
                // Make the convolution for the NorthEast row
                if(flag[6]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 6);
                    flag[8]++;
                }
            }
            if(!flag[7] && flag[1] && flag[2]){
                MPI_Test(&request2[7], &flag[7], &status);
                // Make the convolution for the SouthEast row
                if(flag[7]){
                    local_change += outerConvolution(I, h, NI, sqrt_comm_sz, 7);
                    flag[8]++;
                }
            }
        }

        // wait for sends
        for (int i = 0; i < 8; i++) {
            MPI_Wait(&request1[i], &status);
        }

        // Check if there is a change in the arrays
        MPI_Allreduce(&local_change, &change, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Swapping the arrays
        float **temp;
        temp = I;
        I = NI;
        NI = temp;

        k++;
    }

    finish = MPI_Wtime();
    local_elapsed = finish - start;

    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(my_rank == 0)
        printf("\n\nMax Time Elapsed : %f\n", elapsed);

    // write_Image(row, my_rank, sqrt_comm_sz, I);
    MPI_Type_free(&row);
    MPI_Type_free(&column);
    MPI_Finalize();

    for (int i = 0; i < DIMENSION_X/sqrt_comm_sz +2; ++i)
    {
        free(I[i]);
        free(NI[i]);
    }
    free(I);
    free(NI);

    return 0;
}
