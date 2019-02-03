#include "myfunc.h"

float** get_randImage(int sqrt_comm_sz){

    float** I = initNewArray(sqrt_comm_sz);

    for (int i = 1; i <= (DIMENSION_X / sqrt_comm_sz); i++) {
        for (int j = 1*PIXEL_SIZE; j < (DIMENSION_Y / sqrt_comm_sz+1)*PIXEL_SIZE; j++) {
            I[i][j] = rand()/(RAND_MAX/(255+1)+1);
        }
    }
    return I;
}

float** get_Image(int my_rank, int sqrt_comm_sz){

    char input[10];
    MPI_Status status;
    float** I = initNewArray(sqrt_comm_sz);
    MPI_File image;
    MPI_Offset offset = (my_rank/sqrt_comm_sz) * (DIMENSION_X/sqrt_comm_sz) + (my_rank%sqrt_comm_sz) * (DIMENSION_Y/sqrt_comm_sz) * PIXEL_SIZE;
    MPI_Datatype filetype;

    if (PIXEL_SIZE == 1) {
        strcpy(input, "waterfall_grey_1920_2520.raw");
    }
    else{
        strcpy(input, "waterfall_1920_2520.raw");
    }

    if ( MPI_File_open( MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &image) != MPI_SUCCESS ){
        printf("Problem while opening file\n");

    }

    MPI_Type_contiguous((DIMENSION_Y/sqrt_comm_sz) * PIXEL_SIZE, MPI_CHAR, &filetype);
    MPI_Type_commit(&filetype);


    for (int i = 1; i <= DIMENSION_X/sqrt_comm_sz; i++) {

    	MPI_File_set_view(image, offset, MPI_CHAR, filetype, "native", MPI_INFO_NULL);

        for (int j = PIXEL_SIZE; j < (DIMENSION_Y/sqrt_comm_sz+1)*PIXEL_SIZE; j++) {
            char temp;
            MPI_File_read_at(image,(j-PIXEL_SIZE)*sizeof(char), &temp, 1, MPI_CHAR, &status);
            I[i][j] = temp;
        }

        offset += DIMENSION_Y* PIXEL_SIZE;
    }

    MPI_File_close(&image);

    return I;
}

void write_Image(MPI_Datatype type, int my_rank, int sqrt_comm_sz, float** I){

    MPI_Status status;
    MPI_File image;
    MPI_Offset offset = my_rank/sqrt_comm_sz * DIMENSION_X/sqrt_comm_sz + (my_rank%sqrt_comm_sz) * DIMENSION_Y*PIXEL_SIZE/sqrt_comm_sz;
    MPI_Datatype filetype;


    if ( MPI_File_open( MPI_COMM_SELF, "result.raw", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &image) != MPI_SUCCESS ){
        printf("Problem while opening file\n");
    }

    MPI_Type_contiguous((DIMENSION_Y/sqrt_comm_sz) * PIXEL_SIZE, MPI_CHAR, &filetype);
    MPI_Type_commit(&filetype);

    for (int i = 1; i <= DIMENSION_X/sqrt_comm_sz; i++) {

        MPI_File_set_view(image, offset, MPI_CHAR, filetype, "native", MPI_INFO_NULL);

        for (int j = PIXEL_SIZE; j < (DIMENSION_Y/sqrt_comm_sz+1)*PIXEL_SIZE; j++) {
            char temp = I[i][j];
            MPI_File_write_at(image,(j-PIXEL_SIZE)*sizeof(char), &temp, 1, MPI_CHAR, &status);

        }
        offset += DIMENSION_Y*PIXEL_SIZE;
    }

    MPI_File_close(&image);

}

float** make_filter(){
    int sum=0;
    float** h = malloc(3*sizeof(float*));

    for (int i = 0; i < 3; i++) {
        h[i] = malloc(3*sizeof(float));
    }

    h[0][0] = h[0][2] = h[2][0] = h[2][2] = 1;
    h[0][1] = h[1][0] = h[1][2] = h[2][1] = 2;
    h[1][1] = 4;

    for (int i = 0; i <= 2*V; i++) {
        for (int j = 0; j <= 2*V; j++) {
            sum += h[i][j];
        }
    }

    // Normalize the filter
    for (int i = 0; i <= 2*V; i++) {
        for (int j = 0; j <= 2*V; j++) {
            h[i][j] /= sum;
        }
    }
    return h;
}

void printImage(float** I, int sqrt_comm_sz, int my_rank){
    printf("\nMy_rank : %d\n", my_rank);
    for (int i = 0; i < ((DIMENSION_X / sqrt_comm_sz)+2); i++) {
        for (int j = 0; j < ((DIMENSION_Y / sqrt_comm_sz)+2)*PIXEL_SIZE; j++) {
            printf("%f ", I[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

float** initNewArray(int sqrt_comm_sz){
    float** I = malloc(((DIMENSION_X / sqrt_comm_sz)+2) * sizeof(float*));

    for (int i = 0; i < ((DIMENSION_X / sqrt_comm_sz)+2); i++) {
        I[i] = malloc(((DIMENSION_Y / sqrt_comm_sz)+2) * PIXEL_SIZE * sizeof(float));
        for (int j = 0; j < ((DIMENSION_Y / sqrt_comm_sz)+2)*PIXEL_SIZE; j++) {
            I[i][j] = 0;
        }
    }
    return I;
}

int innerConvolution(float** I, float** h, float** N, int sqrt_comm_sz){
    // Μake the convolution for each pixel 
    int changed = 0;
#   pragma omp parallel for num_threads(4)
    for (int i = 2; i < DIMENSION_X/sqrt_comm_sz; i++){
        for (int j = 2*PIXEL_SIZE; j < (DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE; j++){
            N[i][j] = 0;
            for (int p = -V; p <= V; p++){
                for (int q = -V; q <= V; q++){
                    N[i][j] += I[i-p][j-q*PIXEL_SIZE]*h[p+V][q+V];
                }
            }
            if((N[i][j] - I[i][j]) > 0.9 ||  (I[i][j] - N[i][j]) > 0.9)
            {
                changed = 1;
            }
        }
    }
    return changed;
}

int outerConvolution(float** I, float** h, float** N, int sqrt_comm_sz, int part){
    int changed = 0;

    switch (part) {
        case 0: //North
            for (int j = 2*PIXEL_SIZE; j < (DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE; j++){
                N[1][j] = 0;
                for (int p = -V; p <= V; p++){
                    for (int q = -V; q <= V; q++){
                        N[1][j] += I[1-p][j-q*PIXEL_SIZE]*h[p+V][q+V];
                    }
                }
                if((N[1][j] - I[1][j]) > 0.9 ||  (I[1][j] - N[1][j]) > 0.9)
                {
                    changed = 1;
                }
            }
            break;
        case 1: //South
            for (int j = 2*PIXEL_SIZE; j < (DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE; j++){
                N[DIMENSION_X/sqrt_comm_sz][j] = 0;
                for (int p = -V; p <= V; p++){
                    for (int q = -V; q <= V; q++){
                        N[(DIMENSION_X/sqrt_comm_sz)][j] += I[(DIMENSION_X/sqrt_comm_sz)-p][j-q*PIXEL_SIZE]*h[p+V][q+V];
                    }
                }
                if((N[DIMENSION_X/sqrt_comm_sz][j] - I[DIMENSION_X/sqrt_comm_sz][j]) > 0.9 ||  (I[DIMENSION_X/sqrt_comm_sz][j] - N[DIMENSION_X/sqrt_comm_sz][j]) > 0.9)
                {
                    changed = 1;
                }
            }
            break;
        case 2: //East
            for (int i = 2; i < DIMENSION_X/sqrt_comm_sz; i++){
            	for (int z = 0; z < PIXEL_SIZE; z++)
            	{
                	N[i][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE + z] = 0;
                	for (int p = -V; p <= V; p++){
                    	for (int q = -V; q <= V; q++){
                    	    N[i][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE + z] += I[i-p][DIMENSION_Y/sqrt_comm_sz-q*PIXEL_SIZE]*h[p+V][q+V];
                    	}
                	}
                    if((N[i][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z] - I[i][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z]) > 0.9 ||  (I[i][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z] - N[i][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z]) > 0.9)
                    {
                        changed = 1;
                    }
            	}

            }
            break;
        case 3: //West
            for (int i = 2; i < DIMENSION_X/sqrt_comm_sz; i++){
            	for (int z = 0; z < PIXEL_SIZE; z++)
            	{
                	N[i][1*PIXEL_SIZE + z] = 0;
                	for (int p = -V; p <= V; p++){
                	    for (int q = -V; q <= V; q++){
                	        N[i][1*PIXEL_SIZE + z] += I[i-p][(1*PIXEL_SIZE + z)-q*PIXEL_SIZE]*h[p+V][q+V];
                   	    }
                	}
                    if((N[i][1*PIXEL_SIZE +z] - I[i][1*PIXEL_SIZE +z]) > 0.9 ||  (I[i][1*PIXEL_SIZE +z] - N[i][1*PIXEL_SIZE +z]) > 0.9)
                    {
                       changed = 1;
                    }
            	}
            }
            break;
        case 4: //NorthWest
        	for (int z= 0; z < PIXEL_SIZE; z++)
        	{
           		N[1][PIXEL_SIZE + z] = 0;
            	for (int p = -V; p <= V; p++){
                	for (int q = -V; q <= V; q++){
                    	N[1][PIXEL_SIZE + z] += I[1-p][(PIXEL_SIZE + z)-q*PIXEL_SIZE]*h[p+V][q+V];
                	}
            	}
                if((N[1][1*PIXEL_SIZE +z] - I[1][1*PIXEL_SIZE +z]) > 0.9 ||  (I[1][1*PIXEL_SIZE +z] - N[1][1*PIXEL_SIZE +z]) > 0.9)
                {
                    changed = 1;
                }
        	}
            break;
        case 5: //SouthWest
        	for (int z = 0; z < PIXEL_SIZE; z++)
        	{
            	N[DIMENSION_X/sqrt_comm_sz][1*PIXEL_SIZE + z] = 0;
            	for (int p = -V; p <= V; p++){
            	    for (int q = -V; q <= V; q++){
            	        N[DIMENSION_X/sqrt_comm_sz][1*PIXEL_SIZE + z] += I[DIMENSION_X/sqrt_comm_sz-p][(1*PIXEL_SIZE + z)-q*PIXEL_SIZE]*h[p+V][q+V];
            	    }
            	}
                if((N[DIMENSION_X/sqrt_comm_sz][1*PIXEL_SIZE +z] - I[DIMENSION_X/sqrt_comm_sz][1*PIXEL_SIZE +z]) > 0.9 ||  (I[DIMENSION_X/sqrt_comm_sz][1*PIXEL_SIZE +z] - N[DIMENSION_X/sqrt_comm_sz][1*PIXEL_SIZE +z]) > 0.9)
                {
                    changed = 1;
                }
        	}
            break;
        case 6: //NorthEast
        	for (int z = 0; z < PIXEL_SIZE; z++)
        	{
            	N[1][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE + z] = 0;
            	for (int p = -V; p <= V; p++){
            	    for (int q = -V; q <= V; q++){
            	        N[1][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE + z] += I[1-p][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE + z - q*PIXEL_SIZE]*h[p+V][q+V];
            	    }
            	}
                if((N[1][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z] - I[1][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z]) > 0.9 ||  (I[1][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z] - N[1][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z]) > 0.9)
                {
                    changed = 1;
                }
        	}
            break;
        case 7: //SouthEast
        	for (int z = 0; z < PIXEL_SIZE; z++)
        	{
            	N[DIMENSION_X/sqrt_comm_sz][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE + z] = 0;
            	for (int p = -V; p <= V; p++){
                	for (int q = -V; q <= V; q++){
                	    N[DIMENSION_X/sqrt_comm_sz][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE + z] += I[DIMENSION_X/sqrt_comm_sz-p][(DIMENSION_Y/sqrt_comm_sz)*PIXEL_SIZE+z - q*PIXEL_SIZE]*h[p+V][q+V];
                	}
            	}
                if((N[DIMENSION_X/sqrt_comm_sz][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z] - I[DIMENSION_X/sqrt_comm_sz][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z]) > 0.9 ||  (I[DIMENSION_X/sqrt_comm_sz][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z] - N[DIMENSION_X/sqrt_comm_sz][DIMENSION_Y/sqrt_comm_sz*PIXEL_SIZE +z]) > 0.9)
                {
                    changed = 1;
                }
        	}
            break;
    }

    return changed;
}

void printFilter(float** h){
    printf("\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%2.2f ", h[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void make_neighbors(int* N, int* NW, int* W, int* SW, int* S, int* SE, int* E, int* NE, int sqrt_comm_sz, int my_rank){
    if(my_rank < sqrt_comm_sz)
        *N = MPI_PROC_NULL;
    else
        *N = my_rank - sqrt_comm_sz;

    if(my_rank % sqrt_comm_sz == 0)
        *W = MPI_PROC_NULL;
    else
        *W = my_rank-1;

    if(*W == MPI_PROC_NULL || *N == MPI_PROC_NULL)
        *NW = MPI_PROC_NULL;
    else
        *NW = my_rank - sqrt_comm_sz - 1;

    if(my_rank >= ((sqrt_comm_sz*sqrt_comm_sz) - sqrt_comm_sz))
        *S = MPI_PROC_NULL;
    else
        *S = my_rank + sqrt_comm_sz;

    if(*W == MPI_PROC_NULL || *S == MPI_PROC_NULL)
        *SW = MPI_PROC_NULL;
    else
        *SW = my_rank + sqrt_comm_sz - 1;

    if(my_rank % sqrt_comm_sz == sqrt_comm_sz - 1)
        *E = MPI_PROC_NULL;
    else
        *E = my_rank+1;

    if(*E == MPI_PROC_NULL || *S == MPI_PROC_NULL)
        *SE = MPI_PROC_NULL;
    else
        *SE = my_rank + sqrt_comm_sz + 1;

    if(*E == MPI_PROC_NULL || *N == MPI_PROC_NULL)
        *NE = MPI_PROC_NULL;
    else
        *NE = my_rank - sqrt_comm_sz + 1;
}

void build_mpi_type(int n, MPI_Datatype* line, float** I, int flag){      // flag represents if we need row or column
    int* array_of_blocklenghts = malloc(n*sizeof(int));

    MPI_Aint* array_of_displacement = malloc(n*sizeof(MPI_Aint));
    MPI_Aint x1, x2;

    if(flag == 'c'){

        // Ιnitialize the row offset
        MPI_Get_address(I[0], &x1);
        MPI_Get_address(I[1], &x2);
        for (int i = 0; i < n; i++) {

            array_of_displacement[i] = i*(x2 - x1);
            array_of_blocklenghts[i] = PIXEL_SIZE;
        }
    }
    else if(flag == 'r'){

        // Ιnitialize the column offset
        MPI_Get_address(&I[0][0], &x1);
        MPI_Get_address(&I[0][1], &x2);
        for (int i = 0; i < n; i++) {
            array_of_displacement[i] = i*(x2 - x1);
            array_of_blocklenghts[i] = 1;
        }
    }

    MPI_Datatype* array_of_types = malloc(n*sizeof(MPI_Datatype));

    // Ιnitialize the array
    for (int i = 0; i < n; i++) {
        array_of_types[i] = MPI_FLOAT;
    }

    MPI_Type_create_struct(n, array_of_blocklenghts, array_of_displacement, array_of_types, line);

    MPI_Type_commit(line);
}
