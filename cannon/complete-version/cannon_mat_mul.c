#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define input_1 "./input/ex4/A.txt"
#define input_2 "./input/ex4/B.txt"
#define ROOT 0

/**
 *
 * SQUARED GRID ONLY!
 * SOURCE: https://www3.nd.edu/~zxu2/acms60212-40212/Lec-07-3.pdf
 *
 */

int allocate_mat(int*** mat, int num_rows, int num_cols) {
    int* p = (int*) malloc(sizeof(int) * num_rows * num_cols);;
    if(!p) return -1; // False to allocate.

    *mat = (int**) malloc(num_rows*sizeof(int*));
    if(!mat) {
        free(p);
        return -1; // False to allocate.
    }

    for(int i = 0; i < num_rows; ++i)
        (*mat)[i] = &(p[i * num_cols]);

    return 0; // Succeed allocating.
}

int free_mat(int*** mat) {
    free(&((*mat)[0][0]));
	free(*mat);
	return 0;
}

void mat_mul(int** a, int** b, int*** c, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            int val = 0;
            for(int k = 0; k < n; ++k)
                val += a[i][k]*b[k][j];
            (*c)[i][j] = val;
        }
    }
}

void print_mat(int** mat, int num_rows, int num_cols) {
    for(int i = 0; i < num_rows; ++i) {
        for(int j = 0; j < num_cols; ++j)
            printf("%d ", mat[i][j]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    int **mat1, **mat2, **mat3;
    int meta_data[4]; // Meta data buffer for send-recv.

    MPI_Init(&argc, &argv); // Start MPI.
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get MPI size.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get MPI rank.

    // Communication params
    const int TAG = 1;

    
    // For math stuffs..
    int i, j, k;
    int process_dim, block_dim, num_rows, num_cols;

    FILE *fp;

    double p_sqrt = sqrt((double) size);

    if(rank == ROOT) { // Root MPI Process handles reading I/O.
        int counts, n;
        char c;

        /**
         * Fetching input matrices from .txt files.
         * Using integer type for simplicity.
         */
        num_rows = 0;
        counts = 0;

        fp = fopen(input_1, "r");
        if(!fp) {
            printf("[ERROR] Invalid input_1.\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }

        while(fscanf(fp, "%d", &n) != EOF) {
            c = fgetc(fp);
            if (c == '\n') num_rows++;
            counts++;
        }

        num_cols = counts/num_rows;
        if(num_rows != num_cols) {
            free_mat(&mat1);
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        fclose(fp);
        fp = fopen(input_1, "r");

        if(allocate_mat(&mat1, num_rows, num_cols) == -1)
            return -1;

        // Read matrix 1.
        for(int i = 0; i < num_rows; ++i) {
            for(int j = 0; j < num_cols; ++j) {
                fscanf(fp, "%d", &n);
                mat1[i][j] = n;
            }
        }

        fclose(fp);
        num_rows = 0;
        counts = 0;

        fp = fopen(input_2, "r");
        if(!fp) {
            printf("[ERROR] Invalid input_2.\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }

        while (fscanf(fp, "%d", &n) != EOF) {
            c = fgetc(fp);
            if (c == '\n') num_rows++;
            counts++;
        }

        num_cols = counts/num_rows;
        if(num_rows != num_cols) {
            free_mat(&mat2);
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        fclose(fp);
        fp = fopen(input_2, "r");

        if(allocate_mat(&mat2, num_rows, num_cols) == -1)
            return -1;

        // Read matrix 2.
        for(int i = 0; i < num_rows; ++i) {
            for(int j = 0; j < num_cols; ++j) {
                fscanf(fp, "%d", &n);
                mat2[i][j] = n;
            }
        }
        fclose(fp);
        n = num_rows;

        // Allocate matrix for results.
        if(allocate_mat(&mat3, num_rows, num_cols) == -1)
            MPI_Abort(MPI_COMM_WORLD, 0);

        // Calculate meta-data (partitioned block size..).
        process_dim = (int) p_sqrt; // sqrt(p) --> But using int for evenly divided block.
        block_dim = num_rows/process_dim; // n/sqrt(p)

        /** Sanity check.
        printf("----ROOT----\n");
        printf("Process dim: %d, Block dim: %d\n", process_dim, block_dim);
        printf("Size: %d, Process dim: %d\n", size, process_dim);
        */

        if(size % process_dim != 0 || process_dim * block_dim != num_rows) {
            printf("Input block and number processes are not evenly divided..\n");
            free_mat(&mat1);
            free_mat(&mat2);
            free_mat(&mat3);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Passing meta data to broadcast..
        meta_data[0] = process_dim;
        meta_data[1] = block_dim;
        meta_data[2] = num_rows;
        meta_data[3] = num_cols;
    }
   
    MPI_Bcast(&meta_data, 4, MPI_INT, 0, MPI_COMM_WORLD); // MPI Receiving meta-data from root's broadcasting (One-to-all communication, collective op).
    // Assigning meta-data to other processes.
    process_dim = meta_data[0];
    block_dim = meta_data[1];
    num_rows = meta_data[2];
    num_cols = meta_data[3];

    int dims[2], periods[2]; // For MPI Cartesian grid.
    
    // Make both dimension connected.
    periods[0] = 1;
    periods[1] = 1;

    MPI_Dims_create(size, 2, dims); // Using built in MPI Cartesian grid topartition, Just for sanity check!
    if(dims[0] != dims[1] || dims[0] != process_dim) {
        printf("Cannot partition blocks..\n");
        if(rank == ROOT) {
            free_mat(&mat1);
            free_mat(&mat2);
            free_mat(&mat3);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /** Sanity check..
    printf("dims: %d %d\n", dims[0], dims[1]);
    printf("periods: %d %d\n", periods[0], periods[1]);
    printf("----\n");
    */

    // Create MPI Cartesian grid new comm world..
    MPI_Comm comm_world;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_world);

    int** local_mat1 = NULL;
    int** local_mat2 = NULL;
    int** local_mat3 = NULL;
    if(allocate_mat(&local_mat1, block_dim, block_dim) == -1) {
        printf("[ERROR] Cannot allocate local matrix..\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
        if(rank == ROOT) {
            free_mat(&mat1);
            free_mat(&mat2);
            free_mat(&mat3);
        }
    }

    if(allocate_mat(&local_mat2, block_dim, block_dim) == -1) {
        printf("[ERROR] Cannot allocate local matrix..\n");
        free_mat(&local_mat1);
        if(rank == ROOT) {
            free_mat(&mat1);
            free_mat(&mat2);
            free_mat(&mat3);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Datatype type, subarray_type;
    int global_size[2] = {num_rows, num_cols}; // Global block size.
    int local_size[2] = {block_dim, block_dim}; // Local block size.
    int start[2] = {0, 0}; // Start coordinate for subarray.

    // Create new datatype for MPI Communications.
    MPI_Type_create_subarray(2, global_size, local_size, start, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, block_dim*sizeof(int), &subarray_type); // Row vector type.
    MPI_Type_commit(&subarray_type); // Commit for usage.

    int* global_mat1 = NULL;
    int* global_mat2 = NULL;
    int* global_mat3 = NULL;
    if(rank == ROOT) {
        global_mat1 = &(mat1[0][0]);
        global_mat2 = &(mat2[0][0]);
        global_mat3 = &(mat3[0][0]);
    }

    int send_counts[size];
    int displacements[size];

    // Scatter the local block to all MPI processes.
    if(rank == ROOT) { // Sender.

        for(int i = 0; i < size; ++i)
            send_counts[i] = 1; // Each processor should have 1 local block only.
        // Decide start index for each partitioned block.
        int disp = 0;
        for(int i = 0; i < process_dim; ++i) {
            for(int j = 0; j < process_dim; ++j) {
                displacements[i*process_dim+j] = disp;
                disp++;
            }
            disp += (block_dim-1)*process_dim;
        }

        /** Sanity Check.
        for(int i = 0; i < size; ++i)
            printf("%d ", displacements[i]);
        printf("\n");
        */

        /**
         * Explain: Why num_rows*num_cols/size?
         * Yes, we have n/p_sqrt for each dimension
         * => The total size we need is (n/p_qsqt) ** 2
         * => n**2 / p = num_rows * num_cols / size
         */

        // Send message (from ROOT).
        MPI_Scatterv(global_mat1, send_counts, displacements, subarray_type, &(local_mat1[0][0]), num_rows*num_cols/size, MPI_INT, ROOT, MPI_COMM_WORLD);
        MPI_Scatterv(global_mat2, send_counts, displacements, subarray_type, &(local_mat2[0][0]), num_rows*num_cols/size, MPI_INT, ROOT, MPI_COMM_WORLD);
    } else { // Receiver.
        // Receive message.
        MPI_Scatterv(NULL, NULL, NULL, subarray_type, &(local_mat1[0][0]), num_rows*num_cols/size, MPI_INT, ROOT, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, subarray_type, &(local_mat2[0][0]), num_rows*num_cols/size, MPI_INT, ROOT, MPI_COMM_WORLD);
    }

    if(allocate_mat(&local_mat3, block_dim, block_dim) == -1) {
        MPI_Type_free(&subarray_type);
        free_mat(&local_mat1);
        free_mat(&local_mat2);
        if(rank == ROOT) {
            free_mat(&mat1);
            free_mat(&mat2);
            free_mat(&mat3);
        }
        printf("[Error] Cannot allocate local matrix.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    
    int coord[2];
    MPI_Cart_coords(comm_world, rank, 2, coord); // Get current coordinate in Cartesian topology 2D grid.

    // Shift left coord[0] steps.
    int left, right;
    MPI_Cart_shift(comm_world, 1, (-1)*coord[0], &right, &left);
    MPI_Sendrecv_replace(local_mat1[0], block_dim*block_dim, MPI_INT,
            left, TAG, // Receiving params
            right, TAG, // Sending params
            comm_world, MPI_STATUS_IGNORE);
    
    // Shift up coord[1] steps.
    int up, down;
    MPI_Cart_shift(comm_world, 0, (-1)*coord[1], &down, &up);
    MPI_Sendrecv_replace(local_mat2[0], block_dim*block_dim, MPI_INT,
            up, TAG, // Receiving params
            down, TAG, // Sending params
            comm_world, MPI_STATUS_IGNORE);

    int** tmp_results = NULL;
    if(allocate_mat(&tmp_results, block_dim, block_dim) == -1) {
        MPI_Type_free(&subarray_type);
        free_mat(&local_mat1);
        free_mat(&local_mat2);
        free_mat(&local_mat3);
        if(rank == ROOT) {
            free_mat(&mat1);
            free_mat(&mat2);
            free_mat(&mat3);
        }
        printf("[Error] Cannot allocate local matrix.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    
    // Assign default value (zeroes) to Local mat3.
    for(int i = 0; i < block_dim; ++i) {
        for(int j = 0; j < block_dim; ++j)
            local_mat3[i][j] = 0;
    }

    // Do the math.
    for(int k = 0; k < process_dim; ++k) { // process_dim = (int) p_sqrt

        // Naive matrix mult.
        mat_mul(local_mat1, local_mat2, &tmp_results, block_dim);

        // Append the results.
        for(int i = 0; i < block_dim; ++i) {
            for(int j = 0; j < block_dim; ++j)
                local_mat3[i][j] += tmp_results[i][j];
        }
        
        // Shift local_mat1 left 1 steps.
        MPI_Cart_shift(comm_world, 1, -1, &right, &left);
        MPI_Sendrecv_replace(local_mat1[0], block_dim*block_dim, MPI_INT,
                left, TAG, // Receiving params
                right, TAG, // Sending params
                comm_world, MPI_STATUS_IGNORE);
        
        // Shift local_mat2 up 1 steps.
        MPI_Cart_shift(comm_world, 0, -1, &down, &up);
        MPI_Sendrecv_replace(local_mat2[0], block_dim*block_dim, MPI_INT,
                up, TAG, // Receiving params
                down, TAG, // Sending params
                comm_world, MPI_STATUS_IGNORE);
    }

    // Gather results.
    MPI_Gatherv(&(local_mat3[0][0]), num_rows*num_cols/size, MPI_INT,
            global_mat3, send_counts, displacements, subarray_type,
            ROOT, MPI_COMM_WORLD);

    // Print results.
    if(rank == ROOT) {
        printf("\t-----RESULTS-----\n");
        print_mat(mat3, num_rows, num_cols);
        printf("\t-----------------\n");
    }
    
    // Free the resources.
    MPI_Type_free(&subarray_type);
    free_mat(&local_mat1);
    free_mat(&local_mat2);
    free_mat(&local_mat3);
    free_mat(&tmp_results);
    if(rank == ROOT) {
        free_mat(&mat1);
        free_mat(&mat2);
        free_mat(&mat3);
    }

    MPI_Finalize(); // Close MPI.
    return EXIT_SUCCESS;
}
