#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#define input_1 "./input/ex1/A.txt"
#define input_2 "./input/ex1/B.txt"

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

void print_mat(int** mat, int num_rows, int num_cols) {
    for(int i = 0; i < num_rows; ++i) {
        for(int j = 0; j < num_cols; ++j)
            printf("%d ", mat[i][j]);
        printf("\n");
    }
}

void mat_mul(int** mat1, int** mat2, int** mat3, int n) { // FOR SIMPLICITY, SQUARED-MATRIX ONLY
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                mat3[i][j] += mat1[i][k]*mat2[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int **mat1, **mat2, **mat3;
    int num_rows, num_cols, counts, n;
    FILE *fp;
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
        return -1;
    }

    while(fscanf(fp, "%d", &n) != EOF) {
        c = fgetc(fp);
        if (c == '\n') num_rows++;
        counts++;
    }

    num_cols = counts/num_rows;
    if(num_rows != num_cols) {
        free_mat(&mat1);
        return -1;
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
        return -1;
    }

    while (fscanf(fp, "%d", &n) != EOF) {
        c = fgetc(fp);
        if (c == '\n') num_rows++;
        counts++;
    }

    num_cols = counts/num_rows;
    if(num_rows != num_cols) {
        free_mat(&mat2);
        return -1;
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

    if(allocate_mat(&mat3, n, n) == -1) {
        printf("[ERROR] Cannot allocate mat3.\n");
        free_mat(&mat1);
        free_mat(&mat2);
        return 0;
    }
    
    // Do the math..
    double start_time = omp_get_wtime();
    mat_mul(mat1, mat2, mat3, n);
    double time_taken = omp_get_wtime() - start_time;
    printf("Time taken in SEQUENTIAL: %.8fs.\n", time_taken);
    
    // print results.
    print_mat(mat3, n, n);

    // Free the resources.
    free_mat(&mat1);
    free_mat(&mat2);
    free_mat(&mat3);

    return 0;
}
