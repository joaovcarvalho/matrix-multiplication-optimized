/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
//#include <xmmintrin.h>
#include <x86intrin.h>
//#include <pmmintrin.h>
/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
//#define DEBUGGING(_x) _x
#define DEBUGGING(_x)

struct complex {
    double real;
    double imag;
};

/* write matrix to stdout */
void write_out(struct complex ** a, int dim1, int dim2)
{
    int i, j;

    for ( i = 0; i < dim1; i++ ) {
        for ( j = 0; j < dim2 - 1; j++ ) {
            printf("%.3f + %.3fi |||", a[i][j].real, a[i][j].imag);
        }
        printf("%.3f + %.3fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
    }
}


/* create new empty matrix */
struct complex ** new_empty_matrix(int dim1, int dim2)
{
    struct complex ** result = (struct complex **) malloc(sizeof(struct complex*) * dim1);
    struct complex * new_matrix = (struct complex *) malloc(sizeof(struct complex) * dim1 * dim2);
    int i;

    for ( i = 0; i < dim1; i++ ) {
        result[i] = &(new_matrix[i*dim2]);
    }

    return result;
}

void free_matrix(struct complex ** matrix) {
        free (matrix[0]); /* free the contents */
        free (matrix); /* free the header */
}

/* take a copy of the matrix and return in a newly allocated matrix */
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
    int i, j;
    struct complex ** result = new_empty_matrix(dim1, dim2);

    for ( i = 0; i < dim1; i++ ) {
        for ( j = 0; j < dim2; j++ ) {
            result[i][j] = source_matrix[i][j];
        }
    }

    return result;
}

/* create a matrix and fill it with random numbers */
struct complex ** gen_random_matrix(int dim1, int dim2)
{
        const int random_range = 512; // constant power of 2
        struct complex ** result;
        int i, j;
        struct timeval seedtime;
        int seed;

        result = new_empty_matrix(dim1, dim2);

        /* use the microsecond part of the current time as a pseudorandom seed */
        gettimeofday(&seedtime, NULL);
        seed = seedtime.tv_usec;
        srandom(seed);

        /* fill the matrix with random numbers */
        for ( i = 0; i < dim1; i++ ) {
            for ( j = 0; j < dim2; j++ ) {
                        /* evenly generate values in the range [0, random_range-1)*/
                result[i][j].real = (double)(random() % random_range);
                result[i][j].imag = (double)(random() % random_range);
                        /* at no loss of precision, negate the values sometimes */
                        /* so the range is now (-(random_range-1), random_range-1)*/
                if (random() & 1) result[i][j].real = -result[i][j].real;
                if (random() & 1) result[i][j].imag = -result[i][j].imag;
            }
        }

        return result;
    }

/* check the sum of absolute differences is within reasonable epsilon */
/* returns number of differing values */
    void check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
    {
        int i, j;
        double sum_abs_diff = 0.0;
        const double EPSILON = 0.0625;

        for ( i = 0; i < dim1; i++ ) {
            for ( j = 0; j < dim2; j++ ) {
                double diff;
                diff = abs(control[i][j].real - result[i][j].real);
                sum_abs_diff = sum_abs_diff + diff;

                diff = abs(control[i][j].imag - result[i][j].imag);
                sum_abs_diff = sum_abs_diff + diff;
            }
        }

        if ( sum_abs_diff > EPSILON ) {
            fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
                sum_abs_diff, EPSILON);
        }
    }

/* multiply matrix A times matrix B and put result in matrix C */
    void matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_cols)
    {
        int i, j, k;

        for ( i = 0; i < a_dim1; i++ ) {
            for( j = 0; j < b_cols; j++ ) {
                struct complex sum;
                sum.real = 0.0;
                sum.imag = 0.0;
                for ( k = 0; k < a_dim2; k++ ) {
                                // the following code does: sum += A[i][k] * B[k][j];
                    struct complex product;
                    product.real = A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
                    product.imag = A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
                    sum.real += product.real;
                    sum.imag += product.imag;
                }
                C[i][j] = sum;
            }
        }
    }

    int min(int a, int b){
      if( a > b)
        return b;
      if( b > a)
        return a;
      return a;
    }

    /* the fast version of matmul written by the team */
    void team_matmul(struct complex ** A, struct complex ** b, struct complex ** C, int a_rows, int a_cols, int b_cols)
    {
        int i, j;

        // Calculating the transpose of the Matrix B
        struct complex ** B = new_empty_matrix(b_cols, a_cols);
        for (i = 0; i < a_cols; ++i)
        {
            for (j = 0; j < b_cols; ++j)
            {
                B[j][i] = b[i][j];
            }
        }

        int block_size = 16;
        #pragma omp parallel for
        for ( i = 0; i < a_rows; i += block_size ) {
                struct complex a1, a2, a3, a4;
                struct complex b1, b2, b3, b4;

                float tmpReal[4] = {0,0,0,0};
                float tmpImg[4] = {0,0,0,0};

                int j;

                int stopii = min(i + block_size, a_rows);
                for( j = 0; j < b_cols; j += block_size ) {

                    int stopjj = min(j+block_size, b_cols);

                    for (int ii = i; ii < stopii; ++ii)
                    {
                        
                        for (int jj = j; jj < stopjj; jj++)
                        {
                            struct complex sum;
                            sum.real = 0.0;
                            sum.imag = 0.0;
                            
                            __m128 sum4Real = _mm_set1_ps(0.0);
                            __m128 sum4Img = _mm_set1_ps(0.0);

                            struct complex* ka = A[ii];
                            struct complex* kb = B[jj];

                            while(ka < A[ii]+a_cols){

                              a1 = *(ka);
                              a2 = *(ka + 1);
                              a3 = *(ka + 2);
                              a4 = *(ka + 3);

                              b1 = *(kb);
                              b2 = *(kb + 1);
                              b3 = *(kb + 2);
                              b4 = *(kb + 3);

                              __m128 aReal  = _mm_set_ps(a1.real, a2.real, a3.real, a4.real);
                              __m128 bReal  = _mm_set_ps(b1.real, b2.real, b3.real, b4.real);

                              __m128 aImg   = _mm_set_ps(a1.imag, a2.imag, a3.imag, a4.imag);
                              __m128 bImg   = _mm_set_ps(b1.imag, b2.imag, b3.imag, b4.imag);

                              __m128 productReal = _mm_sub_ps( _mm_mul_ps(aReal, bReal), _mm_mul_ps(aImg, bImg) );
                              __m128 productImg = _mm_add_ps( _mm_mul_ps(aReal, bImg), _mm_mul_ps(aImg, bReal) );

                              sum4Real = _mm_add_ps(sum4Real, productReal);
                              sum4Img = _mm_add_ps(sum4Img, productImg);

                              ka += 4;
                              kb += 4;
                            }

                            _mm_storeu_ps(&tmpReal[0], sum4Real);
                            _mm_storeu_ps(&tmpImg[0], sum4Img);

                            for (int p = 0; p < 4; ++p)
                            {
                                sum.real += tmpReal[p];
                                sum.imag += tmpImg[p];
                            }

                            C[ii][jj] = sum;
                        }

                    }
                }
        }
    }

    long long time_diff(struct timeval * start, struct timeval * end) {
        return (end->tv_sec - start->tv_sec) * 1000000L + (end->tv_usec - start->tv_usec);
    }

    int main(int argc, char ** argv)
    {
        struct complex ** A, ** B, ** C;
        struct complex ** control_matrix;
        long long control_time, mul_time;
        double speedup;
        int a_dim1, a_dim2, b_dim1, b_cols, errs;
        struct timeval pre_time, start_time, stop_time;

        if ( argc != 5 ) {
            fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
            exit(1);
        }
        else {
            a_dim1 = atoi(argv[1]);
            a_dim2 = atoi(argv[2]);
            b_dim1 = atoi(argv[3]);
            b_cols = atoi(argv[4]);
        }

        /* check the matrix sizes are compatible */
        if ( a_dim2 != b_dim1 ) {
            fprintf(stderr,
                "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
                a_dim2, b_dim1);
            exit(1);
        }

        /* allocate the matrices */
        A = gen_random_matrix(a_dim1, a_dim2);
        B = gen_random_matrix(b_dim1, b_cols);
        C = new_empty_matrix(a_dim1, b_cols);
        control_matrix = new_empty_matrix(a_dim1, b_cols);

        DEBUGGING( {
            printf("matrix A:\n");
            write_out(A, a_dim1, a_dim2);
            printf("\nmatrix B:\n");
            write_out(B, b_dim1, b_cols);
            printf("\n");
        } )

        /* record control start time */
        gettimeofday(&pre_time, NULL);

        /* use a simple matmul routine to produce control result */
        matmul(A, B, control_matrix, a_dim1, a_dim2, b_cols);

        /* record starting time */
        gettimeofday(&start_time, NULL);

        /* perform matrix multiplication */
        team_matmul(A, B, C, a_dim1, a_dim2, b_cols);

        /* record finishing time */
        gettimeofday(&stop_time, NULL);

        /* compute elapsed times and speedup factor */
        control_time = time_diff(&pre_time, &start_time);
        mul_time = time_diff(&start_time, &stop_time);
        speedup = (double) control_time / mul_time;

        printf("Matmul time: %lld microseconds\n", mul_time);
        printf("control time : %lld microseconds\n", control_time);
        if (mul_time > 0 && control_time > 0) {
            printf("speedup: %.2fx\n", speedup);
        }

        /* now check that the team's matmul routine gives the same answer
           as the known working version */
        check_result(C, control_matrix, a_dim1, b_cols);

        /* free all matrices */
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
        free_matrix(control_matrix);

        return 0;
    }

