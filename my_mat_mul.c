#include <stdio.h>      /* printf                         */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdlib.h>     /* malloc, calloc, free, atoi     */
#include <string.h>     /* memset                         */
#include <stdint.h>     /* uint32_t, uint64_t             */
#include <omp.h>        /* OpenMP                         */
#include <immintrin.h> /* avx_family                      */
#include <avx2intrin.h> /* avx2                           */

#pragma GCC optimize ("O3")

#define rdpmc(ecx, eax, edx)    \
    asm volatile (              \
        "rdpmc"                 \
        : "=a"(eax),            \
          "=d"(edx)             \
        : "c"(ecx))

/*
 *  usage - how to run the program
 *      @return: -1
 */
int32_t
usage(void)
{
    printf("\t./mat_mul <N>\n");
    return -1;
}

/*
 *  print_matrix - if you need convincing that it works just fine
 *      @N: square matrix size
 *      @m: pointer to matrix
 */
void
print_matrix(uint32_t N, long *m)
{
    for (uint32_t i=0; i<N; ++i) {
        for (uint32_t j=0; j<N; ++j)
            printf("%3ld ", m[i*N + j]);
        printf("\n");
    }
}

/*
 *  main - program entry point
 *      @argc: number of arguments & program name
 *      @argv: arguments
 */
int32_t
main(int32_t argc, char *argv[])
{
    if (argc != 2)
        return usage();

    /* allocate space for matrices */
    clock_t t;
    uint32_t N   = atoi(argv[1]);
    int64_t  *m1 = malloc(N * N * sizeof(int64_t));
    int64_t  *m2 = malloc(N * N * sizeof(int64_t));
    int64_t  *r  = malloc(N * N * sizeof(int64_t));

    /* initialize matrices */
    for (uint32_t i=0; i<N*N; ++i) {
        m1[i] = i;
        m2[i] = i;
    }

    /* result matrix clear; clock init */
    memset(r, 0, N * N * sizeof(int64_t));
    t = clock();

    /* TODO: count L2 cache misses for the next block using RDPMC */

    /* my fast multiplication */
    //#pragma omp parallel for schedule(static) num_threads(8)
    //unroll 8
    //#pragma omp parallel for schedule(static) num_threads(8)
    int64_t *m2_block = malloc(N * N * sizeof(int64_t));
    uint32_t cnt = 0;
    for (uint32_t k = 0; k < N; k+=8)
    {
        for (uint32_t j = 0; j < N; j+=8)
        {
            for (uint32_t i = 0; i < 8; ++i)
            {
                __m512i temp = _mm512_set1_epi64(m1[(i+k)*N + j]);
                _mm512_storeu_epi64(m2_block + cnt, temp);
                cnt+=8;
            }
        }
    }
    for (uint32_t i=0; i<N; ++i)
    {
        __m512i temp[N/8];
        for (uint32_t j=0; j<N; j+=8)
        {
            temp[j/8] = _mm512_setzero_epi32();
        }
        for (uint32_t k=0; k<N; k+=8)
        {
            __m512i a1 = _mm512_set1_epi64(m1[i*N + k]);
            __m512i a2 = _mm512_set1_epi64(m1[i*N + k+1]);
            __m512i a3 = _mm512_set1_epi64(m1[i*N + k+2]);
            __m512i a4 = _mm512_set1_epi64(m1[i*N + k+3]);
            __m512i a5 = _mm512_set1_epi64(m1[i*N + k+4]);
            __m512i a6 = _mm512_set1_epi64(m1[i*N + k+5]);
            __m512i a7 = _mm512_set1_epi64(m1[i*N + k+6]);
            __m512i a8 = _mm512_set1_epi64(m1[i*N + k+7]);
            for (uint32_t j = 0; j < N; j+=8)
            {
		        __m512i b1 = _mm512_load_epi64(m2_block + k/8*N + j*8);
                __m512i b2 = _mm512_load_epi64(m2_block + k/8*N + j*8 + 8);
                __m512i b3 = _mm512_load_epi64(m2_block + k/8*N + j*8 + 16);
                __m512i b4 = _mm512_load_epi64(m2_block + k/8*N + j*8 + 24);
                __m512i b5 = _mm512_load_epi64(m2_block + k/8*N + j*8 + 32);
                __m512i b6 = _mm512_load_epi64(m2_block + k/8*N + j*8 + 40);
                __m512i b7 = _mm512_load_epi64(m2_block + k/8*N + j*8 + 48);
                __m512i b8 = _mm512_load_epi64(m2_block + k/8*N + j*8 + 56);
                b1 = _mm512_mullo_epi64(a1, b1);
                b2 = _mm512_mullo_epi64(a2, b2);
                b3 = _mm512_mullo_epi64(a3, b3);
                b4 = _mm512_mullo_epi64(a4, b4);
                b5 = _mm512_mullo_epi64(a5, b5);
                b6 = _mm512_mullo_epi64(a6, b6);
                b7 = _mm512_mullo_epi64(a7, b7);
                b8 = _mm512_mullo_epi64(a8, b8);
                b1 = _mm512_add_epi64(b1, b2);
                b3 = _mm512_add_epi64(b3, b4);
                b5 = _mm512_add_epi64(b5, b6);
                b7 = _mm512_add_epi64(b7, b8);
                b1 = _mm512_add_epi64(b1, b3);
                b5 = _mm512_add_epi64(b5, b7);
                b1 = _mm512_add_epi64(b1, b5);
                temp[j/8] = _mm512_add_epi64(temp[j/8], b1);
            }
        }
        for (uint32_t j=0; j<N; j+=8)
        {
            _mm512_storeu_epi64(r + i*N + j, temp[j/8]);
        }
    }

    /* clock delta */
    t = clock() - t;

    printf("My Multiplication finished in %6.2f s\n",
           ((float)t)/CLOCKS_PER_SEC);

    printf("%6.2f GIOPS\n",
           (2.0 * N * N * N)/((float)t)*CLOCKS_PER_SEC/1e9);

    return 0;
}
