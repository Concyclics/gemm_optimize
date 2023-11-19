# gemm_optimize
optimize matrix multiply with int64
#### Optimization

* using `8*8` block to compute
* reorder `m2` for sequential fetch
* use `AVX512` simd instruction sets to accelerate

#### compile

```shell
 gcc -fopenmp -mavx512dq -mavx512f -O3 my_mat_mul.c -o my_mat_mul
```

#### moniter with perf

```shell
#overview
perf stat -d -d -d ./my_mat_mul <N>
#detail
perf record ./my_mat_mul <N>
perf report
```
#### test log
the test log run at nus soc cluster xcne node with xeon gold 6230.
Run with command:
```shell
perf stat -d -d -d ./mat_mul <N>
```
where N varies [1000, 10000, 1000]

there are 4 log files:
* mat_mul1.csv: no optimized
* mat_mul2.csv: provide mul2 with sequential vector mul-add.
* avx512.csv: only optimized with AVX512 unrolling
* my_mat_mul.csv: optimized with AVX512 and block reordering for cache by kernel block of 8*8.

