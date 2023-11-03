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
perf stat ./my_mat_mul <N>
#cache miss
perf stat -e LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,LLC-prefetch-misses ./my_mat_mul <N>
#detail with assemble
perf record ./my_mat_mul <N>
perf report
```
