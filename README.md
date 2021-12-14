General compile: `gcc -O2 ./≪filename≫ -fopenmp -g3`

Specify number of threads: `OMP_NUM_THREADS=≪threadnumber≫ ./a.out`

Enable prefetch: `icc ./≪filename≫ -qopenmp -O2 -qopt-prefetch`

Enable aggressive SIMD: `gcc ./≪filename≫ -fopenmp -ftree-vectorize -ftree-slp-vectorize -ffast-math -funsafe-loop-optimizations -ftree-loop-if-convert-stores -march=native -mtune=native -Ofast`

Enable thread affinity: `export GOMP_CPU_AFFINITY=0-3`
			`Export OMP_PROC_BIND=true`
