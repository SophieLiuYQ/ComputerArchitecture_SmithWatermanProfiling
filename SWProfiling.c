#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cpuid.h>
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#include <omp.h>
#include <time.h>

#define REPEAT 3

//stanley98yu/opt-lcs

// C timer starts
// rdtsc is a counter in x86 that has current # of ticks, this function returns its value
inline uint64_t rdtsc(void) {
  unsigned long a, d;
  asm volatile("rdtsc" : "=a"(a), "=d"(d));
  return a | ((uint64_t)d << 32);
}

inline uint64_t rdtscp(void) {
  unsigned long a, d;
  asm volatile("rdtscp" : "=a"(a), "=d"(d));
  // The lower 32-bits are in EAX, and the upper 32-bits are in EDX
  return a | ((uint64_t)d << 32);
}
// C timer ends


// Smith Waterman Algorithm helper functions start
// return a specific location
static __inline__ int mat_get(const int*mat, unsigned c_len, unsigned i, unsigned j)
{
    return mat[(size_t)i * c_len + j];
}

// set the value of a specific location
static __inline__ void mat_set(int*mat, unsigned c_len, unsigned i, unsigned j, int val)
{
    mat[(size_t)i * c_len + j] = val;
}

// compare values
static __inline__ int max(int a, int b) {
    return a > b ? a : b;
}

// compare matches
int scoring_function(char a, char b)
{
    if (a == b) {
        return 1;      //match - assign 1
    } else {
        return -1;     //mismatch - assign -1
    }
}

#define min(x, y) (((x) < (y)) ? (x) : (y))
// Smith Waterman Algorithm helper functions end


// https://stackoverflow.com/questions/3463426/in-c-how-should-i-read-a-text-file-and-print-all-strings
//Read file starts
char* ReadFile(char *filename)
{
   char *buffer = NULL;
   int string_size, read_size;
   FILE *handler = fopen(filename, "r");

   if (handler)
   {
        // Seek the last byte of the file
        fseek(handler, 0, SEEK_END);
        // Offset from the first to the last byte, or in other words, filesize
        string_size = ftell(handler);
        // go back to the start of the file
        rewind(handler);
        // Allocate a string that can hold it all
        buffer = (char*) malloc(sizeof(char) * (string_size + 1) );
        // Read it all in one operation
        read_size = fread(buffer, sizeof(char), string_size, handler);
        // fread doesn't set it so put a \0 in the last position
        // and buffer is now officially a string
        buffer[string_size] = '\0';

        if (string_size != read_size)
        {
            // Something went wrong, throw away the memory and set
            // the buffer to NULL
            free(buffer);
            buffer = NULL;
        }
        // Always remember to close the file.
        fclose(handler);
    }
    return buffer;
}
//Read file ends


//https://github.com/masyagin1998/bio-alignment/blob/master/src/smith-waterman/smith-waterman.c
// naive Smith Waterman Algorithm calculates the length of the longest match string and return this value to max_val (score_max)
int SW_naive (char* A, char* B){
    int m = strlen(A);            //m = length of A     
    int n = strlen(B);            //n = length of B
    int max_val = INT_MIN;        //C INT_MIN keeps track of the smallest value in the program
    int gap = -1;                 // assign "gap" value to be -1
    int*scores;                   
    scores = (int*) calloc(((size_t)(m + 1)) * (n + 1), sizeof(int));     //allocate memory to keep track of scores
    unsigned i,j;

    mat_set(scores, n+ 1, 0, 0, 0);    //set scores at (0,0) to 0 (row major)
    for (i = 1; i < (m + 1); i++) {    //set scores at (i,0) to 0 (row major)
        mat_set(scores, n+ 1, i, 0, 0);
    }
    for (i = 1; i < (n + 1); i++) {    //set scores at (0,i) to 0 (row major)
        mat_set(scores, n + 1, 0, i, 0);
    }

    for (i = 1; i < m + 1; i++) {
        for (j = 1; j < n + 1; j++) {
            int score_up   = mat_get(scores, n + 1, i - 1, j) + gap;   //get the score values at score_up location
            int score_left = mat_get(scores, n + 1, i, j - 1) + gap;   //get the score values at score_left location
            int score_diag = mat_get(scores, n + 1, i - 1, j - 1) + scoring_function(A[i - 1], B[j - 1]);
            int score_max = max(0, max(score_up, max(score_left, score_diag)));   //find max score
            mat_set(scores, n + 1, i, j, score_max);    //find the max score location 

            if (score_max > max_val) {       //keep track of the max score and assign it to max_val
                // printf("\n%d\n", score_max);
                max_val = score_max;           
            }
        }
    }

    free(scores);

    return max_val;
}


//https://github.com/Kartikay77/PARALLELIZATION-USING-WATERMAN-SMITH-ALGORITHM/blob/main/omp_smithW.c
//https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Jian-Chen-Spring-2019.pdf 
// parallel Smith Waterman Algorithm uses C OpenMP to enable parallel computation of the max matching length
int SW_parallel_critical(char* A, char* B) {
    int m = strlen(A);            
    int n = strlen(B);            
    int max_val = INT_MIN;
    int gap = -1; 
    int*scores;
    scores = (int*) calloc(((size_t)(m + 1)) * (n + 1), sizeof(int));

    unsigned int nDiag = m + n - 1;   //Parallel version performs calculation m+n-1 times (see slides)

  // parallelization starts
  #pragma omp parallel default(none) shared(A, B, scores, max_val, nDiag, m, n, gap)  //Shared variables are shared between threads
  {
      long long i, j, nEle, si, sj, ai, aj;  //implicit private variables
      for (i = 1; i <= nDiag; ++i)
      {
          // Calculate the number of i-diagonal elements
          if (i < m && i < n) {
            nEle = i;
          }
          else if (i < max(m, n)) {
            nEle = min(m, n)-1;
          }
          else {
            nEle = 2 * min(m, n) - i + abs(m - n) - 2;
          }

          if (i < n) {
              si = i;
              sj = 1;
          } else {
              si = n - 1;
              sj = i - n + 2;
          }

          #pragma omp for
          for (j = 1; j <= nEle; ++j)
          {
              ai = si - j + 1;
              aj = sj + j - 1;

              long long int index = m * ai + aj;    
              int score_up   = scores[index - m] + gap;
              int score_left = scores[index - 1] + gap;
              int score_diag = scores[index - m - 1] + scoring_function(A[ai - 1], B[aj - 1]);
              int score_max = max(0, max(score_up, max(score_left, score_diag)));
              scores[index] = score_max;

              //using critical operation to update max score
              #pragma omp critical(max_val)
              {
                if (score_max > max_val) {
                    // printf("\n%d\n", score_max);
                    max_val = score_max;
                }
              }
          }
      }
  }

  free(scores);
  
  return max_val;
}


// parallel atomic SW uses atomic operation to update the max score

int SW_parallel_atomic(char* A, char* B) {
    int m = strlen(A);            
    int n = strlen(B);            
    int max_val = INT_MIN;
    int gap = -1; 
    int*scores;
    scores = (int*) calloc(((size_t)(m + 1)) * (n + 1), sizeof(int));

    unsigned int nDiag = m + n - 1;   //Parallel version performs calculation m+n-1 times (see slides)

  // parallelization starts
  #pragma omp parallel default(none) shared(A, B, scores, max_val, nDiag, m, n, gap)  //Shared variables are shared between threads
  {
      long long i, j, nEle, si, sj, ai, aj;  //implicit private variables
      for (i = 1; i <= nDiag; ++i)
      {
          // Calculate the number of i-diagonal elements
          if (i < m && i < n) {
            nEle = i;
          }
          else if (i < max(m, n)) {
            nEle = min(m, n)-1;
          }
          else {
            nEle = 2 * min(m, n) - i + abs(m - n) - 2;
          }

          if (i < n) {
              si = i;
              sj = 1;
          } else {
              si = n - 1;
              sj = i - n + 2;
          }

          #pragma omp for
          for (j = 1; j <= nEle; ++j)
          {
              ai = si - j + 1;
              aj = sj + j - 1;

              long long int index = m * ai + aj;    
              int score_up   = scores[index - m] + gap;
              int score_left = scores[index - 1] + gap;
              int score_diag = scores[index - m - 1] + scoring_function(A[ai - 1], B[aj - 1]);
              int score_max = max(0, max(score_up, max(score_left, score_diag)));
              scores[index] = score_max;

              //using atomic operation to update max score
              int prev_val = atomic_load(&max_val);
              while((prev_val < score_max) &&
                    !atomic_compare_exchange_weak(&max_val, &prev_val, score_max));
          }
      }
  }

  free(scores);

  return max_val;
}

enum sw_version_t {
  NAIVE, ATOMIC, CRITICAL
};

typedef int (*sw_function_t)(char*, char*);

sw_function_t get_sw_function(enum sw_version_t which) {
  switch (which)
  {
  case NAIVE:
    return &SW_naive;
  case ATOMIC:
    return &SW_parallel_atomic;
  case CRITICAL:
    return &SW_parallel_critical;
  default:
    return NULL;
  }
}

const char* get_sw_pretty(enum sw_version_t which) {
  switch (which)
  {
  case NAIVE:
    return "naive ---";
  case ATOMIC:
    return "atomic --";
  case CRITICAL:
    return "critical ";
  default:
    return NULL;
  }
}

void run_sw(size_t size, enum sw_version_t which) {
  int i;
  long int rep;
  uint64_t offset;
  uint64_t start, end, clock = 0;
  uint32_t eax, ebx, ecx, edx;

  char *A = ReadFile ("ecoli_1.txt");    //arbiturary ecoli genomes up to a hundred thousand long
  char *B = ReadFile ("ecoli_2.txt");    //arbiturary ecoli genomes up to a hundred thousand long

  sw_function_t fn = get_sw_function(which);
  A[size] = '\0';
  B[size] = '\0';

  // collect data REPEAT times to ensure reliability
  for (rep = 0; rep < REPEAT; rep++) {

    __cpuid(1, eax, ebx, ecx, edx);    // prevent cpu OOO to make sure timer is accurate
    start = rdtsc();
    int result = fn(A, B);
    end = rdtscp();
    __cpuid(1, eax, ebx, ecx, edx);    // prevent cpu OOO to make sure timer is accurate
 
    uint64_t time = end - start;
    clock += time;
    printf("result was %d\n", result);
    //fprintf(fptr, "%lu,%lu\n", rep, time);
  }
  // compute average
  double avg = clock / ((double)REPEAT);
  printf("test %lu> took %lu ticks total, avg time to copy was %g ticks\n",
         size, clock, avg);
  // free inputs
  free(A);
  free(B);
}

int main(int ac, char** av) {
  int sizes[4] = {
    100, 1000, 10000, 100000
  };

  enum sw_version_t versions[3] = {
    NAIVE, CRITICAL, ATOMIC
  };

  int ver, sz;
  for (ver = 0; ver < 3; ver++) {
    printf("----- starting version %s---\n", get_sw_pretty(versions[ver]));
    
    for (sz = 0; sz < 4; sz++) {
      run_sw(sizes[sz], versions[ver]);
      printf("-----------------------------------\n");
    }
  }

  return 0;
}