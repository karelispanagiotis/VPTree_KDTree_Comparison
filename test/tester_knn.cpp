/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

#include "tester_helper.h"
#include "vptree.h"

int main(int argc, char *argv[])
{

  int n=atoi(argv[1]); // corpus
  int m=atoi(argv[2]); // query 
  int d=atoi(argv[3]); // dimensions
  int k=atoi(argv[4]); // # neighbors

  float  * corpus = (float * ) malloc( n*d * sizeof(float) );
  float  * query  = (float * ) malloc( m*d * sizeof(float) );

  for (int i=0;i<n*d;i++)
    corpus[i] = rand()%100 + ( (float) (rand()) ) / (float) RAND_MAX;

  for (int i=0;i<m*d;i++)
    query[i]  = rand()%100 + ( (float) (rand()) ) / (float) RAND_MAX;
  
  struct timeval start, end;

  printf("Using Vantage-Point Tree:\n");
  gettimeofday(&start, NULL);
  vptree *rootvp = buildvp(corpus, n, d);
  gettimeofday(&end, NULL);

  knnresult knnres = vptree_kNN(rootvp, query, m, d, k);
  int isValidR = validateResult( knnres, corpus, query, n, m, d, k, ROWMAJOR );

  long time_usec = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
  printf("Construction of tree took %lf\n", (double)time_usec/1000000);
  printf("Tester validation: %s NEIGHBORS\n", STR_CORRECT_WRONG[isValidR]);

  /////////////////////////////////////////////////////////////////////////////////////////

  printf("Using KD Tree:\n");
  gettimeofday(&start, NULL);
  kdtree *rootkd = buildkd(corpus, n, d);
  gettimeofday(&end, NULL);

  knnres = kdtree_kNN(rootkd, query, m, d, k);
  isValidR = validateResult( knnres, corpus, query, n, m, d, k, ROWMAJOR );

  time_usec = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
  printf("Construction of tree took %lf\n", (double)time_usec/1000000);
  printf("Tester validation: %s NEIGHBORS\n", STR_CORRECT_WRONG[isValidR]);

  free( corpus );
  free( query );
  return 0;
}
