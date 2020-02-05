/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "tester_helper.h"
#include "vptree.h"

int main()
{

  int n=3000;                    // corpus
  int m=100;                    // query 
  int d=7;                      // dimensions
  int k=23;                     // # neighbors

  float  * corpus = (float * ) malloc( n*d * sizeof(float) );
  float  * query  = (float * ) malloc( m*d * sizeof(float) );

  for (int i=0;i<n*d;i++)
    corpus[i] = rand()%100 + ( (float) (rand()) ) / (float) RAND_MAX;

  for (int i=0;i<m*d;i++)
    query[i]  = rand()%100 + ( (float) (rand()) ) / (float) RAND_MAX;

  vptree *root = buildvp(corpus, n, d);

  knnresult knnres = vptree_kNN(root, query, m, d, k);

  int isValidC = validateResult( knnres, corpus, query, n, m, d, k, COLMAJOR );

  int isValidR = validateResult( knnres, corpus, query, n, m, d, k, ROWMAJOR );
  printf("Row Major: %d\n", isValidR);
  printf("Column Major: %d\n", isValidC);
  printf("Tester validation: %s NEIGHBORS\n",
         STR_CORRECT_WRONG[isValidC||isValidR]);

  free( corpus );
  free( query );

  return 0;
  
}
