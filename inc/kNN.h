#ifndef KNN_H
#define KNN_H

#include "vptree.h"

// Definition of the kNN result struct
typedef struct knnresult
{
    int *nidx;     //!< Indices (0-based) of nearest neighbors [m-by-k]
    float *ndist;  //!< Distance of nearest neighbors          [m-by-k]
    int m;         //!< Number of query points                 [scalar]
    int k;         //!< Number of nearest neighbors            [scalar]
} knnresult;

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!
    \param root   The Vantage Point Tree 
    \param query  Query data points      [n-by-d]
    \param n      Number of data points  [scalar]
    \param d      Number of dimensions   [scalar]
    \param k      Number of neighbors    [scalar]

*/
knnresult vptree_kNN(vptree *root, float *query, int n, int d, int k);

#endif