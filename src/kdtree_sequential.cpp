#include "kdtree.h"
#include "utilities.h"
#include <bits/stdc++.h>

void buildNode_kdt(kdtree *node, float *X, int n, int d, float *auxArray, int *idArr, int start, int end, int depth)
{
    float(*dataArr)[d] = (float(*)[d])X;

    node->axis = depth%d;

    if(start==end)
    {
        node->idx = idArr[start];
        node->p   = dataArr[ idArr[start] ];
        node->mc  = 0.0;
        node->left = node->right = NULL;
        return;
    }

    for(int i=start; i<=end; i++)
        auxArray[i] = dataArr[idArr[i]][node->axis];

    quickSelect((start+end)/2, auxArray, idArr, start, end);

    node->idx = idArr[(start+end)/2];
    node->p   = dataArr[ idArr[(start+end)/2] ];
    node->mc  = auxArray[(start+end)/2];
    node->right = (kdtree *)malloc(sizeof(kdtree));

    // Recursion
    buildNode_kdt(node->right, X, n, d, auxArray, idArr, (start+end)/2+1, end, depth+1);
    if(start<(start+end)/2)
    {
        node->left  = (kdtree *)malloc(sizeof(kdtree));
        buildNode_kdt(node->left, X, n, d, auxArray, idArr, start, (start+end)/2 - 1, depth+1);
    }
    else node->left = NULL;
}

kdtree *buildkd(float *X, int n, int d)
{
    kdtree *root = (kdtree *)malloc(sizeof(kdtree));
    float *aux_array = (float *)malloc(n*sizeof(float));
    int *idArr = (int *)malloc(n*sizeof(int));
    for(int i=0; i<n; i++) idArr[i] = i;

    buildNode_kdt(root, X, n, d, aux_array, idArr, 0, n-1, 0);

    free(idArr);
    free(aux_array);
    return root;
}

float* getPoint(kdtree *node)   {return node->p;}
float getMC(kdtree *node)       {return node->mc;}
int getIdx(kdtree *node)        {return node->idx;}
int getAxis(kdtree *node)       {return node->axis;}
kdtree *getLeft(kdtree *node)   {return node->left;}
kdtree *getRight(kdtree *node)  {return node->right;}