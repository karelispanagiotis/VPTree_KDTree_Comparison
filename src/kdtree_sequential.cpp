#include "kdtree.h"
#include <cstdlib>
#include <bits/stdc++.h>

static int *idArr;
static kdtree *treeArr;
static float *Y, *aux_array;
static int N, D;

void quickSelect_kdt(int kpos, int start, int end)
{
    int store;
    float pivot;
    while(start<=end)
    {
        store = start;
        pivot = aux_array[end];

        for (int i=start; i<=end; i++)
        if (aux_array[i] <= pivot)
        {
            std::swap(aux_array[i], aux_array[store]);
            std::swap(idArr    [i], idArr    [store]);
            store++;
        }        
        store--;

        if(store ==kpos) return;
        else if(store < kpos) start = store + 1;
        else end = store - 1; 
    }
}


void buildNode_kdt(int start, int end, int ndNum)
{
    float(*dataArr)[D] = (float(*)[D])Y;

    int depth = (32 - __builtin_clz(ndNum+1)) - 1;
    treeArr[ndNum].axis = depth%D;

    if(start==end)
    {
        treeArr[ndNum].idx = idArr[start];
        treeArr[ndNum].p   = dataArr[ idArr[start] ];
        treeArr[ndNum].mc  = 0.0;
        treeArr[ndNum].left = treeArr[ndNum].right = NULL;
        return;
    }

    for(int i=start; i<=end; i++)
        aux_array[i] = dataArr[idArr[i]][treeArr[ndNum].axis];

    int middle = (start+end)/2;
    quickSelect_kdt(middle, start, end);

    treeArr[ndNum].idx = idArr[middle];
    treeArr[ndNum].p   = dataArr[ idArr[middle] ];
    treeArr[ndNum].mc  = aux_array[middle];
    treeArr[ndNum].left  = &treeArr[2*ndNum + 1];
    treeArr[ndNum].right = &treeArr[2*ndNum + 2];

    // Recursion
    buildNode_kdt(middle+1, end, 2*ndNum+2);
    if(start<middle) buildNode_kdt(start, middle-1, 2*ndNum+1);
    else treeArr[ndNum].left = NULL;
}

kdtree *buildkd(float *X, int n, int d)
{
    size_t arraySize = 1<<(32 - __builtin_clz(n-1));
    treeArr = (kdtree *)malloc(arraySize*sizeof(kdtree));
    aux_array = (float *)malloc(n*sizeof(float));
    idArr = (int *)malloc(n*sizeof(int));

    Y=X; N=n; D=d;
    for(int i=0; i<n; i++) idArr[i] = i;

    buildNode_kdt(0, n-1, 0);

    free(idArr);
    free(aux_array);
    return &treeArr[0];
}

float* getPoint(kdtree *node)   {return node->p;}
float getMC(kdtree *node)       {return node->mc;}
int getIdx(kdtree *node)        {return node->idx;}
int getAxis(kdtree *node)       {return node->axis;}
kdtree *getLeft(kdtree *node)   {return node->left;}
kdtree *getRight(kdtree *node)  {return node->right;}