#include "vptree.h"
#include "utilities.h"
#include <bits/stdc++.h>
#include <math.h>
////////////////////////////////////////////////////////////////////////

inline float sqr(float x) {return x*x;}
void distCalc(float *vp, float *X, int d, int *idArr, float *distArr, int start, int end)
{
    float(*dataArr)[d] = (float(*)[d])X;
    for (int i=start; i<=end; i++)
        distArr[i] = sqr(vp[0] - dataArr[idArr[i]][0]);
    for (int i=start; i<=end; i++)
        for (int j=1; j<d; j++)
            distArr[i] += sqr(vp[j] - dataArr[idArr[i]][j]);
};

////////////////////////////////////////////////////////////////////////

void recursiveBuildTree(vptree* node, float *X, int n, int  d, float *distArr, int *idArr, int start, int end, int idOffset)
{
    float(*dataArr)[d] = (float(*)[d])X;
    //consider X[ idArr[end] ] as vantage point
    node->idx = idArr[end] + idOffset;
    node->vp  = dataArr[ idArr[end] ]; 
    
    if (start==end)
    {
        node->inner = node->outer = NULL;
        node->md = 0.0;
        return;
    }
    end--; //end is the vantage point, we're not dealing with it again
    distCalc(node->vp, X, d, idArr, distArr, start, end);
    
    quickSelect( (start+end)/2, distArr, idArr, start, end );
    // now idArr[start .. (start+end)/2] contains the indexes
    // for the points which belong inside the radius (inner)

    node->md  = sqrt(distArr[ (start+end)/2 ]);
    node->inner = (vptree *)malloc( sizeof(vptree) );
    recursiveBuildTree(node->inner, X, n, d, distArr, idArr, start, (start+end)/2, idOffset);
    if (end>start)
    {
        node->outer = (vptree *)malloc( sizeof(vptree) );
        recursiveBuildTree(node->outer, X, n, d, distArr, idArr, (start+end)/2 +1, end, idOffset);
    }
    else node->outer = NULL;
};

/////////////////////////////////////////////////////////////////////////////

vptree *buildvp(float *X, int n, int d, int idOffset)
{
    vptree *root   = (vptree *)malloc( sizeof(vptree) );
    int *idArr     = (int *)malloc( n*sizeof(int) );
    float *distArr = (float *)malloc( n*sizeof(float) );
    for (int i=0; i<n; i++) idArr[i] = i;

    recursiveBuildTree(root, X, n, d, distArr, idArr, 0, n-1, idOffset);
    
    free(idArr);
    free(distArr);
    return root;
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
float getMD(vptree* T)     {    return T->md;}  
float* getVP(vptree* T)    {    return T->vp;}
int getIDX(vptree* T)      {    return T->idx;}