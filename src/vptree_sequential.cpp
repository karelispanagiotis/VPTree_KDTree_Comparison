#include "vptree.h"
#include <stdlib.h>
#include <math.h>
#include <bits/stdc++.h>

//Globally defined variables for easy data access by threads
static int *idArr;
static float *distArr;
static vptree *treeArr;
static float *Y; //data array
static int N, D;  //data dimensions

////////////////////////////////////////////////////////////////////////

float sqr(float x) {return x*x;}
void distCalc(float *vp, int start, int end)
{
    float(*dataArr)[D] = (float(*)[D])Y;
    for (int i=start; i<=end; i++)
        distArr[i] = sqr(vp[0] - dataArr[idArr[i]][0]);
    for (int i=start; i<=end; i++)
        for (int j=1; j<D; j++)
            distArr[i] += sqr(vp[j] - dataArr[idArr[i]][j]);
};

////////////////////////////////////////////////////////////////////////

void quickSelect_vpt(int kpos, int start, int end)
{
    int store;
    float pivot;
    while(start<=end)
    {
        store = start;
        pivot = distArr[end];

        for (int i=start; i<=end; i++)
        if (distArr[i] <= pivot)
        {
            std::swap(distArr[i], distArr[store]);
            std::swap(idArr  [i], idArr  [store]);
            store++;
        }        
        store--;

        if(store ==kpos) return;
        else if(store < kpos) start = store + 1;
        else end = store - 1; 
    }
}

////////////////////////////////////////////////////////////////////////

void buildNode_vpt(int start, int end, int nodeNumber, int offset)
{
    float(*dataArr)[D] = (float(*)[D])Y;
    //consider X[ idArr[end] ] as vantage point
    treeArr[nodeNumber].idx = idArr[end] + offset;
    treeArr[nodeNumber].vp  = dataArr[ idArr[end] ]; 
    
    if (start==end)
    {
        treeArr[nodeNumber].inner = treeArr[nodeNumber].outer = NULL;
        treeArr[nodeNumber].md = 0.0;
        return;
    }
    end--; //end is the vantage point, we're not dealing with it again
    distCalc(treeArr[nodeNumber].vp,start,end);
    
    quickSelect_vpt( (start+end)/2, start, end );
    // now idArr[start .. (start+end)/2] contains the indexes
    // for the points which belong inside the radius (inner)

    treeArr[nodeNumber].md  = sqrt(distArr[ (start+end)/2 ]);
    treeArr[nodeNumber].inner = &treeArr[2*nodeNumber + 1];
    treeArr[nodeNumber].outer = &treeArr[2*nodeNumber + 2];
    buildNode_vpt( start, (start+end)/2, 2*nodeNumber + 1, offset);
    if (end>start)
        buildNode_vpt( (start+end)/2 +1, end, 2*nodeNumber + 2, offset);
    else treeArr[nodeNumber].outer = NULL;
};

/////////////////////////////////////////////////////////////////////////////

vptree *buildvp(float *X, int n, int d, int offset)
{
    size_t arraySize = 1<<(32 - __builtin_clz(n-1));// Gets the next largest power of 2
    treeArr = (vptree *)malloc( arraySize*sizeof(vptree) );
    idArr           =(int *)malloc( n*sizeof(int) );
    distArr         = (float *)malloc( n*sizeof(float) );
    Y=X, N=n, D=d;
    for (int i=0; i<N; i++) idArr[i] = i;

    buildNode_vpt(0, n-1, 0, offset);
    
    free(idArr);
    free(distArr);
    return &treeArr[0];
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
float getMD(vptree* T)    {    return T->md;}  
float* getVP(vptree* T)   {    return T->vp;}
int getIDX(vptree* T)      {    return T->idx;}
