#include "vptree.h"
#include <stdlib.h>
#include <math.h>

//Globally defined variables for easy data access by threads
int *idArr;
float *distArr;
vptree *treeArr;
float *Y; //data array
int N, D;  //data dimensions

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

void swapFloat(float* a, float* b)
{
    float temp = *a;
    *a = *b;
    *b = temp;
}
void swapInt(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}
void quickSelect(int kpos, int start, int end)
{
    int store=start;
    float pivot=distArr[end];
    for (int i=start; i<=end; i++)
        if (distArr[i] <= pivot)
        {
            swapFloat(distArr+i, distArr+store);
            swapInt   (idArr+i,   idArr+store);
            store++;
        }        
    store--;
    if (store == kpos) return;
    else if (store < kpos) quickSelect(kpos, store+1, end);
    else quickSelect(kpos, start, store-1);
}

////////////////////////////////////////////////////////////////////////

void recursiveBuildTree(int start, int end, int nodeNumber)
{
    float(*dataArr)[D] = (float(*)[D])Y;
    //consider X[ idArr[end] ] as vantage point
    treeArr[nodeNumber].idx = idArr[end];
    treeArr[nodeNumber].vp  = dataArr[ idArr[end] ]; 
    
    if (start==end)
    {
        treeArr[nodeNumber].inner = treeArr[nodeNumber].outer = NULL;
        treeArr[nodeNumber].md = 0.0;
        return;
    }
    end--; //end is the vantage point, we're not dealing with it again
    distCalc(treeArr[nodeNumber].vp,start,end);
    
    quickSelect( (start+end)/2, start, end );
    // now idArr[start .. (start+end)/2] contains the indexes
    // for the points which belong inside the radius (inner)

    treeArr[nodeNumber].md  = sqrt(distArr[ (start+end)/2 ]);
    treeArr[nodeNumber].inner = &treeArr[2*nodeNumber + 1];
    treeArr[nodeNumber].outer = &treeArr[2*nodeNumber + 2];
    recursiveBuildTree( start, (start+end)/2, 2*nodeNumber + 1);
    if (end>start)
        recursiveBuildTree( (start+end)/2 +1, end, 2*nodeNumber + 2);
    else treeArr[nodeNumber].outer = NULL;
};

/////////////////////////////////////////////////////////////////////////////

vptree *buildvp(float *X, int n, int d)
{
    size_t arraySize = 1<<(32 - __builtin_clz(n-1));// Gets the next largest power of 2
    treeArr = (vptree *)malloc( arraySize*sizeof(vptree) );
    idArr           =(int *)malloc( n*sizeof(int) );
    distArr         = (float *)malloc( n*sizeof(float) );
    Y=X, N=n, D=d;
    for (int i=0; i<N; i++) idArr[i] = i;

    recursiveBuildTree(0, n-1, 0);
    
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
