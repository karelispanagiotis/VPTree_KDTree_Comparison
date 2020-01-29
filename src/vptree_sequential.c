#include "vptree.h"
#include <stdlib.h>
#include <math.h>

//Globally defined variables for easy data access by threads
int *idArr;
double *distArr;
double *Y; //data array
int N, D;  //data dimensions

////////////////////////////////////////////////////////////////////////

double sqr(double x) {return x*x;}
void distCalc(double *vp, int start, int end)
{
    double(*dataArr)[D] = (double(*)[D])Y;
    for (int i=start; i<=end; i++)
        distArr[i] = sqr(vp[0] - dataArr[idArr[i]][0]);
    for (int i=start; i<=end; i++)
        for (int j=1; j<D; j++)
            distArr[i] += sqr(vp[j] - dataArr[idArr[i]][j]);
};

////////////////////////////////////////////////////////////////////////

void swapDouble(double* a, double* b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}
void swapInt(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}
void quickSelect(int kpos, double* distArr, int* idArr, int start, int end)
{
    int store=start;
    double pivot=distArr[end];
    for (int i=start; i<=end; i++)
        if (distArr[i] <= pivot)
        {
            swapDouble(distArr+i, distArr+store);
            swapInt   (idArr+i,   idArr+store);
            store++;
        }        
    store--;
    if (store == kpos) return;
    else if (store < kpos) quickSelect(kpos, distArr, idArr, store+1, end);
    else quickSelect(kpos, distArr, idArr, start, store-1);
}

////////////////////////////////////////////////////////////////////////

void recursiveBuildTree(vptree* array, int start, int end, int nodeNumber)
{
    double(*dataArr)[D] = (double(*)[D])Y;
    //consider X[ idArr[end] ] as vantage point
    array[nodeNumber].idx = idArr[end];
    array[nodeNumber].vp  = dataArr[ idArr[end] ]; 
    
    if (start==end)
    {
        array[nodeNumber].inner = array[nodeNumber].outer = NULL;
        array[nodeNumber].md = 0.0;
        return;
    }
    end--; //end is the vantage point, we're not dealing with it again
    distCalc(array[nodeNumber].vp,start,end);
    
    quickSelect( (start+end)/2, distArr, idArr, start, end );
    // now idArr[start .. (start+end)/2] contains the indexes
    // for the points which belong inside the radius (inner)

    array[nodeNumber].md  = sqrt(distArr[ (start+end)/2 ]);
    array[nodeNumber].inner = &array[2*nodeNumber + 1];
    array[nodeNumber].outer = &array[2*nodeNumber + 2];
    recursiveBuildTree(array, start, (start+end)/2, 2*nodeNumber + 1);
    if (end>start)
        recursiveBuildTree(array, (start+end)/2 +1, end, 2*nodeNumber + 2);
    else array[nodeNumber].outer = NULL;
};

/////////////////////////////////////////////////////////////////////////////

vptree *buildvp(double *X, int n, int d)
{
    size_t arraySize = 1<<(32 - __builtin_clz(n-1));// Gets the next largest power of 2
    vptree *treeArr = malloc( arraySize*sizeof(vptree) );
    idArr        = malloc( n*sizeof(int) );
    distArr      = malloc( n*sizeof(double) );
    Y=X, N=n, D=d;
    for (int i=0; i<N; i++) idArr[i] = i;

    recursiveBuildTree(treeArr, 0, n-1, 0);
    
    free(idArr);
    free(distArr);
    return &treeArr[0];
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
double getMD(vptree* T)    {    return T->md;}  
double* getVP(vptree* T)   {    return T->vp;}
int getIDX(vptree* T)      {    return T->idx;}
