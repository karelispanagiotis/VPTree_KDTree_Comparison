#include "vptree.h"
#include <stdlib.h>
#include <math.h>

//Globally defined variables for easy data access by threads
int *idArr;
double *distArr;
double *Y; //data array
int N, D;  //data dimensions

////////////////////////////////////////////////////////////////////////

inline double sqr(double x) {return x*x;}
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

inline void swapDouble(double* a, double* b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}
inline void swapInt(int* a, int* b)
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

void recursiveBuildTree(vptree* node, int start, int end)
{
    double(*dataArr)[D] = (double(*)[D])Y;
    //consider X[ idArr[end] ] as vantage point
    node->idx = idArr[end];
    node->vp  = dataArr[ node->idx ]; 
    
    if (start==end)
    {
        node->inner = node->outer = NULL;
        node->md = 0.0;
        return;
    }
    end--; //end is the vantage point, we're not dealing with it again
    distCalc(node->vp,start,end);
    
    quickSelect( (start+end)/2, distArr, idArr, start, end );
    // now idArr[start .. (start+end)/2] contains the indexes
    // for the points which belong inside the radius (inner)

    node->md  = sqrt(distArr[ (start+end)/2 ]);
    node->inner = malloc( sizeof(vptree) );
    node->outer = malloc( sizeof(vptree) );
    recursiveBuildTree(node->inner, start, (start+end)/2);
    if (end>start)
        recursiveBuildTree(node->outer, (start+end)/2 +1, end);
    else node->outer = NULL;
};

/////////////////////////////////////////////////////////////////////////////

vptree *buildvp(double *X, int n, int d)
{
    vptree *root = malloc( sizeof(vptree) );
    idArr        = malloc( n*sizeof(int) );
    distArr      = malloc( n*sizeof(double) );
    Y=X, N=n, D=d;
    for (int i=0; i<N; i++) idArr[i] = i;

    recursiveBuildTree(root, 0, n-1);
    
    free(idArr);
    free(distArr);
    return root;
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
double getMD(vptree* T)    {    return T->md;}  
double* getVP(vptree* T)   {    return T->vp;}
int getIDX(vptree* T)      {    return T->idx;}
