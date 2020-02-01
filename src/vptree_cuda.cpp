#include "vptree.h"
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

#define BLK_SZ 512

//Globally defined variables for easy data access by functions
__managed__ int *dev_idArr;
__managed__ float *dev_distArr;
__managed__ float *dev_X;
__managed__ vptree *dev_treeArr;
__managed__ int dev_n, dev_d;

__device__ __forceinline__
float sqr(float x) {return x*x;}

__global__ 
void distCalc(float *vp, int start, int end)
{
    int idx = start + blockIdx.x*BLK_SZ + threadIdx.x;  //calculate idx
    if(idx <= end)
    {
        float distance = 0;
        for(int j=0; j<dev_d; j++)
            distance += sqr(vp[j] - dev_X[dev_idArr[idx]*dev_d + j]); 
        
        dev_distArr[idx] = distance;
    }
}

__device__ __forceinline__
void swapFloat(float* a, float* b)
{
    float temp = *a;
    *a = *b;
    *b = temp;
}

__device__ __forceinline__
void swapInt(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__
void quickSelect(int kpos, int start, int end)
{
    int store = start;
    float pivot = dev_distArr[end];
    for (int i=start; i<=end; i++)
        if (dev_distArr[i] <= pivot)
        {
            swapFloat(dev_distArr+i, dev_distArr+store);
            swapInt  (dev_idArr+i  , dev_idArr+store);
            store++;
        }        
    store--;
    if (store == kpos) return;
    else if (store < kpos) quickSelect(kpos, store+1, end);
    else quickSelect(kpos, start, store-1);
}

__global__
void recursiveBuildTree(int start, int end, int nodeNumber)
{
    //consider X[ idArr[end] ] as vantage point
    dev_treeArr[nodeNumber].idx = dev_idArr[end];
    dev_treeArr[nodeNumber].vp = &dev_X[ dev_idArr[end]*dev_d ];

    if(start == end)
    {
        dev_treeArr[nodeNumber].inner = dev_treeArr[nodeNumber].outer = NULL;
        dev_treeArr[nodeNumber].md = 0.0;
        return;
    }
    end--; //end is the vantage point, we're not dealing with it again

    distCalc<<<(end-start+BLK_SZ)/BLK_SZ, BLK_SZ>>>(dev_treeArr[nodeNumber].vp, start, end);
    cudaDeviceSynchronize();

    quickSelect((start+end)/2, start, end);

    dev_treeArr[nodeNumber].md = sqrtf(dev_distArr[ (start+end)/2 ]);
    dev_treeArr[nodeNumber].inner = &dev_treeArr[2*nodeNumber + 1];
    dev_treeArr[nodeNumber].outer = &dev_treeArr[2*nodeNumber + 2];

    recursiveBuildTree<<<1,1>>>(start, (start+end)/2, 2*nodeNumber + 1);
    if(end>start)
        recursiveBuildTree<<<1,1>>>((start+end)/2 + 1, end, 2*nodeNumber + 2);
    else dev_treeArr[nodeNumber].outer = NULL;
    cudaDeviceSynchronize();

}

__global__ void idx_init()
{
    int idx = blockIdx.x*BLK_SZ + threadIdx.x;
    if(idx<dev_n);
        dev_idArr[idx] = idx; 
}

vptree *buildvp(float *X, int n, int d)
{
    size_t treeSize = 1<<(32 - __builtin_clz(n-1)); // next greater power of 2 than n

    // Allocate Memory on Device
    cudaMalloc(&dev_X, n*d*sizeof(float));
    cudaMalloc(&dev_idArr, n*sizeof(int));
    cudaMalloc(&dev_distArr, n*sizeof(float));
    cudaMalloc(&dev_treeArr, treeSize*sizeof(vptree));

    // Initialise Device Variables
    cudaMemcpy(dev_X, X, n*d*sizeof(int), cudaMemcpyHostToDevice); //copy Data
    cudaMemset(&dev_n, n, sizeof(int)); //Set n
    cudaMemset(&dev_d, d, sizeof(int)); //Set d
    idx_init<<<(n+BLK_SZ-1)/BLK_SZ, BLK_SZ>>>(); //set idx [0...n-1]
    cudaDeviceSynchronize();
    
    // Recursively build tree in GPU
    recursiveBuildTree<<<1,1>>>(0, n-1, 0); //This kernel only needs one thread

    // Copy the result back to host
    vptree *treeArr = (vptree *)malloc(treeSize*sizeof(vptree));
    cudaMemcpy(treeArr, dev_treeArr, treeSize*sizeof(vptree), cudaMemcpyDeviceToHost);

    // Clean-up
    cudaFree(dev_X);
    cudaFree(dev_idArr);
    cudaFree(dev_distArr);
    cudaFree(dev_treeArr);
    
    // Return
    return &treeArr[0];
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
float getMD(vptree* T)    {    return T->md;}  
float* getVP(vptree* T)   {    return T->vp;}
int getIDX(vptree* T)      {    return T->idx;}