#include "vptree.h"
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <thrust/swap.h>

#define BLK_SZ 512

#define cudaErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//Globally defined variables for easy data access by functions
__device__ __managed__ int *dev_idArr;
__device__ __managed__ float *dev_distArr;
__device__ __managed__ float *dev_X;
__device__ __managed__ vptree *dev_treeArr;
__device__ __managed__ int dev_n, dev_d;


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

__device__
void quickSelect(int kpos, int start, int end)
{
    int store = start;
    float pivot = dev_distArr[end];
    for (int i=start; i<=end; i++)
        if (dev_distArr[i] <= pivot)
        {
            thrust::swap(dev_distArr[i], dev_distArr[store]);
            thrust::swap(dev_idArr  [i], dev_idArr  [store]);
            store++;
        }        
    store--;
    if (store == kpos) return;
    else if (store < kpos) quickSelect(kpos, store+1, end);
    else quickSelect(kpos, start, store-1);
}

__global__
void recursiveBuildTree(int start, int end, int nodeNumber, float *hostPtr_d, vptree *hostPtr_t )
{
    //consider X[ idArr[end] ] as vantage point
    float *vp = &dev_X[ dev_idArr[end]*dev_d ]; 
    dev_treeArr[nodeNumber].idx = dev_idArr[end];
    dev_treeArr[nodeNumber].vp = &hostPtr_d[ dev_idArr[end]*dev_d ];

    if(start == end)
    {
        dev_treeArr[nodeNumber].inner = dev_treeArr[nodeNumber].outer = NULL;
        dev_treeArr[nodeNumber].md = 0.0;
        return;
    }
    end--; //end is the vantage point, we're not dealing with it again
    
    distCalc<<<(end-start+BLK_SZ)/BLK_SZ, BLK_SZ>>>(vp, start, end);
    cudaDeviceSynchronize();

    quickSelect((start+end)/2, start, end);
    
    dev_treeArr[nodeNumber].md = sqrtf(dev_distArr[ (start+end)/2 ]);
    dev_treeArr[nodeNumber].inner = &hostPtr_t[2*nodeNumber + 1];
    dev_treeArr[nodeNumber].outer = &hostPtr_t[2*nodeNumber + 2];

    recursiveBuildTree<<<1,1>>>(start, (start+end)/2, 2*nodeNumber + 1, hostPtr_d, hostPtr_t);
    if(end>start)
        recursiveBuildTree<<<1,1>>>((start+end)/2 + 1, end, 2*nodeNumber + 2, hostPtr_d, hostPtr_t);
    else dev_treeArr[nodeNumber].outer = NULL;
}

__global__ void idx_init()
{
    int idx = blockIdx.x*BLK_SZ + threadIdx.x;
    if(idx<dev_n)
        dev_idArr[idx] = idx;
}

vptree *buildvp(float *X, int n, int d, int offset)
{
    size_t treeSize = 1<<(32 - __builtin_clz(n-1)); // next greater power of 2 than n
    vptree *treeArr = (vptree *)malloc(treeSize*sizeof(vptree));

    // Allocate Memory on Device
    cudaErrChk( cudaMalloc(&dev_X, n*d*sizeof(float)) );
    cudaErrChk( cudaMalloc(&dev_idArr, n*sizeof(int)) );
    cudaErrChk( cudaMalloc(&dev_distArr, n*sizeof(float)) );
    cudaErrChk( cudaMalloc(&dev_treeArr, treeSize*sizeof(vptree) ));

    // Initialise Device Variables
    cudaErrChk(cudaMemcpy(dev_X, X, n*d*sizeof(int), cudaMemcpyHostToDevice)); //copy Data
    dev_n = n; //Set n
    dev_d = d; //Set d
    idx_init<<<(n+BLK_SZ-1)/BLK_SZ, BLK_SZ>>>(); //set idx [0...n-1]
    cudaDeviceSynchronize();
    
    // Recursively build tree in GPU
    recursiveBuildTree<<<1,1>>>(0, n-1, 0, X, treeArr); //This kernel only needs one thread
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaErrChk(cudaMemcpy(treeArr, dev_treeArr, treeSize*sizeof(vptree), cudaMemcpyDeviceToHost));

    // Clean-up
    cudaErrChk(cudaFree(dev_X));
    cudaErrChk(cudaFree(dev_idArr));
    cudaErrChk(cudaFree(dev_distArr));
    cudaErrChk(cudaFree(dev_treeArr));

    // Return
    return &treeArr[0];
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
float getMD(vptree* T)     {    return T->md;   }  
float* getVP(vptree* T)    {    return T->vp;   }
int getIDX(vptree* T)      {    return T->idx;  }