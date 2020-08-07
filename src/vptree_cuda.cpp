#include "vptree.h"
#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <bits/stdc++.h>

#define BLK_SZ 256
#define WRK_THRD 4

#define cudaErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ __forceinline__
float sqr(float x) {return x*x;}

__device__
void quickSelect(int kpos, float* distArr, int* idArr, int start, int end)
{
    int store=start;
    double pivot=distArr[end];
    for (int i=start; i<=end; i++)
        if (distArr[i] <= pivot)
        {
            thrust::swap(distArr[i], distArr[store]);
            thrust::swap(idArr[i], idArr[store]);
            store++;
        }        
    store--;
    if (store == kpos) return;
    else if (store < kpos) quickSelect(kpos, distArr, idArr, store+1, end);
    else quickSelect(kpos, distArr, idArr, start, store-1);
}


__global__ 
void distCalc(float *X, int *idArr, float *distArr, int *ends, int n, int d)
{
    for(int tid=blockIdx.x*BLK_SZ + threadIdx.x; tid<n; tid+=BLK_SZ*gridDim.x)
    {
        int end=ends[tid], vpidx=idArr[end], idx=idArr[tid];

        if(tid>=end) continue;

        float distance = 0;
        for(int i=0; i<d; i++)
            distance += sqr(X[vpidx*d + i] - X[idx*d + i]); 
        
        distArr[tid] = sqrt(distance);
    }
}

__global__
void sort(float *distArr, int *idArr, int *starts, int *ends, int n)
{
    int tid = blockIdx.x*BLK_SZ + threadIdx.x;
    if(tid<n)
    {
        int start=starts[tid], end=ends[tid];
        // Shut down threads corresponding to vantage points
        if(tid!=(start+end)/2 || start>=end) return;

        end--;
        quickSelect((start+end)/2, distArr, idArr, start, end);
    }
}

__global__
void update_arrays(int *starts, int *ends, int n)
{
    for(int tid=blockIdx.x*BLK_SZ + threadIdx.x; tid<n; tid+=BLK_SZ*gridDim.x)
    {
        int start=starts[tid], end=ends[tid]-1;

        starts[tid] = (tid<=(start+end)/2) ? start : (start+end)/2 + 1;
        ends  [tid] = (tid<=(start+end)/2) ? (start+end)/2 : end;
    }
}

__global__
void print_arrays(float *dist, int *id, int *start, int *end, int *node, int n)
{
    printf("Dist: ");
    for(int i=0; i<n; i++)
        printf("%lf ", dist[i]);

    printf("\nIDS: ");
    for(int i=0; i<n; i++)
        printf("%d ", id[i]);

    printf("\nstarts: ");
    for(int i=0; i<n; i++)
        printf("%d ", start[i]);

    printf("\nends: ");
    for(int i=0; i<n; i++)
        printf("%d ", end[i]);

    printf("\nnodes: ");
    for(int i=0; i<n; i++)
        printf("%d ", node[i]);
    printf("\n\n");

}

void recursiveBuildTree(vptree *node, float *X, int d, float *distArr, int *idArr, int start, int end)
{
    node->idx = idArr[end];
    node->vp  = &X[d*node->idx];

    if (start==end)
    {
        node->inner = node->outer = NULL;
        node->md = 0.0;
        return;
    }
    end--;

    node->md    = distArr[ (start+end)/2 ];
    node->inner = (vptree *)malloc(sizeof(vptree));
    recursiveBuildTree(node->inner, X, d, distArr, idArr, start, (start+end)/2);
    if(end>start)
    {
        node->outer = (vptree *)malloc(sizeof(vptree));
        recursiveBuildTree(node->outer, X, d, distArr, idArr, (start+end)/2 + 1, end);
    }
    else node->outer = NULL;
}

vptree *buildvp(float *X, int n, int d)
{
    int *idArr     = (int *)malloc(n*sizeof(int));
    float *distArr = (float *)malloc(n*sizeof(float));

    int *dev_idArr, *dev_starts, *dev_ends;
    float *dev_distArr, *dev_X;

    // Allocate Memory on Device
    cudaErrChk( cudaMalloc((void **)&dev_X, n*d*sizeof(float)) );
    cudaErrChk( cudaMalloc((void **)&dev_idArr, 4*n*sizeof(int)) );
    cudaErrChk( cudaMalloc((void **)&dev_starts, n*sizeof(int)) );
    cudaErrChk( cudaMalloc((void **)&dev_ends, n*sizeof(int)) );
    cudaErrChk( cudaMalloc((void **)&dev_distArr, n*sizeof(float)) );


    // Initialize Device Variables
    cudaErrChk(cudaMemcpy(dev_X, X, n*d*sizeof(float), cudaMemcpyHostToDevice)); //copy Data
    thrust::sequence(thrust::device, dev_idArr, dev_idArr+n);
    thrust::fill(thrust::device, dev_starts, dev_starts + n, 0);
    thrust::fill(thrust::device, dev_ends, dev_ends + n, n-1);
    
    // Build tree in GPU (level by level in parallel)
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    for(int i=0; i<floor(log2(n))+1; i++)
    {
        // Parallel Distance Calculation
        distCalc<<<32*numSMs, BLK_SZ>>>(dev_X, dev_idArr, dev_distArr, dev_ends, n, d);

        // Parallel Sorting of intervals [start, end]
        sort<<<(n+BLK_SZ-1)/BLK_SZ, BLK_SZ>>>(dev_distArr, dev_idArr, dev_starts, dev_ends, n);
        
        // Update Arrays that show each thread what job to do
        update_arrays<<<32*numSMs, BLK_SZ>>>(dev_starts, dev_ends, n);
    }

    // Copy the result back to host
    cudaErrChk(cudaMemcpy(idArr, dev_idArr, n*sizeof(int), cudaMemcpyDeviceToHost));
    cudaErrChk(cudaMemcpy(distArr, dev_distArr, n*sizeof(float), cudaMemcpyDeviceToHost));

    // Clean-up
    cudaErrChk(cudaFree(dev_X));
    cudaErrChk(cudaFree(dev_idArr));
    cudaErrChk(cudaFree(dev_distArr));
    cudaErrChk(cudaFree(dev_starts));
    cudaErrChk(cudaFree(dev_ends));
    
    // Tree build
    vptree *root = (vptree *)malloc(sizeof(vptree));
    recursiveBuildTree(root, X, d, distArr, idArr, 0, n-1);

    // Return
    return root;
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
float getMD(vptree* T)     {    return T->md;   }  
float* getVP(vptree* T)    {    return T->vp;   }
int getIDX(vptree* T)      {    return T->idx;  }
