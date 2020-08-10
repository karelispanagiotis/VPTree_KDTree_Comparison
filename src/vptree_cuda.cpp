#include "vptree.h"
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#define BLK_SZ 256

using namespace thrust;

__device__ __forceinline__ float sqr(float x) {return x*x;}

__global__ void distCalc(float *X, int *idArr, float *distArr, int *ends, int n, int d)
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

__global__ void update_arrays(int *starts, int *ends, int n)
{
    for(int tid=blockIdx.x*BLK_SZ + threadIdx.x; tid<n; tid+=BLK_SZ*gridDim.x)
    {
        int start=starts[tid], end=ends[tid]-1;

        starts[tid] = (tid<=(start+end)/2) ? start : (start+end)/2 + 1;
        ends  [tid] = (tid<=(start+end)/2) ? (start+end)/2 : end;
    }
}

__global__ void make_segments(int *segments, int *starts, int *ends, int n)
{
    for(int tid=blockIdx.x*BLK_SZ + threadIdx.x; tid<n; tid+=BLK_SZ*gridDim.x)
    {
        int start=starts[tid], end=ends[tid];
        segments[tid] = (tid==start || tid>=end) ? 1 : 0;
    }
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

    // Allocate Memory on Device
    device_vector<float> dev_distArr(n), dev_X(n*d);
    device_vector<int> dev_idArr(n), dev_segments(n), dev_starts(n), dev_ends(n);

    // Initialize Device Variables
    copy(X, X+n*d, dev_X.begin());
    sequence(dev_idArr.begin(), dev_idArr.end());
    fill(dev_starts.begin(), dev_starts.end(), 0);
    fill(dev_ends.begin(), dev_ends.end(), n-1);
    
    // Build tree in GPU (level by level in parallel)
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    for(int i=0; i<floor(log2(n)); i++)
    {
        // Parallel Distance Calculation
        distCalc<<<32*numSMs, BLK_SZ>>>(raw_pointer_cast(&dev_X[0]), raw_pointer_cast(&dev_idArr[0]), raw_pointer_cast(&dev_distArr[0]), raw_pointer_cast(&dev_ends[0]), n, d);
        make_segments<<<32*numSMs, BLK_SZ>>>(raw_pointer_cast(&dev_segments[0]), raw_pointer_cast(&dev_starts[0]), raw_pointer_cast(&dev_ends[0]), n);

        // Parallel Sorting of segments [start, end]
        inclusive_scan(dev_segments.begin(), dev_segments.end(), dev_segments.begin());
        stable_sort_by_key(dev_distArr.begin(), dev_distArr.end(), make_zip_iterator(make_tuple(dev_idArr.begin(), dev_segments.begin())));
        stable_sort_by_key(dev_segments.begin(), dev_segments.end(), make_zip_iterator(make_tuple(dev_distArr.begin(), dev_idArr.begin())));

        // Update Arrays that show for each array position in which segment [start, end] it belongs to
        update_arrays<<<32*numSMs, BLK_SZ>>>(raw_pointer_cast(&dev_starts[0]), raw_pointer_cast(&dev_ends[0]), n);
    }

    // Copy the result back to host
    copy(dev_distArr.begin(), dev_distArr.end(), distArr);
    copy(dev_idArr.begin(), dev_idArr.end(), idArr);   

    // Tree build
    vptree *root = (vptree *)malloc(sizeof(vptree));
    recursiveBuildTree(root, X, d, distArr, idArr, 0, n-1);

    // Clean-up
    free(idArr); free(distArr);
    return root;
}

/////////////////////////////////////////////////////////////////////////////
vptree* getInner(vptree* T){    return T->inner;}
vptree* getOuter(vptree* T){    return T->outer;}
float getMD(vptree* T)     {    return T->md;   }  
float* getVP(vptree* T)    {    return T->vp;   }
int getIDX(vptree* T)      {    return T->idx;  }
