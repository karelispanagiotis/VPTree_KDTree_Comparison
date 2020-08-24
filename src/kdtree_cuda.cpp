#include "kdtree.h"
#include "bits/stdc++.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#define BLK_SZ 256

using namespace thrust;

__global__ void copy_dim(float *X, int *idArr, float *auxArr, int *starts, int n, int d, int dim)
{
    for(int tid=blockIdx.x*BLK_SZ + threadIdx.x; tid<n; tid+=BLK_SZ*gridDim.x)
    {
        int start=starts[tid], idx=idArr[tid];

        if(tid<start) continue;
        
        auxArr[tid] = X[idx*d + dim];
    }
}

__global__ static void make_segments(int *segments, int *starts, int *ends, int n)
{
    for(int tid=blockIdx.x*BLK_SZ + threadIdx.x; tid<n; tid+=BLK_SZ*gridDim.x)
    {
        int start=starts[tid], end=ends[tid];
        segments[tid] = (tid<=start) ? 1 : 0;
    }
}

__global__ static void update_arrays(int *starts, int *ends, int n)
{
    for(int tid=blockIdx.x*BLK_SZ + threadIdx.x; tid<n; tid+=BLK_SZ*gridDim.x)
    {
        int start=starts[tid], end=ends[tid];

        starts[tid] = (tid<(start+end)/2) ? start : (start+end)/2 + 1;
        ends  [tid] = (tid<(start+end)/2) ? (start+end)/2 - 1 : end;
    }
}

void recursiveBuildTree(kdtree *node, float *X, int n, int d, float *mdArr, int *idArr, int start, int end, int depth)
{
    float(*dataArr)[d] = (float(*)[d])X;

    node->axis = depth%d;

    if(start==end)
    {
        node->idx = idArr[start];
        node->p   = dataArr[ idArr[start] ];
        node->mc  = 0.0;
        node->left = node->right = NULL;
        return;
    }

    node->idx = idArr[(start+end)/2];
    node->p   = dataArr[ idArr[(start+end)/2] ];
    node->mc  = mdArr[(start+end)/2];
    node->right = (kdtree *)malloc(sizeof(kdtree));

    // Recursion
    recursiveBuildTree(node->right, X, n, d, mdArr, idArr, (start+end)/2+1, end, depth+1);
    if(start<(start+end)/2)
    {
        node->left  = (kdtree *)malloc(sizeof(kdtree));
        recursiveBuildTree(node->left, X, n, d, mdArr, idArr, start, (start+end)/2 - 1, depth+1);
    }
    else node->left = NULL;
}

kdtree *buildkd(float *X, int n, int d)
{
    int *idArr = (int *)malloc(n*sizeof(int));
    float *mdArr = (float *)malloc(n*sizeof(int));

    // Allocate Memory on Device
    device_vector<float> dev_auxArr(n), dev_X(n*d);
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
        // Parallel Copy of Dim
        copy_dim<<<32*numSMs, BLK_SZ>>>(raw_pointer_cast(&dev_X[0]), raw_pointer_cast(&dev_idArr[0]), raw_pointer_cast(&dev_auxArr[0]), raw_pointer_cast(&dev_starts[0]), n, d, i%d );
        make_segments<<<32*numSMs, BLK_SZ>>>(raw_pointer_cast(&dev_segments[0]), raw_pointer_cast(&dev_starts[0]), raw_pointer_cast(&dev_ends[0]), n);

        // Parallel Sorting of segments [start, end]
        inclusive_scan(dev_segments.begin(), dev_segments.end(), dev_segments.begin());
        stable_sort_by_key(dev_auxArr.begin(), dev_auxArr.end(), make_zip_iterator(make_tuple(dev_idArr.begin(), dev_segments.begin())));
        stable_sort_by_key(dev_segments.begin(), dev_segments.end(), make_zip_iterator(make_tuple(dev_auxArr.begin(), dev_idArr.begin())));

        // Update Arrays that show for each array position in which segment [start, end] it belongs to
        update_arrays<<<32*numSMs, BLK_SZ>>>(raw_pointer_cast(&dev_starts[0]), raw_pointer_cast(&dev_ends[0]), n);
    }

    // Copy the result back to host
    copy(dev_auxArr.begin(), dev_auxArr.end(), mdArr);
    copy(dev_idArr.begin(), dev_idArr.end(), idArr);  

    // Tree build
    kdtree *root = (kdtree *)malloc(sizeof(kdtree));
    recursiveBuildTree(root, X, n, d, mdArr, idArr, 0, n-1, 0);

    // Clean-up
    free(idArr); free(mdArr);
    return root;
}

float* getPoint(kdtree *node)   {return node->p;}
float getMC(kdtree *node)       {return node->mc;}
int getIdx(kdtree *node)        {return node->idx;}
int getAxis(kdtree *node)       {return node->axis;}
kdtree *getLeft(kdtree *node)   {return node->left;}
kdtree *getRight(kdtree *node)  {return node->right;}