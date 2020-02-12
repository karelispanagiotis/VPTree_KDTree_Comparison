#ifndef KDTREE_H
#define KDTREE_H

typedef struct kdtree kdtree;

struct kdtree{
    float *p;
    float mc;
    int idx;
    int axis;
    kdtree *left;
    kdtree *right;
};

kdtree *buildkd(float *X, int n, int d);

float* getPoint(kdtree *node);   
float getMC(kdtree *node);       
int getIdx(kdtree *node);        
int getAxis(kdtree *node);       
kdtree *getLeft(kdtree *node);   
kdtree *getRight(kdtree *node);

#endif