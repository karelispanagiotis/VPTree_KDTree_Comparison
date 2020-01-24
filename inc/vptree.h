#ifndef VPTREE_H
#define VPTREE_H

// type definition of vptree
typedef struct vptree vptree;

// ========== LIST OF ACCESSORS
struct vptree
{
    double *vp; //the vantage point
    double md;  //the median distance of the vantage point to the others
    int idx;    //the index of the vantage point in the original set
    vptree *inner;
    vptree *outer;
};

/////////////////////////////////////////////////////////////////////////////

//! Build vantage-point tree given input dataset X
/*!
    \param X Input data points, stored as [n-by-d] array
    \param n Number of data points (rows of X)
    \param d number of dimensions (columns of X)
    \return The vantage-point tree
*/
vptree *buildvp(double *X, int n, int d);

/////////////////////////////////////////////////////////////////////////////

//! Return vantage-point subtree with points inside radius
/*!
    \param node A vantage-point tree
    \return The vantage-point subtree
*/
vptree * getInner(vptree * T);

/////////////////////////////////////////////////////////////////////////////

//! Return vantage-point subtree with points outside radius
/*!
    \param node A vantage-point tree
    \return The vantage point subtree
*/
vptree * getOuter(vptree * T);

/////////////////////////////////////////////////////////////////////////////

//! Return median of distances to vantage point
/*!
    \param node A vantage-point tree
    \return The median distance
*/
double getMD(vptree * T);


/////////////////////////////////////////////////////////////////////////////

//! Return the coordinates of the vantage point
/*!
    \param node A vantage-point tree
    \return The coordinates [d-dimensional vector]
*/
double * getVP(vptree * T);

/////////////////////////////////////////////////////////////////////////////

//! Return the index of the vantage point
/*!
    \param node A vantage-point tree
    \return The index to the input vector of data points
*/
int getIDX(vptree * T);


/////////////////////////////////////////////////////////////////////////////

#endif
