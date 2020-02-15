#include "kNN.h"
#include "vptree.h"
#include <mpi.h>

void swapPtr(float** ptr1, float** ptr2)
{
    float *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

void updateResult(knnresult* store, knnresult* newRes)
{
    // This function merges old and new knn results
    // since both results are sorted. The result is
    // stored inside store (1st parameter).

    int m = newRes->m, k = newRes->k; //get dimensions
    float tempDist[m*k];   //temp array, will be used in merging
    int tempId[m*k];        //temp array, will be used in merging

    memcpy(tempDist, store->ndist, m * k * sizeof(float));   //copies the data of store
    memcpy(tempId, store->nidx, m * k * sizeof(int));


    int t, n;  //indexes for temp and new arrays, used in merging  
    //for each point in query set (each row)
    for(int i=0; i<m; i++)
    {
        t = n = 0;  //all indexes point at the beginning of each array
        
        //for each of the k neighbours
        for(int j=0; j<k; j++)
        {
            //merge the arrays until k elements are complete
            if(tempDist[i*k + t] < newRes->ndist[i*k + n])
            {
                store->ndist[i*k + j] = tempDist[i*k + t];
                store->nidx [i*k + j] = tempId  [i*k + t]; 
                t++;
            }
            else
            {
                store->ndist[i*k + j] = newRes->ndist[i*k + n];
                store->nidx [i*k + j] = newRes->nidx [i*k + n];
                n++;
            }
        }
    }
}

int mod(int x, int n) { return (n+x%n)%n; }

knnresult vptree_distrAllkNN(float *X, int n, int d, int k)
{
    knnresult result, tempResult;
    /* result:     Holds the updated result
     *             in each iteration
     * tempResult: Holds the kNN of local data X (query)
     *             inside received data Y (corpus)
     */   

    int numtasks, rank, prev, next, tag=1;
    MPI_Request requests[2];
    MPI_Status  statuses[2];

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    prev = mod(rank-1, numtasks);
    next = mod(rank+1, numtasks);

    int idOffset = n*mod(rank-1, numtasks);
    printf("Process %d, idOffset = %d\n", rank, idOffset);
    
    float *Y = (float *)malloc(n*d*sizeof(float)); //will process and send data
    float *Z = (float *)malloc(n*d*sizeof(float)); //will receive data while processing

    MPI_Isend(X, n*d, MPI_FLOAT, next, tag, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(Y, n*d, MPI_FLOAT, prev, tag, MPI_COMM_WORLD, &requests[1]);

    vptree *root = buildvp(X, n, d, idOffset);
    result = vptree_kNN(root, X, n, d, k);

    free(root); //tree is stored in an array, freeing root deallocates all nodes of the tree

    MPI_Waitall(2, requests, statuses);

    for(int iter=1; iter<numtasks; iter++)
    {
        MPI_Isend(Y, n*d, MPI_FLOAT, next, tag, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(Z, n*d, MPI_FLOAT, prev, tag, MPI_COMM_WORLD, &requests[1]);

        idOffset = mod(idOffset-n, numtasks*n);
        root = buildvp(Y, n, d, idOffset);
        
        tempResult = vptree_kNN(root, X, n, d, k);
        updateResult(&result, &tempResult);

        free(root);
        free(tempResult.ndist); free(tempResult.nidx);

        MPI_Waitall(2, requests, statuses);
        swapPtr(&Y, &Z);
    }

    free(Y); free(Z);
    return result;
}