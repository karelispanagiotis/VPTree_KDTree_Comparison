# VP-Tree vs KD-Tree Performance Comparison
* _Implementation of VP-Tree and KD-Tree construction algorithms on GPU, using CUDA C/C++ and comparison of execution times._
* _Comparison of points visited by kNN search algorithms using the above tree structures._

## How to Use
You can test construction times and points visited for a random dataset with size and dimensions of your preference by doing:

>$ cd test/  
 $ make test_knn  
 $ ./test_knn n m d k

Where
* n : size of dataset (Corpus Set)
* m : size of query points (Query Set)
* d : number of dimensions
* k : number of Nearest Neighbors

This benchmark shows construction times of each tree, nodes visited during kNN search and results validation. 

Compilation might take a while.