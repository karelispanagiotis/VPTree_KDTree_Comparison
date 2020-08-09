#include "utilities.h"
#include <bits/stdc++.h>

void quickSelect(int kpos, float* values, int* idArr, int start, int end)
{
    int store=start;
    float pivot=values[end];
    for (int i=start; i<=end; i++)
        if (values[i] <= pivot)
        {
            std::swap(values[i], values[store]);
            std::swap( idArr[i],  idArr[store]);
            store++;
        }        
    store--;
    if (store == kpos) return;
    else if (store < kpos) quickSelect(kpos, values, idArr, store+1, end);
    else quickSelect(kpos, values, idArr, start, store-1);
}