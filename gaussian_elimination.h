#ifndef GAUSSIAN_ELIMINATION_H
#define GAUSSIAN_ELIMINATION_H

#include <vector>
#include <pthread.h>
#include <cmath>
#include <algorithm>

void* eliminateRows(void* arg);
int gaussianElimination(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x, int n, int num_threads);

#endif
