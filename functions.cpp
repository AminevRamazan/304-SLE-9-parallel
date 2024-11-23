#include "functions.h"

struct ThreadDataResidual {
    const std::vector<std::vector<double>>& A;
    const std::vector<double>& x;
    const std::vector<double>& b;
    std::vector<double>& Ax;
    double norm_b = 0.0;
    double norm_residual = 0.0;
    int n;
    int tid;
    int num_threads;

    ThreadDataResidual(const std::vector<std::vector<double>>& A_, const std::vector<double>& x_, const std::vector<double>& b_, std::vector<double>& Ax_, int n_, int tid_, int num_threads_)
        : A(A_), x(x_), b(b_), Ax(Ax_), n(n_), tid(tid_), num_threads(num_threads_) {}
};

void* calculateAx(void* arg) {
    ThreadDataResidual* data = static_cast<ThreadDataResidual*>(arg);
    int tid = data->tid;
    int num_threads = data->num_threads;
    for (int i = tid; i < data->n; i += num_threads) {
        for (int j = 0; j < data->n; ++j) {
            data->Ax[i] += data->A[i][j] * data->x[j];
        }
    }
    return nullptr;
}

void* calculateNorms(void* arg) {
    ThreadDataResidual* data = static_cast<ThreadDataResidual*>(arg);
    int tid = data->tid;
    int num_threads = data->num_threads;
    for (int i = tid; i < data->n; i += num_threads) {
        data->norm_b += data->b[i] * data->b[i];
        data->norm_residual += (data->Ax[i] - data->b[i]) * (data->Ax[i] - data->b[i]);
    }
    return nullptr;
}

double calculateResidualNorm(const std::vector<std::vector<double>>& A, const std::vector<double>& x, const std::vector<double>& b, const int n, const int num_threads) {
    std::vector<double> Ax(n, 0.0);
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadDataResidual> thread_data;

    thread_data.reserve(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
        thread_data.emplace_back(A, x, b, Ax, n, tid, num_threads);
        pthread_create(&threads[tid], nullptr, calculateAx, &thread_data[tid]);
    }
    for (int tid = 0; tid < num_threads; ++tid) {
        pthread_join(threads[tid], nullptr);
    }

    for (int tid = 0; tid < num_threads; ++tid) {
        pthread_create(&threads[tid], nullptr, calculateNorms, &thread_data[tid]);
    }
    for (int tid = 0; tid < num_threads; ++tid) {
        pthread_join(threads[tid], nullptr);
    }

    double norm_b = 0.0, norm_residual = 0.0;
    for (int tid = 0; tid < num_threads; ++tid) {
        norm_b += thread_data[tid].norm_b;
        norm_residual += thread_data[tid].norm_residual;
    }

    return std::sqrt(norm_residual) / std::sqrt(norm_b);
}

/////////////////////////////////////

struct ThreadDataError {
    const std::vector<double>& x;
    double norm = 0.0;
    int n;
    int tid;
    int num_threads;

    ThreadDataError(const std::vector<double>& x_, int n_, int tid_, int num_threads_)
        : x(x_), n(n_), tid(tid_), num_threads(num_threads_) {}
};

void* calculatePartialNormError(void* arg) {
    ThreadDataError* data = static_cast<ThreadDataError*>(arg);
    int tid = data->tid;
    int num_threads = data->num_threads;

    for (int k = tid; k < data->n; k += num_threads) {
        data->norm += (data->x[k] - (k % 2)) * (data->x[k] - (k % 2));
    }
    return nullptr;
}

double calculateNormError(const std::vector<double>& x, const int n, const int num_threads) {
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadDataError> thread_data;
    thread_data.reserve(num_threads);

    for (int tid = 0; tid < num_threads; ++tid) {
        thread_data.emplace_back(x, n, tid, num_threads);
        pthread_create(&threads[tid], nullptr, calculatePartialNormError, &thread_data[tid]);
    }
    for (int tid = 0; tid < num_threads; ++tid) {
        pthread_join(threads[tid], nullptr);
    }

    double norm = 0.0;
    for (int tid = 0; tid < num_threads; ++tid) {
        norm += thread_data[tid].norm;
    }

    return std::sqrt(norm);
}