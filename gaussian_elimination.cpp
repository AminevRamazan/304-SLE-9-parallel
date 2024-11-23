#include "gaussian_elimination.h"

pthread_barrier_t barrier; // Барьер для синхронизации
pthread_mutex_t mutex; // Мьютекс для синхронизации доступа к глобальному максимуму

struct ThreadData {
    std::vector<std::vector<double>>& A;
    std::vector<double>& b;
    std::vector<double>& x;
    int n;
    int tid; // ID потока
    int num_threads;
    int* globalMaxRow; // Указатель на переменную для строки с максимальным элементом

    ThreadData(std::vector<std::vector<double>>& A_, std::vector<double>& b_, std::vector<double>& x_, int n_, int tid_, int num_threads_, int* globalMaxRow_)
        : A(A_), b(b_), x(x_), n(n_), tid(tid_), num_threads(num_threads_), globalMaxRow(globalMaxRow_) {}
};

void* gaussianStep(void* arg) {
    struct ThreadData *data = (struct ThreadData*)arg;
    int n = data->n;
    int tid = data->tid;
    int num_threads = data->num_threads;

    for (int i = 0; i < n; ++i) {
        *(data->globalMaxRow) = i;
        
        // 1. Поиск строки с максимальным элементом в столбце i
        int localMaxRow = i;
        for (int k = i + tid; k < n; k += num_threads) {
            if (std::fabs(data->A[k][i]) > std::fabs(data->A[localMaxRow][i])) {
                localMaxRow = k;
            }
        }

        // Синхронизация всех потоков
        pthread_barrier_wait(&barrier);

        // Обновляем глобальный максимум, если локальный больше
        pthread_mutex_lock(&mutex);
        if (std::fabs(data->A[localMaxRow][i]) > std::fabs(data->A[*data->globalMaxRow][i])) {
            *(data->globalMaxRow) = localMaxRow;
        }
        pthread_mutex_unlock(&mutex);

        // Синхронизация всех потоков после поиска максимума
        pthread_barrier_wait(&barrier);

        // Перестановка строк
        if (tid == 0) {
            std::swap(data->A[i], data->A[*data->globalMaxRow]);
            std::swap(data->b[i], data->b[*data->globalMaxRow]);
            double factor = 1 / data->A[i][i];
            data->b[i] = factor * data->b[i];
            for (int j = i; j < n; ++j) {
                data->A[i][j] = factor * data->A[i][j];
            }
        }

        // Синхронизация перед началом прямого хода
        pthread_barrier_wait(&barrier);

        // 2. Прямой ход
        for (int k = i + 1 + tid; k < n; k += num_threads) {
            double factor = data->A[k][i];
            data->b[k] -= factor * data->b[i];
            for (int j = i; j < n; ++j) {
                data->A[k][j] -= factor * data->A[i][j];
            }
        }

        // Синхронизация после завершения прямого хода
        pthread_barrier_wait(&barrier);
    }

    // 3. Обратный ход
    for (int i = n - 1; i >= 0; --i) {
        if (tid == 0) {
            data->x[i] = data->b[i];
        }

        // Синхронизация после вычисления x[i]
        pthread_barrier_wait(&barrier);

        for (int k = i - 1 - tid; k >= 0; k -= num_threads) {
            data->b[k] -= data->A[k][i] * data->x[i];
        }

        // Синхронизация перед переходом к следующей строке
        pthread_barrier_wait(&barrier);
    }

    return nullptr;
}

int gaussianElimination(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x, int n, int num_threads) {
    x.resize(n);
    std::vector<std::vector<double>> A_copy = A;
    std::vector<double> b_copy = b;

    // Инициализация барьера для синхронизации потоков
    pthread_barrier_init(&barrier, nullptr, num_threads);

    int globalMaxRow = 0;

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data;
    thread_data.reserve(num_threads);

    // Создание и запуск потоков
    for (int tid = 0; tid < num_threads; ++tid) {
        thread_data.emplace_back(A_copy, b_copy, x, n, tid, num_threads, &globalMaxRow);
        pthread_create(&threads[tid], nullptr, gaussianStep, &thread_data[tid]);
    }

    // Ожидание завершения работы всех потоков
    for (int tid = 0; tid < num_threads; ++tid) {
        pthread_join(threads[tid], nullptr);
    }

    // Уничтожение барьера
    pthread_barrier_destroy(&barrier);

    return 1;
}
