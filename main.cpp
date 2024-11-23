#include <iostream>
#include <chrono>
#include "matrix_io.h"
#include "gaussian_elimination.h"
#include "functions.h"

int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " n p m k filename" << std::endl;
        return EXIT_FAILURE;
    }

    int n = std::stoi(argv[1]);       // Размерность матрицы
    int p = std::stoi(argv[2]);       // Количество потоков
    int m = std::stoi(argv[3]);       // Количество выводимых значений
    int k = std::stoi(argv[4]);       // Номер формулы или 0 для чтения из файла

    if (m > n) {
        std::cerr << "Ошибка: Количество выводимых значений больше размерности матрицы " << std::endl;
        return 1;
    }

    if (p <= 0) {
        std::cerr << "Ошибка: Количество потоков неположительно " << std::endl;
        return 2;
    }

    int err = 0;
    std::vector<std::vector<double>> A;
    std::vector<double> b, x;
    A.resize(n, std::vector<double>(n));
    b.resize(n);

    // Инициализация матрицы
    if (k == 0) {
        std::string filename = argv[5];   // Имя файла
        err = readMatrixFromFile(filename, A, n);
        if (!err) {
            std::cerr << "Ошибка: Не удалось открыть файл " << filename << std::endl;
            return err;
        }
    }
    else {
        initializeMatrix(A, k, n, p);
    }

    std::cout << "Исходная матрица A:" << std::endl;
    printMatrix(A, m);

    // Построение вектора b
    for (int i = 0; i < n; i++) {
        double sum_value = 0.0;
        for (int k = 0; (2 * k + 1) < n; k++) {
            sum_value += A[i][2 * k + 1];
        }
        b[i] = sum_value;
    }

    std::cout << "Правая часть b:" << std::endl;
    printVector(b, m);

    // Запуск таймера
    auto start = std::chrono::high_resolution_clock::now();

    // Решение системы методом Гаусса
    err = gaussianElimination(A, b, x, n, p);
    if (!err) {
        std::cerr << "Ошибка: Матрица вырождена " << std::endl;
        return err;
    }

    // Остановка таймера
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Вывод решения
    std::cout << "Решение x:" << std::endl;
    printVector(x, m);

    // Вычисление норм
    double residualNorm = calculateResidualNorm(A, x, b, n, p);
    double normError = calculateNormError(x, n, p);

    // Вывод результатов
    std::cout << "Норма невязки: " << std::scientific << residualNorm << std::endl;
    std::cout << "Норма погрешности: " << std::scientific << normError << std::endl;

    // Вывод времени выполнения
    std::cout << "Время решения: " << elapsed.count() << " секунд" << std::endl;

    return 0;
}
