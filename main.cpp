#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>
#include <immintrin.h>

struct Stats {
    double sum;
    double min;
    double max;
};

Stats compute_stats_avx2(const std::vector<double>& data) {
    const std::size_t n = data.size();
    const std::size_t simdWidth = 4;

    double global_sum = 0.0;
    double global_min = std::numeric_limits<double>::max();
    double global_max = std::numeric_limits<double>::lowest();

#pragma omp parallel
    {
        __m256d local_sum_vec = _mm256_setzero_pd();
        __m256d local_min_vec = _mm256_set1_pd(std::numeric_limits<double>::max());
        __m256d local_max_vec = _mm256_set1_pd(std::numeric_limits<double>::lowest());

        double local_sum = 0.0;
        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();

#pragma omp for nowait
        for (std::size_t i = 0; i < n - (n % simdWidth); i += simdWidth) {
            __m256d v = _mm256_loadu_pd(&data[i]);

            local_sum_vec = _mm256_add_pd(local_sum_vec, v);
            local_min_vec = _mm256_min_pd(local_min_vec, v);
            local_max_vec = _mm256_max_pd(local_max_vec, v);
        }

        alignas(32) double temp_sum[4];
        alignas(32) double temp_min[4];
        alignas(32) double temp_max[4];

        _mm256_store_pd(temp_sum, local_sum_vec);
        _mm256_store_pd(temp_min, local_min_vec);
        _mm256_store_pd(temp_max, local_max_vec);

        for (int i = 0; i < 4; ++i) {
            local_sum += temp_sum[i];
            local_min = std::min(local_min, temp_min[i]);
            local_max = std::max(local_max, temp_max[i]);
        }

#pragma omp for nowait
        for (std::size_t i = n - (n % simdWidth); i < n; ++i) {
            local_sum += data[i];
            local_min = std::min(local_min, data[i]);
            local_max = std::max(local_max, data[i]);
        }

#pragma omp critical
        {
            global_sum += local_sum;
            global_min = std::min(global_min, local_min);
            global_max = std::max(global_max, local_max);
        }
    }

    return {global_sum, global_min, global_max};
}

int main() {
    //omp_set_num_threads(1);
    double startTime = omp_get_wtime();
    std::ifstream file("BTCUSDT.csv");

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file ETHUST.csv" << std::endl;
        return 1;
    }

    std::vector<double> prices;
    prices.reserve(5575530);

    std::string line;

    // Skip header (remove if file has no header)
    std::getline(file, line);

    while (std::getline(file, line)) {

        size_t first = line.find(',');
        size_t second = line.find(',', first + 1);
        size_t third = line.find(',', second + 1);

        if (second == std::string::npos || third == std::string::npos)
            continue;

        std::string priceStr = line.substr(second + 1, third - second - 1);

        try {
            double price = std::stod(priceStr);
            prices.emplace_back(std::stod(
                line.substr(second + 1, third - second - 1)
            ));
        } catch (...) {
            std::cerr << "Invalid price in line: " << line << std::endl;
        }
    }

    file.close();

    Stats stats = compute_stats_avx2(prices);

    double endTime = omp_get_wtime();

    std::cout << std::fixed;
    std::cout << "===================================================================" << std::endl;
    std::cout << "= Valor total negociado: $ " << stats.sum << std::endl;
    std::cout << "= Máximo valor negociado: $ " << stats.max << std::endl;
    std::cout << "= Mínimo valor negociado: $ " << stats.min << std::endl;
    std::cout << "= Valor médio negociado: $ " << stats.sum/prices.size() << std::endl;
    std::cout << "= Total de linhas processadas: " << prices.size() << std::endl;
    std::cout << "Tempo de processamento: " << (endTime - startTime) << " s" << std::endl;
    std::cout << "===================================================================" << std::endl;

    return 0;
}