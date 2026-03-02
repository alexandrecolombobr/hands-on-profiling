#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <immintrin.h>
#include <limits>
#include <random>
#include <cstdlib>

struct Stats {
    double sum;
    double min;
    double max;
};

Stats compute_stats_avx2(const std::vector<int32_t>& data) {
    const size_t n = data.size();
    const size_t simdWidth = 8;               // AVX2: 8 x int32
    const size_t vec_blocks = n / simdWidth;

    int nthreads = omp_get_max_threads();

    std::vector<int64_t> thread_sum(nthreads, 0);
    std::vector<int32_t> thread_min(nthreads, std::numeric_limits<int32_t>::max());
    std::vector<int32_t> thread_max(nthreads, std::numeric_limits<int32_t>::lowest());

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

        __m256i sum_lo = _mm256_setzero_si256(); // 4 x int64
        __m256i sum_hi = _mm256_setzero_si256(); // 4 x int64

        __m256i min_vec = _mm256_set1_epi32(std::numeric_limits<int32_t>::max());
        __m256i max_vec = _mm256_set1_epi32(std::numeric_limits<int32_t>::lowest());

#pragma omp for schedule(static)
        for (size_t b = 0; b < vec_blocks; ++b) {
            size_t i = b * simdWidth;

            __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&data[i])
            );

            // min / max 32-bit
            min_vec = _mm256_min_epi32(min_vec, v);
            max_vec = _mm256_max_epi32(max_vec, v);

            // widening para 64-bit
            __m128i lo128 = _mm256_castsi256_si128(v);
            __m128i hi128 = _mm256_extracti128_si256(v, 1);

            __m256i lo64 = _mm256_cvtepi32_epi64(lo128);
            __m256i hi64 = _mm256_cvtepi32_epi64(hi128);

            sum_lo = _mm256_add_epi64(sum_lo, lo64);
            sum_hi = _mm256_add_epi64(sum_hi, hi64);
        }

        // Redução horizontal soma
        alignas(32) int64_t tmp_lo[4];
        alignas(32) int64_t tmp_hi[4];

        _mm256_store_si256((__m256i*)tmp_lo, sum_lo);
        _mm256_store_si256((__m256i*)tmp_hi, sum_hi);

        int64_t local_sum = 0;
        for (int i = 0; i < 4; ++i)
            local_sum += tmp_lo[i] + tmp_hi[i];

        // Redução horizontal min/max
        alignas(32) int32_t tmp_min[8];
        alignas(32) int32_t tmp_max[8];

        _mm256_store_si256((__m256i*)tmp_min, min_vec);
        _mm256_store_si256((__m256i*)tmp_max, max_vec);

        int32_t local_min = std::numeric_limits<int32_t>::max();
        int32_t local_max = std::numeric_limits<int32_t>::lowest();

        for (int i = 0; i < 8; ++i) {
            local_min = std::min(local_min, tmp_min[i]);
            local_max = std::max(local_max, tmp_max[i]);
        }

        // Processa cauda
#pragma omp for schedule(static)
        for (size_t i = vec_blocks * simdWidth; i < n; ++i) {
            int32_t v = data[i];
            local_sum += v;
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }

        thread_sum[tid] = local_sum;
        thread_min[tid] = local_min;
        thread_max[tid] = local_max;
    }

    // Redução final
    int64_t global_sum = 0;
    int32_t global_min = std::numeric_limits<int32_t>::max();
    int32_t global_max = std::numeric_limits<int32_t>::lowest();

    for (int t = 0; t < nthreads; ++t) {
        global_sum += thread_sum[t];
        global_min = std::min(global_min, thread_min[t]);
        global_max = std::max(global_max, thread_max[t]);
    }

    return {(global_sum / 100.0)
        , (global_min / 100.0)
        , (global_max / 100.0)};
}

inline int32_t parse_price_scaled(const char* begin, const char* end) {
    int32_t value = 0;
    int decimals = 0;
    bool after_dot = false;

    for (const char* p = begin; p < end; ++p) {
        char c = *p;

        if (c == '.') {
            after_dot = true;
            continue;
        }

        if (after_dot && decimals >= 2)
            break;

        value = value * 10 + (c - '0');

        if (after_dot)
            decimals++;
    }

    if (after_dot && decimals == 1)
        value *= 10;

    return value;
}

int main() {
    //omp_set_num_threads(1);
    double startTime = omp_get_wtime();

    // Quantidade configurável via argumento
    std::size_t valuesSize = 200000000; // default

    std::vector<int32_t> prices;
    prices.reserve(valuesSize);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(100000, 900000);

    for (std::size_t i = 0; i < valuesSize; ++i) {
        prices.emplace_back(dist(rng));
    }

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