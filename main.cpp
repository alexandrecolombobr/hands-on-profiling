#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

void totalizeValue(const double &price, double &total, double &max, double &min) {
    total += price;

    if (price > max)
        max = price;

    if (price < min || min == 0.0)
        min = price;
}

int main() {
    double startTime = omp_get_wtime();
    std::ifstream file("ETHUSDT.csv");

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file ETHUST.csv" << std::endl;
        return 1;
    }

    std::string line;

    double total = 0.0;
    double max = 0.0;
    double min = 0.0;
    double mean = 0.0;
    std::size_t lineCount = 0;

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
            totalizeValue(price, total, max, min);
            lineCount++;
        } catch (...) {
            std::cerr << "Invalid price in line: " << line << std::endl;
        }
    }

    file.close();
    double endTime = omp_get_wtime();

    std::cout << std::fixed;
    std::cout << "===================================================================" << std::endl;
    std::cout << "= Valor total negociado: $ " << total << std::endl;
    std::cout << "= Máximo valor negociado: $ " << max << std::endl;
    std::cout << "= Mínimo valor negociado: $ " << min << std::endl;
    std::cout << "= Valor médio negociado: $ " << total/lineCount << std::endl;
    std::cout << "= Total de linhas processadas: " << lineCount << std::endl;
    std::cout << "Tempo de processamento: " << (endTime - startTime) << " s" << std::endl;
    std::cout << "===================================================================" << std::endl;

    return 0;
}