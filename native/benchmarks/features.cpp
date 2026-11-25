#include "faultwatch/inference.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using namespace faultwatch;

struct AlignedDelete {
    void operator()(double* p) const noexcept { std::free(p); }
};

void run(std::size_t batch, std::size_t offset, FeatureKernel kernel, int samples) {
    constexpr std::size_t alignment = 64;
    const auto count = batch * window_size * sensor_count;
    const auto bytes = (count + offset) * sizeof(double);
    const auto allocation = ((bytes + alignment - 1) / alignment) * alignment;
    std::unique_ptr<double, AlignedDelete> buffer(
        static_cast<double*>(std::aligned_alloc(alignment, allocation)));
    if (!buffer) {
        throw std::bad_alloc();
    }
    std::span<double> sensors(buffer.get() + offset, count);
    std::vector<std::int64_t> cycles(batch * window_size);
    std::vector<double> features(batch * feature_count);
    for (std::size_t i = 0; i < count; ++i) {
        sensors[i] = 1000 * std::sin(static_cast<double>(i) * 0.015625);
    }
    for (std::size_t i = 0; i < cycles.size(); ++i) {
        cycles[i] = static_cast<std::int64_t>(i % window_size + 1);
    }
    const auto repeats = std::max<std::size_t>(8, 8192 / batch);
    const auto measure = [&] {
        for (std::size_t i = 0; i < repeats; ++i) {
            extract_features(cycles, sensors, features, kernel);
        }
    };
    measure();
    std::vector<double> ns;
    for (int i = 0; i < samples; ++i) {
        const auto start = std::chrono::steady_clock::now();
        measure();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        ns.push_back(std::chrono::duration<double, std::nano>(elapsed).count() / (batch * repeats));
    }
    std::ranges::sort(ns);
    const auto percentile = [&](double p) {
        return ns[static_cast<std::size_t>(std::ceil(p * samples)) - 1];
    };
    std::cout << batch << ',' << offset * sizeof(double) << ','
              << (kernel == FeatureKernel::scalar ? "scalar" : "neon") << ',' << percentile(0.50)
              << ',' << percentile(0.95) << ',' << 1e9 / percentile(0.50) << ',' << features.back()
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const int samples = argc == 2 ? std::atoi(argv[1]) : 15;
    if (samples < 1 || samples > 1000) {
        return 2;
    }
    std::cout << "batch,buffer_offset_bytes,kernel,p50_ns_per_window,p95_ns_per_window,"
                 "windows_per_second,last_feature\n"
              << std::setprecision(10);
    for (const std::size_t batch : {1, 32, 1024}) {
        for (const std::size_t offset : {0, 1}) {
            run(batch, offset, FeatureKernel::scalar, samples);
            if (neon_available()) {
                run(batch, offset, FeatureKernel::neon, samples);
            }
        }
    }
}
