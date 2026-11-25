#include "faultwatch/inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace faultwatch {
namespace {

constexpr auto sensor_window = window_size * sensor_count;
using Window = std::span<const double, sensor_window>;
using Sums = std::array<double, sensor_count>;

Sums sum_scalar(Window sensors) {
    Sums sums{};
    for (std::size_t row = 0; row < window_size; ++row) {
        const auto readings = sensors.subspan(row * sensor_count, sensor_count);
        for (std::size_t sensor = 0; sensor < sensor_count; ++sensor) {
            if (!std::isfinite(readings[sensor])) {
                throw std::invalid_argument("Telemetry must contain finite sensor values");
            }
            sums[sensor] += readings[sensor];
        }
    }
    return sums;
}

#if defined(__aarch64__)
Sums sum_neon(Window sensors) {
    constexpr auto pairs = sensor_count / 2;
    std::array<float64x2_t, pairs> sums{};
    auto finite = vdupq_n_u64(std::numeric_limits<std::uint64_t>::max());
    const auto limit = vdupq_n_f64(std::numeric_limits<double>::max());
    double tail = 0.0;
    for (std::size_t row = 0; row < window_size; ++row) {
        const auto* values = sensors.data() + row * sensor_count;
        for (std::size_t pair = 0; pair < pairs; ++pair) {
            const auto v = vld1q_f64(values + 2 * pair);
            sums[pair] = vaddq_f64(sums[pair], v);
            finite = vandq_u64(finite, vcleq_f64(vabsq_f64(v), limit));
        }
        tail += values[sensor_count - 1];
    }
    if (!vgetq_lane_u64(finite, 0) || !vgetq_lane_u64(finite, 1) || !std::isfinite(tail)) {
        throw std::invalid_argument("Telemetry contains nonfinite readings or overflowed");
    }
    Sums result;
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        vst1q_f64(result.data() + 2 * pair, sums[pair]);
    }
    result.back() = tail;
    return result;
}
#endif

void extract_window(std::span<const std::int64_t, window_size> cycles,
                    Window sensors, std::span<double, feature_count> features,
                    Sums (*sum)(Window)) {
    constexpr auto max_exact_cycle = std::int64_t{1} << 53;
    for (std::size_t row = 0; row < window_size; ++row) {
        if (cycles[row] < 1 || cycles[row] >= max_exact_cycle ||
            (row > 0 && cycles[row] - cycles[row - 1] != 1)) {
            throw std::invalid_argument("Telemetry cycles must be positive and consecutive");
        }
    }
    const auto sums = sum(sensors);
    features[0] = static_cast<double>(cycles.back());
    const auto last = sensors.last<sensor_count>();
    for (std::size_t sensor = 0; sensor < sensor_count; ++sensor) {
        features[1 + sensor] = last[sensor];
        features[1 + sensor_count + sensor] = sums[sensor] / window_size;
        features[1 + 2 * sensor_count + sensor] = last[sensor] - sensors[sensor];
    }
    if (!std::ranges::all_of(features, [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Telemetry feature computation overflowed");
    }
}

}  // namespace

bool neon_available() noexcept {
#if defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

void extract_features(std::span<const std::int64_t> cycles, std::span<const double> sensors,
                      std::span<double> features, FeatureKernel kernel) {
    const auto windows = cycles.size() / window_size;
    if (cycles.size() % window_size != 0 || sensors.size() / sensor_window != windows ||
        sensors.size() % sensor_window != 0 || features.size() / feature_count != windows ||
        features.size() % feature_count != 0) {
        throw std::invalid_argument("Incompatible telemetry or feature batch size");
    }
    auto sum = sum_scalar;
#if defined(__aarch64__)
    if (kernel != FeatureKernel::scalar) {
        sum = sum_neon;
    }
#else
    if (kernel == FeatureKernel::neon) {
        throw std::invalid_argument("ARM NEON is unavailable on this build");
    }
#endif
    for (std::size_t i = 0; i < windows; ++i) {
        extract_window(
            std::span<const std::int64_t, window_size>(cycles.data() + i * window_size, window_size),
            std::span<const double, sensor_window>(sensors.data() + i * sensor_window, sensor_window),
            std::span<double, feature_count>(features.data() + i * feature_count, feature_count),
            sum);
    }
}

}  // namespace faultwatch
