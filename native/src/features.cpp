#include "faultwatch/inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace faultwatch {
namespace {

void extract_window(std::span<const std::int64_t, window_size> cycles,
                    std::span<const double, window_size * sensor_count> sensors,
                    std::span<double, feature_count> features) {
    constexpr auto max_exact_cycle = std::int64_t{1} << 53;
    std::array<double, sensor_count> sums{};
    for (std::size_t row = 0; row < window_size; ++row) {
        if (cycles[row] < 1 || cycles[row] >= max_exact_cycle ||
            (row > 0 && cycles[row] - cycles[row - 1] != 1)) {
            throw std::invalid_argument("Telemetry cycles must be positive and consecutive");
        }
        const auto readings = sensors.subspan(row * sensor_count, sensor_count);
        for (std::size_t sensor = 0; sensor < sensor_count; ++sensor) {
            if (!std::isfinite(readings[sensor])) {
                throw std::invalid_argument("Telemetry must contain finite sensor values");
            }
            sums[sensor] += readings[sensor];
        }
    }
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

void extract_features(std::span<const std::int64_t> cycles, std::span<const double> sensors,
                      std::span<double> features) {
    constexpr auto sensor_window = window_size * sensor_count;
    const auto windows = cycles.size() / window_size;
    if (cycles.size() % window_size != 0 || sensors.size() / sensor_window != windows ||
        sensors.size() % sensor_window != 0 || features.size() / feature_count != windows ||
        features.size() % feature_count != 0) {
        throw std::invalid_argument("Incompatible telemetry or feature batch size");
    }
    for (std::size_t i = 0; i < windows; ++i) {
        extract_window(
            std::span<const std::int64_t, window_size>(cycles.data() + i * window_size, window_size),
            std::span<const double, sensor_window>(sensors.data() + i * sensor_window, sensor_window),
            std::span<double, feature_count>(features.data() + i * feature_count, feature_count));
    }
}

}  // namespace faultwatch
