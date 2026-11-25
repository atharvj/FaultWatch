#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace faultwatch {

inline constexpr std::size_t window_size = 20;
inline constexpr std::size_t sensor_count = 21;
inline constexpr std::size_t feature_count = 1 + 3 * sensor_count;

// Each batch row contains 20 cycles and 20 by 21 row-major sensor values.
void extract_features(std::span<const std::int64_t> cycles, std::span<const double> sensors,
                      std::span<double> features);

class Engine {
public:
    explicit Engine(std::span<const char> model, int threads = 2);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    std::string metadata(std::string_view key) const;
    void predict_features(std::span<double> features, std::span<float> predictions) const;
    void predict(std::span<const std::int64_t> cycles, std::span<const double> sensors,
                 std::span<float> predictions) const;

private:
    struct Session;
    std::unique_ptr<Session> session_;
};

std::string runtime_version();

}  // namespace faultwatch
