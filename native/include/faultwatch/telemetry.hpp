#pragma once

#include "faultwatch/inference.hpp"

#include <array>
#include <iosfwd>
#include <vector>

namespace faultwatch {

struct TelemetryWindow {
    std::array<std::int64_t, window_size> cycles{};
    std::array<double, window_size * sensor_count> sensors{};
};

struct EngineWindow {
    std::int64_t engine_id;
    TelemetryWindow telemetry;
};

std::vector<EngineWindow> read_latest_windows(std::istream& source);

}  // namespace faultwatch
