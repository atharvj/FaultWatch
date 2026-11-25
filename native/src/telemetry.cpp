#include "faultwatch/telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <sstream>
#include <stdexcept>

namespace faultwatch {
namespace {

using Sensors = std::array<double, sensor_count>;

struct Reading {
    std::int64_t cycle;
    Sensors sensors;
};

}  // namespace

std::vector<EngineWindow> read_latest_windows(std::istream& source) {
    std::vector<EngineWindow> engines;
    std::deque<Reading> window;
    std::int64_t current_engine = 0;

    const auto flush = [&] {
        if (window.size() == window_size) {
            auto& output = engines.emplace_back(EngineWindow{current_engine, {}});
            for (std::size_t i = 0; i < window_size; ++i) {
                output.telemetry.cycles[i] = window[i].cycle;
                std::ranges::copy(window[i].sensors,
                                  output.telemetry.sensors.data() + i * sensor_count);
            }
        }
        window.clear();
    };

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(source, line)) {
        ++line_number;
        std::istringstream row(line);
        std::int64_t engine_id;
        Reading reading{};
        const auto invalid = [&] {
            return std::runtime_error("Invalid FD001 telemetry at line " +
                                      std::to_string(line_number));
        };
        if (!(row >> engine_id >> reading.cycle) || engine_id < 1 || reading.cycle < 1 ||
            engine_id < current_engine) {
            throw invalid();
        }
        double setting;
        for (int i = 0; i < 3; ++i) {
            if (!(row >> setting) || !std::isfinite(setting)) {
                throw invalid();
            }
        }
        for (auto& value : reading.sensors) {
            if (!(row >> value) || !std::isfinite(value)) {
                throw invalid();
            }
        }
        row >> std::ws;
        if (!row.eof()) {
            throw invalid();
        }
        if (engine_id != current_engine) {
            flush();
            current_engine = engine_id;
        }
        if (!window.empty() && reading.cycle - window.back().cycle != 1) {
            throw invalid();
        }
        if (window.size() == window_size) {
            window.pop_front();
        }
        window.push_back(reading);
    }
    if (!source.eof()) {
        throw std::runtime_error("Could not read FD001 telemetry file");
    }
    flush();
    return engines;
}

}  // namespace faultwatch
