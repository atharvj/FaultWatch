#include "faultwatch/inference.hpp"

#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using namespace faultwatch;
using Sensors = std::array<double, sensor_count>;

struct Reading {
    std::int64_t cycle;
    Sensors sensors;
};

void predict_file(const Engine& engine, std::istream& source, std::ostream& output) {
    std::vector<std::int64_t> engines;
    std::vector<std::int64_t> cycles;
    std::vector<double> sensors;
    std::deque<Reading> window;
    std::int64_t current_engine = 0;

    const auto flush = [&] {
        if (window.size() == window_size) {
            engines.push_back(current_engine);
            for (const auto& row : window) {
                cycles.push_back(row.cycle);
                sensors.insert(sensors.end(), row.sensors.begin(), row.sensors.end());
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

    std::vector<float> predictions(engines.size());
    engine.predict(cycles, sensors, predictions);
    output << "engine_id,cycle,predicted_rul_cycles\n"
           << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < engines.size(); ++i) {
        output << engines[i] << ',' << cycles[(i + 1) * window_size - 1] << ',' << predictions[i]
               << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: faultwatch-native model.onnx test_FD001.txt\n";
        return 2;
    }
    try {
        std::ifstream model(argv[1], std::ios::binary);
        std::ifstream telemetry(argv[2]);
        if (!model || !telemetry) {
            throw std::runtime_error("Could not open model or telemetry file");
        }
        const std::vector<char> bytes{std::istreambuf_iterator<char>(model),
                                      std::istreambuf_iterator<char>()};
        const faultwatch::Engine engine(bytes);
        predict_file(engine, telemetry, std::cout);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
