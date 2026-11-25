#include "faultwatch/telemetry.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace faultwatch;
void predict_file(const Engine& engine, std::istream& source, std::ostream& output) {
    const auto engines = read_latest_windows(source);
    std::vector<std::int64_t> cycles;
    std::vector<double> sensors;
    cycles.reserve(engines.size() * window_size);
    sensors.reserve(engines.size() * window_size * sensor_count);
    for (const auto& window : engines) {
        const auto& telemetry = window.telemetry;
        cycles.insert(cycles.end(), telemetry.cycles.begin(), telemetry.cycles.end());
        sensors.insert(sensors.end(), telemetry.sensors.begin(), telemetry.sensors.end());
    }
    std::vector<float> predictions(engines.size());
    engine.predict(cycles, sensors, predictions);
    output << "engine_id,cycle,predicted_rul_cycles\n"
           << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < engines.size(); ++i) {
        output << engines[i].engine_id << ',' << cycles[(i + 1) * window_size - 1] << ',' << predictions[i]
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
