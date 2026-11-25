#include "faultwatch/worker_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <latch>

namespace {
using namespace faultwatch;
using Clock = std::chrono::steady_clock;

struct DeviceReport {
    std::vector<double> submit_us;
    double checksum = 0;
};

void run(const std::shared_ptr<const Engine>& engine, const std::vector<EngineWindow>& windows,
         int devices, int workers, int capacity) {
    constexpr int windows_per_device = 1000;
    WorkerPool pool(engine, workers, capacity);
    std::latch start(1);
    std::vector<std::future<DeviceReport>> producers;
    double expected = 0;
    for (int device = 0; device < devices; ++device) {
        const auto& window = windows[device].telemetry;
        std::array<float, 1> score;
        engine->predict(window.cycles, window.sensors, score);
        expected += score.front() * static_cast<double>(windows_per_device);
    }
    try {
        for (int device = 0; device < devices; ++device) {
            producers.push_back(std::async(std::launch::async, [&, device] {
                DeviceReport output;
                output.submit_us.reserve(windows_per_device);
                std::vector<std::future<float>> scores;
                scores.reserve(windows_per_device);
                start.wait();
                for (int i = 0; i < windows_per_device; ++i) {
                    const auto before = Clock::now();
                    scores.push_back(pool.submit(windows[device].telemetry));
                    output.submit_us.push_back(
                        std::chrono::duration<double, std::micro>(Clock::now() - before).count());
                }
                for (auto& score : scores) {
                    output.checksum += score.get();
                }
                return output;
            }));
        }
    } catch (...) {
        start.count_down();
        throw;
    }
    const auto begin = Clock::now();
    start.count_down();
    std::vector<double> latencies;
    double checksum = 0;
    for (auto& producer : producers) {
        auto observed = producer.get();
        checksum += observed.checksum;
        latencies.insert(latencies.end(), observed.submit_us.begin(), observed.submit_us.end());
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    pool.shutdown();
    if (checksum != expected) {
        throw std::runtime_error("Concurrent scores changed");
    }
    std::ranges::sort(latencies);
    const auto p95 = latencies[static_cast<std::size_t>(std::ceil(0.95 * latencies.size())) - 1];
    std::cout << devices << ',' << workers << ',' << capacity << ',' << latencies.size() << ','
              << seconds << ',' << latencies.size() / seconds << ',' << p95 << ',' << checksum
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: faultwatch-worker-bench model.onnx test_FD001.txt\n";
        return 2;
    }
    try {
        std::ifstream model(argv[1], std::ios::binary);
        std::ifstream source(argv[2]);
        if (!model || !source) {
            throw std::runtime_error("Could not open model or telemetry");
        }
        const auto windows = read_latest_windows(source);
        if (windows.size() < 16) {
            throw std::runtime_error("The workload needs 16 complete device windows");
        }
        const std::vector<char> bytes{std::istreambuf_iterator<char>(model),
                                      std::istreambuf_iterator<char>()};
        const auto engine = std::make_shared<const faultwatch::Engine>(bytes, 1);
        std::cout << "devices,workers,queue_capacity,windows,seconds,windows_per_second,"
                     "submit_p95_us,checksum\n"
                  << std::setprecision(10);
        for (int devices : {1, 4, 16}) {
            for (int workers : {1, 2, 4}) {
                run(engine, windows, devices, workers, 16);
            }
        }
        run(engine, windows, 16, 4, 64);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
