#include "faultwatch/worker_pool.hpp"

namespace faultwatch {
namespace {

WorkerPool::Scorer engine_scorer(std::shared_ptr<const Engine> engine) {
    if (!engine) {
        throw std::invalid_argument("An inference engine is required");
    }
    return [engine = std::move(engine)](const TelemetryWindow& window) {
        std::array<float, 1> output;
        engine->predict(window.cycles, window.sensors, output);
        return output.front();
    };
}

}  // namespace

WorkerPool::WorkerPool(Scorer scorer, std::size_t workers, std::size_t capacity)
    : queue_(capacity), scorer_(std::move(scorer)) {
    if (!scorer_ || workers == 0) {
        throw std::invalid_argument("A scorer and at least one worker are required");
    }
    workers_.reserve(workers);
    try {
        for (std::size_t i = 0; i < workers; ++i) {
            workers_.emplace_back([this] {
                while (auto task = queue_.pop()) {
                    (*task)();
                }
            });
        }
    } catch (...) {
        queue_.close();
        throw;
    }
}

WorkerPool::WorkerPool(std::shared_ptr<const Engine> engine, std::size_t workers,
                       std::size_t capacity)
    : WorkerPool(engine_scorer(std::move(engine)), workers, capacity) {}

WorkerPool::~WorkerPool() { shutdown(); }

std::future<float> WorkerPool::submit(TelemetryWindow window) {
    std::packaged_task<float()> task([this, window = std::move(window)] { return scorer_(window); });
    auto ready = task.get_future();
    if (!queue_.push(std::move(task))) {
        throw std::runtime_error("Inference pool is closed");
    }
    return ready;
}

void WorkerPool::shutdown() {
    std::call_once(shutdown_, [this] {
        queue_.close();
        for (auto& worker : workers_) {
            worker.join();
        }
    });
}

}  // namespace faultwatch
