#pragma once

#include "faultwatch/telemetry.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace faultwatch {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("Queue capacity must be positive");
        }
    }

    // Wait for a free slot. Closing the queue interrupts blocked producers.
    bool push(T item) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
        if (closed_) {
            return false;
        }
        items_.push_back(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !items_.empty(); });
        if (items_.empty()) {
            return std::nullopt;
        }
        auto item = std::move(items_.front());
        items_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return item;
    }

    void close() {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        not_full_.notify_all();
        not_empty_.notify_all();
    }

private:
    const std::size_t capacity_;
    std::deque<T> items_;
    std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    bool closed_ = false;
};

class WorkerPool {
public:
    using Scorer = std::function<float(const TelemetryWindow&)>;

    WorkerPool(Scorer scorer, std::size_t workers, std::size_t capacity);
    WorkerPool(std::shared_ptr<const Engine> engine, std::size_t workers, std::size_t capacity);
    ~WorkerPool();

    // Accepted jobs resolve their futures even when shutdown starts.
    std::future<float> submit(TelemetryWindow window);
    void shutdown();

private:
    BoundedQueue<std::packaged_task<float()>> queue_;
    Scorer scorer_;
    std::vector<std::jthread> workers_;
    std::once_flag shutdown_;
};

}  // namespace faultwatch
