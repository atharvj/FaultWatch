#include "faultwatch/worker_pool.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <numeric>

namespace {
using namespace faultwatch;
using namespace std::chrono_literals;

TEST(BoundedQueueTest, DrainsInFifoOrderAfterClose) {
    BoundedQueue<int> queue(2);
    EXPECT_TRUE(queue.push(10));
    EXPECT_TRUE(queue.push(20));
    queue.close();
    EXPECT_FALSE(queue.push(30));
    EXPECT_EQ(queue.pop(), 10);
    EXPECT_EQ(queue.pop(), 20);
    EXPECT_FALSE(queue.pop());
}

TEST(BoundedQueueTest, WakesBlockedProducerOnClose) {
    BoundedQueue<int> queue(1);
    ASSERT_TRUE(queue.push(10));
    std::promise<void> entered;
    auto producer = std::async(std::launch::async, [&] {
        entered.set_value();
        return queue.push(20);
    });
    entered.get_future().wait();
    EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
    queue.close();
    EXPECT_FALSE(producer.get());
    EXPECT_EQ(queue.pop(), 10);
    EXPECT_FALSE(queue.pop());
}

TEST(BoundedQueueTest, FreesCapacityWhenConsumerPops) {
    BoundedQueue<int> queue(1);
    ASSERT_TRUE(queue.push(10));
    auto producer = std::async(std::launch::async, [&] { return queue.push(20); });
    EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(queue.pop(), 10);
    EXPECT_TRUE(producer.get());
    EXPECT_EQ(queue.pop(), 20);
}

TEST(WorkerPoolTest, BackpressuresAndDrainsMultipleDevices) {
    constexpr int devices = 8;
    constexpr int windows = 500;
    WorkerPool pool(
        [](const TelemetryWindow& input) { return static_cast<float>(input.cycles.back()); }, 4, 3);
    std::vector<std::future<double>> producers;
    for (int device = 0; device < devices; ++device) {
        producers.push_back(std::async(std::launch::async, [&pool, device] {
            std::vector<std::future<float>> predictions;
            for (int i = 1; i <= windows; ++i) {
                TelemetryWindow window;
                std::iota(window.cycles.begin(), window.cycles.end(), device * windows + i);
                predictions.push_back(pool.submit(std::move(window)));
            }
            double checksum = 0;
            for (auto& prediction : predictions) {
                checksum += prediction.get();
            }
            return checksum;
        }));
    }
    double actual = 0;
    for (auto& producer : producers) {
        actual += producer.get();
    }
    auto last = pool.submit({});
    auto closing = std::async(std::launch::async, [&] { pool.shutdown(); });
    pool.shutdown();
    closing.get();
    EXPECT_EQ(last.get(), 0.0);
    constexpr int count = devices * windows;
    EXPECT_EQ(actual, count * (count + 1) / 2 + count * (window_size - 1));
    EXPECT_THROW(pool.submit({}), std::runtime_error);
}

TEST(WorkerPoolTest, DeliversInputErrorsWithoutStoppingWorkers) {
    WorkerPool pool(
        [](const TelemetryWindow& window) {
            std::array<double, feature_count> features;
            extract_features(window.cycles, window.sensors, features);
            return static_cast<float>(features[0]);
        },
        1, 2);
    TelemetryWindow valid;
    std::iota(valid.cycles.begin(), valid.cycles.end(), 1);
    auto invalid = pool.submit({});
    auto good = pool.submit(valid);
    EXPECT_THROW((void)invalid.get(), std::invalid_argument);
    EXPECT_EQ(good.get(), 20.0);
    EXPECT_THROW(BoundedQueue<int>(0), std::invalid_argument);
    EXPECT_THROW(WorkerPool(WorkerPool::Scorer{}, 0, 1), std::invalid_argument);
}

}  // namespace
