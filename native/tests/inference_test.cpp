#include "faultwatch/inference.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
using namespace faultwatch;

class Window : public testing::Test {
protected:
    std::array<std::int64_t, window_size> cycles{};
    std::array<double, window_size * sensor_count> sensors{};
    std::array<double, feature_count> features{};

    void SetUp() override {
        std::iota(cycles.begin(), cycles.end(), 71);
        for (std::size_t row = 0; row < window_size; ++row) {
            for (std::size_t sensor = 0; sensor < sensor_count; ++sensor) {
                sensors[row * sensor_count + sensor] = static_cast<double>(row * 2 + sensor);
            }
        }
    }
};

TEST_F(Window, ComputesOrderedFeatures) {
    extract_features(cycles, sensors, features);
    EXPECT_DOUBLE_EQ(features[0], 90);
    for (std::size_t sensor = 0; sensor < sensor_count; ++sensor) {
        EXPECT_DOUBLE_EQ(features[1 + sensor], 38 + sensor);
        EXPECT_DOUBLE_EQ(features[1 + sensor_count + sensor], 19 + sensor);
        EXPECT_DOUBLE_EQ(features[1 + 2 * sensor_count + sensor], 38);
    }
}

TEST_F(Window, RejectsBrokenWindows) {
    EXPECT_THROW(extract_features(std::span(cycles).first(19), sensors, features),
                 std::invalid_argument);
    EXPECT_THROW(extract_features(cycles, std::span(sensors).subspan(1), features),
                 std::invalid_argument);
    cycles[12] = cycles[11];
    EXPECT_THROW(extract_features(cycles, sensors, features), std::invalid_argument);
}

TEST_F(Window, RejectsNonfiniteReadingsAndOverflow) {
    sensors[20] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(extract_features(cycles, sensors, features), std::invalid_argument);
    sensors[20] = 0;
    sensors[0] = std::numeric_limits<double>::max();
    sensors[sensor_count] = std::numeric_limits<double>::max();
    EXPECT_THROW(extract_features(cycles, sensors, features), std::invalid_argument);
}

TEST(EngineTest, RejectsEmptyOrMalformedModels) {
    EXPECT_THROW(Engine(std::span<const char>{}), std::invalid_argument);
    const std::string junk = "not an ONNX graph";
    EXPECT_THROW(Engine(std::span(junk.data(), junk.size())), std::invalid_argument);
}

}  // namespace
