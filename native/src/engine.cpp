#include "faultwatch/inference.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace faultwatch {
namespace {

Ort::Env& environment() {
    static Ort::Env env = [] {
        Ort::Env value(ORT_LOGGING_LEVEL_WARNING, "faultwatch");
        value.DisableTelemetryEvents();
        return value;
    }();
    return env;
}

Ort::SessionOptions options(int threads) {
    if (threads < 1) {
        throw std::invalid_argument("Inference thread count must be positive");
    }
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

}  // namespace

struct Engine::Session {
    Ort::Session value;
    Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    Session(std::span<const char> model, int threads)
        : value(environment(), model.data(), model.size(), options(threads)) {
        if (value.GetInputCount() != 1 || value.GetOutputCount() != 1) {
            throw std::invalid_argument("Expected one ONNX input and one output");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = value.GetInputNameAllocated(0, allocator);
        const auto output_name = value.GetOutputNameAllocated(0, allocator);
        const auto input_type = value.GetInputTypeInfo(0);
        const auto output_type = value.GetOutputTypeInfo(0);
        const auto input = input_type.GetTensorTypeAndShapeInfo();
        const auto output = output_type.GetTensorTypeAndShapeInfo();
        if (std::string_view(input_name.get()) != "features" ||
            std::string_view(output_name.get()) != "rul_cycles" ||
            input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
            output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input.GetShape() != std::vector<std::int64_t>{-1, feature_count} ||
            output.GetShape() != std::vector<std::int64_t>{-1}) {
            throw std::invalid_argument("The ONNX model does not match the FaultWatch tensor schema");
        }
    }
};

Engine::Engine(std::span<const char> model, int threads) {
    if (model.empty()) {
        throw std::invalid_argument("ONNX model is empty");
    }
    try {
        session_ = std::make_unique<Session>(model, threads);
    } catch (const Ort::Exception& error) {
        throw std::invalid_argument(std::string("Cannot load ONNX model: ") + error.what());
    }
}

Engine::~Engine() = default;

std::string Engine::metadata(std::string_view key) const {
    Ort::AllocatorWithDefaultOptions allocator;
    const auto value = session_->value.GetModelMetadata().LookupCustomMetadataMapAllocated(
        std::string(key).c_str(), allocator);
    return value ? value.get() : "";
}

void Engine::predict_features(std::span<double> features, std::span<float> predictions) const {
    if (features.size() % feature_count != 0 ||
        features.size() / feature_count != predictions.size()) {
        throw std::invalid_argument("Incompatible inference batch size");
    }
    if (predictions.empty()) {
        return;
    }
    const auto batch = static_cast<std::int64_t>(predictions.size());
    const std::array<std::int64_t, 2> input_shape{batch, feature_count};
    const std::array<std::int64_t, 1> output_shape{batch};
    const char* input_names[] = {"features"};
    const char* output_names[] = {"rul_cycles"};
    auto input = Ort::Value::CreateTensor<double>(session_->memory, features.data(), features.size(),
                                                 input_shape.data(), input_shape.size());
    auto output =
        Ort::Value::CreateTensor<float>(session_->memory, predictions.data(), predictions.size(),
                                        output_shape.data(), output_shape.size());
    session_->value.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, &output, 1);
    for (auto& value : predictions) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Model returned a nonfinite prediction");
        }
        value = std::max(value, 0.0F);
    }
}

void Engine::predict(std::span<const std::int64_t> cycles, std::span<const double> sensors,
                     std::span<float> predictions) const {
    if (cycles.size() % window_size != 0 || cycles.size() / window_size != predictions.size()) {
        throw std::invalid_argument("Incompatible prediction batch size");
    }
    std::array<double, feature_count> single{};
    std::vector<double> batch;
    std::span<double> features = single;
    if (predictions.size() != 1) {
        batch.resize(predictions.size() * feature_count);
        features = batch;
    }
    extract_features(cycles, sensors, features);
    predict_features(features, predictions);
}

std::string runtime_version() { return Ort::GetVersionString(); }

}  // namespace faultwatch
