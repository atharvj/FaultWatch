#include "faultwatch/inference.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;
using namespace faultwatch;

namespace {

using Cycles = py::array_t<std::int64_t, py::array::c_style>;
using Sensors = py::array_t<double, py::array::c_style>;

std::size_t batch_size(const Cycles& cycles, const Sensors& sensors) {
    if (cycles.ndim() != 2 || sensors.ndim() != 3 || cycles.shape(0) != sensors.shape(0) ||
        cycles.shape(1) != window_size || sensors.shape(1) != window_size ||
        sensors.shape(2) != sensor_count) {
        throw std::invalid_argument("Expected cycles (batch, 20) and sensors (batch, 20, 21)");
    }
    if (reinterpret_cast<std::uintptr_t>(cycles.data()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(sensors.data()) % alignof(double) != 0) {
        throw std::invalid_argument("Telemetry arrays must be element-aligned");
    }
    return static_cast<std::size_t>(cycles.shape(0));
}

template <typename T, int Flags>
std::span<const T> view(const py::array_t<T, Flags>& array) {
    return {array.data(), static_cast<std::size_t>(array.size())};
}

template <typename T>
std::span<T> writable_view(py::array_t<T>& array) {
    return {array.mutable_data(), static_cast<std::size_t>(array.size())};
}

}  // namespace

PYBIND11_MODULE(_faultwatch_native, module) {
    module.def("runtime_version", &runtime_version);
    module.def(
        "features",
        [](const Cycles& cycles, const Sensors& sensors) {
            const auto count = static_cast<py::ssize_t>(batch_size(cycles, sensors));
            py::array_t<double> result({count, static_cast<py::ssize_t>(feature_count)});
            const auto input_cycles = view(cycles);
            const auto input_sensors = view(sensors);
            const auto output = writable_view(result);
            {
                py::gil_scoped_release release;
                extract_features(input_cycles, input_sensors, output);
            }
            return result;
        },
        py::arg("cycles").noconvert(), py::arg("sensors").noconvert());

    py::class_<Engine>(module, "Engine")
        .def(py::init([](py::bytes model, int threads) {
                 const std::string content = model;
                 py::gil_scoped_release release;
                 return std::make_unique<Engine>(std::span(content.data(), content.size()), threads);
             }),
             py::arg("model"), py::arg("threads") = 2)
        .def("metadata", &Engine::metadata)
        .def(
            "predict",
            [](const Engine& engine, const Cycles& cycles, const Sensors& sensors) {
                py::array_t<float> result(batch_size(cycles, sensors));
                const auto input_cycles = view(cycles);
                const auto input_sensors = view(sensors);
                const auto output = writable_view(result);
                {
                    py::gil_scoped_release release;
                    engine.predict(input_cycles, input_sensors, output);
                }
                return result;
            },
            py::arg("cycles").noconvert(), py::arg("sensors").noconvert());
}
