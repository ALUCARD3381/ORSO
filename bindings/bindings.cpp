#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.hpp"

#include <string>
#include <vector>

namespace py = pybind11;
using orso::Tensor;

PYBIND11_MODULE(orso_core, m) {
    m.doc() = "ORSO native core - Phase 2 Tensor Engine";

    m.def("hello", []() {
        return std::string("ORSO core OK");
    }, "Verifica se o módulo C++ está carregado corretamente.");

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<std::size_t>&>(), py::arg("shape"))
        .def(py::init<const std::vector<std::size_t>&, const std::vector<float>&>(),
             py::arg("shape"), py::arg("data"))
        .def_static("zeros", &Tensor::zeros, py::arg("shape"))
        .def_static("ones", &Tensor::ones, py::arg("shape"))
        .def_static("full", &Tensor::full, py::arg("shape"), py::arg("value"))
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def("size", &Tensor::size)
        .def("empty", &Tensor::empty)
        .def("item", &Tensor::item)
        .def("get", &Tensor::get, py::arg("indices"))
        .def("set", &Tensor::set, py::arg("indices"), py::arg("value"))
        .def("data", [](const Tensor& self) { return self.data(); })
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("transpose", py::overload_cast<>(&Tensor::transpose, py::const_))
        .def("transpose", py::overload_cast<const std::vector<std::size_t>&>(&Tensor::transpose, py::const_), py::arg("axes"))
        .def("add", &Tensor::add, py::arg("other"))
        .def("sub", &Tensor::sub, py::arg("other"))
        .def("mul", py::overload_cast<const Tensor&>(&Tensor::mul, py::const_), py::arg("other"))
        .def("mul", py::overload_cast<float>(&Tensor::mul, py::const_), py::arg("scalar"))
        .def("div", py::overload_cast<const Tensor&>(&Tensor::div, py::const_), py::arg("other"))
        .def("div", py::overload_cast<float>(&Tensor::div, py::const_), py::arg("scalar"))
        .def("neg", &Tensor::neg)
        .def("matmul", &Tensor::matmul, py::arg("other"))
        .def("fill", &Tensor::fill, py::arg("value"))
        .def("__repr__", &Tensor::repr)
        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", [](const Tensor& self, py::object other) {
            if (py::isinstance<Tensor>(other)) {
                return self.mul(other.cast<Tensor>());
            }
            return self.mul(other.cast<float>());
        })
        .def("__rmul__", [](const Tensor& self, float scalar) {
            return self.mul(scalar);
        })
        .def("__truediv__", [](const Tensor& self, py::object other) {
            if (py::isinstance<Tensor>(other)) {
                return self.div(other.cast<Tensor>());
            }
            return self.div(other.cast<float>());
        })
        .def("__neg__", &Tensor::neg)
        .def("__matmul__", &Tensor::matmul);
}
