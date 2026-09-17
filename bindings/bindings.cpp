#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "orso/tensor.hpp"
#include "orso/transformer.hpp"

namespace py = pybind11;
using orso::Tensor;

PYBIND11_MODULE(orso_core, m) {
    m.doc() = "ORSO native core — Tensor Engine, Autograd, NEON and Transformer building blocks";

    m.def("hello", [](){ return std::string("ORSO core OK"); });
    m.def("neon_enabled", [](){
#ifdef ORSO_USE_NEON
        return true;
#else
        return false;
#endif
    });

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const orso::Shape&, float, bool>(), py::arg("shape"), py::arg("fill")=0.0f, py::arg("requires_grad")=false)
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("requires_grad")=false)
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("requires_grad")=false)
        .def_static("full", &Tensor::full, py::arg("shape"), py::arg("value"), py::arg("requires_grad")=false)
        .def_static("random_normal", &Tensor::random_normal, py::arg("shape"), py::arg("mean")=0.0f, py::arg("stddev")=1.0f, py::arg("seed")=0ULL, py::arg("requires_grad")=false)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("size", &Tensor::size)
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def_property_readonly("data", [](const Tensor& t){ return t.data(); })
        .def_property_readonly("grad", [](const Tensor& t){ return t.grad(); })
        .def("item", &Tensor::item)
        .def("get", &Tensor::get)
        .def("set", &Tensor::set)
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", [](Tensor& self, py::object grad_obj){
            if (grad_obj.is_none()) { self.backward(); return; }
            auto grad = grad_obj.cast<Tensor>(); self.backward(&grad);
        }, py::arg("grad")=py::none())
        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose)
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("__repr__", &Tensor::repr);

    m.def("add", py::overload_cast<const Tensor&, const Tensor&>(&orso::add));
    m.def("add_scalar", py::overload_cast<const Tensor&, float>(&orso::add));
    m.def("sub", py::overload_cast<const Tensor&, const Tensor&>(&orso::sub));
    m.def("sub_scalar", py::overload_cast<const Tensor&, float>(&orso::sub));
    m.def("mul", py::overload_cast<const Tensor&, const Tensor&>(&orso::mul));
    m.def("mul_scalar", py::overload_cast<const Tensor&, float>(&orso::mul));
    m.def("div", py::overload_cast<const Tensor&, const Tensor&>(&orso::div));
    m.def("div_scalar", py::overload_cast<const Tensor&, float>(&orso::div));
    m.def("neg", &orso::neg);
    m.def("exp", &orso::exp);
    m.def("log", &orso::log);
    m.def("sqrt", &orso::sqrt);
    m.def("silu", &orso::silu);
    m.def("matmul", &orso::matmul);
    m.def("softmax", &orso::softmax, py::arg("x"), py::arg("axis")=-1);
    m.def("rmsnorm", &orso::rmsnorm, py::arg("x"), py::arg("weight"), py::arg("eps")=1e-5f);
    m.def("rope", &orso::rope, py::arg("x"), py::arg("theta")=10000.0f);
    m.def("embedding", &orso::embedding_lookup, py::arg("weight"), py::arg("token_ids"));
    m.def("causal_mask", &orso::causal_mask);
    m.def("multi_head_attention", &orso::multi_head_attention,
          py::arg("x"), py::arg("wq"), py::arg("wk"), py::arg("wv"), py::arg("wo"),
          py::arg("num_heads"), py::arg("causal")=true, py::arg("rope_theta")=10000.0f);
    m.def("swiglu", &orso::swiglu);
}
