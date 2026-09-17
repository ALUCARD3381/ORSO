#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "orso/tensor.hpp"
#include "orso/transformer.hpp"
#include "orso/model.hpp"
#include "orso/optimizer.hpp"
#include "orso/model.hpp"

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
    m.def("cross_entropy", &orso::cross_entropy, py::arg("logits"), py::arg("targets"), py::arg("ignore_index")=-1);

    py::class_<orso::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &orso::ModelConfig::vocab_size)
        .def_readwrite("d_model", &orso::ModelConfig::d_model)
        .def_readwrite("num_heads", &orso::ModelConfig::num_heads)
        .def_readwrite("hidden_dim", &orso::ModelConfig::hidden_dim)
        .def_readwrite("num_layers", &orso::ModelConfig::num_layers)
        .def_readwrite("context_length", &orso::ModelConfig::context_length)
        .def_readwrite("seed", &orso::ModelConfig::seed);

    py::class_<orso::TransformerModel>(m, "Model")
        .def(py::init<const orso::ModelConfig&>(), py::arg("config")=orso::ModelConfig{})
        .def(py::init([](std::size_t vocab_size, std::size_t d_model, std::size_t num_heads,
                         std::size_t hidden_dim, std::size_t num_layers, std::size_t context_length,
                         std::uint64_t seed) {
            orso::ModelConfig cfg;
            cfg.vocab_size = vocab_size; cfg.d_model = d_model; cfg.num_heads = num_heads;
            cfg.hidden_dim = hidden_dim; cfg.num_layers = num_layers; cfg.context_length = context_length; cfg.seed = seed;
            return orso::TransformerModel(cfg);
        }), py::arg("vocab_size")=768, py::arg("d_model")=64, py::arg("num_heads")=8,
            py::arg("hidden_dim")=256, py::arg("num_layers")=6, py::arg("context_length")=32, py::arg("seed")=1234ULL)
        .def("forward", &orso::TransformerModel::forward)
        .def("parameters", &orso::TransformerModel::parameters)
        .def_property_readonly("parameter_count", &orso::TransformerModel::parameter_count)
        .def_property_readonly("context_length", [](const orso::TransformerModel& model){ return model.config().context_length; })
        .def_property_readonly("config", [](const orso::TransformerModel& model){ return model.config(); })
        .def("parameter_data", &orso::TransformerModel::parameter_data)
        .def("load_parameter_data", &orso::TransformerModel::load_parameter_data);

    py::class_<orso::AdamW>(m, "AdamW")
        .def(py::init<const std::vector<Tensor>&, float, float, float, float, float, float>(),
             py::arg("parameters"), py::arg("lr")=3.0e-4f, py::arg("beta1")=0.9f,
             py::arg("beta2")=0.999f, py::arg("eps")=1.0e-8f, py::arg("weight_decay")=0.01f,
             py::arg("max_grad_norm")=0.0f)
        .def("zero_grad", &orso::AdamW::zero_grad)
        .def("step", &orso::AdamW::step)
        .def_property("lr", &orso::AdamW::lr, &orso::AdamW::set_lr)
        .def_property_readonly("step_count", &orso::AdamW::step_count)
        .def_property_readonly("beta1", &orso::AdamW::beta1)
        .def_property_readonly("beta2", &orso::AdamW::beta2)
        .def_property_readonly("eps", &orso::AdamW::eps)
        .def_property_readonly("weight_decay", &orso::AdamW::weight_decay)
        .def_property_readonly("max_grad_norm", &orso::AdamW::max_grad_norm)
        .def("first_moment", &orso::AdamW::first_moment)
        .def("second_moment", &orso::AdamW::second_moment)
        .def("load_state", &orso::AdamW::load_state);
}
