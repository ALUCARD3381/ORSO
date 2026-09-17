#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(orso_core, m) {
    m.doc() = "ORSO native core - initial pybind11 test";

    m.def("hello", []() {
        return std::string("ORSO core OK");
    }, "Verifica se o módulo C++ está carregado corretamente.");
}
