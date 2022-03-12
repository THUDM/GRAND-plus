#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
#include "graph.h"
namespace py = pybind11;


PYBIND11_MODULE(propagation, m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<py::array_t<int> , py::array_t<int> , int>())
        .def("gfpush_omp", &Graph::gfpush_omp);
    }
