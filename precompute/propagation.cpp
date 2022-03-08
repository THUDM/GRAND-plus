#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
#include "graph.h"
namespace py = pybind11;


PYBIND11_MODULE(propagation, m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<py::array_t<int> , py::array_t<int> , int>())
        .def("forward_ppr_omp", &Graph::forward_ppr_omp)
        .def("forward_ppr_omp_map", &Graph::forward_ppr_omp_map)
        .def("forward_ppr_omp_map_w", &Graph::forward_ppr_omp_map_w)
        .def("forward_rw_omp", &Graph::forward_rw_omp)
        .def("rand_max", &Graph::rand_max);
}
