#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
#include "graph.h"
namespace py = pybind11;


// namespace py = pybind11;


/*
py::array_t<double> reverse_prop(py::array_t<int> indptr, py::array_t<int> indices, py::array_t<double>
feats, double rmax, double alpha, int p) {
    py::buffer_info indptr_bf = indptr.request(), indices_bf = indices.request(), feats_bf = feats.request();

    // if (buf1.ndim != 1 || buf2.ndim != 1)
        // throw std::runtime_error("Number of dimensions must be one");

    // if (buf1.size != buf2.size)
        // throw std::runtime_error("Input shapes must match");

    // No pointer is passed, so NumPy will allocate the buffer 
    auto result = py::array_t<double>(feats_bf.size * 2);

    py::buffer_info result_bf = result.request();
    int *indptr_ptr = static_cast<int *>(indptr_bf.ptr);
    int *indices_ptr = static_cast<int *>(indices_bf.ptr);
    double *feats_ptr = static_cast<double *>(feats_bf.ptr);
    double *reserve_ptr = static_cast<double *>(result_bf.ptr);
    double *residue_ptr = reserve_ptr + feats_bf.size;
    int indptr_size = indptr_bf.shape[0];
    int indices_size = indices_bf.shape[0];//shape.size();
    int num_nodes = feats_bf.shape[0];
    int feat_dim = feats_bf.shape[1];
    Graph G(indptr_ptr, indices_ptr, feats_ptr, reserve_ptr, residue_ptr, indptr_size, indices_size, feat_dim, num_nodes, rmax, alpha);
    //G.print_graph();
    // G.ppr_push();
    if(p==1){
        G.BackwardPush_ppr_omp();
    }
    else{
        //G.print_graph();
        // G.ppr_push();
        G.ppr_push();
    }
    // cout << num_nodes << feat_dim <<indices_size<< G.indices_size<<endl;
    copy(G.positive_feat, G.positive_feat + feat_dim * num_nodes, feats_ptr);
    return result;
}


PYBIND11_MODULE(propagation, m) {
    m.def("reverse_prop", &reverse_prop, "Add two NumPy arrays");
}
*/
PYBIND11_MODULE(propagation, m) {
    py::class_<Graph>(m, "Graph")
        .def(py::init<py::array_t<int> , py::array_t<int> , int>())
        .def("forward_ppr_omp", &Graph::forward_ppr_omp)
        .def("forward_ppr_omp_map", &Graph::forward_ppr_omp_map)
        .def("forward_ppr_omp_map_w", &Graph::forward_ppr_omp_map_w)
        .def("forward_rw_omp", &Graph::forward_rw_omp)
        .def("rand_max", &Graph::rand_max);
        // .def_readonly("row_idx", &Graph::row_idx)
        // .def_readonly("col_idx", &Graph::col_idx)
        // .def_readonly("ppr_value", &Graph::ppr_value)
}