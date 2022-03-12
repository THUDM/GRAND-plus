#ifndef GRAPH_H
#define GRAPH_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
#include <unordered_map>
#include <unordered_set>
using namespace std;
namespace py = pybind11;
template<typename K, typename V>
void print_map(std::unordered_map<K,V> const &m)
{
for (auto const& pair: m) {
    std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
}
class Graph
{
public:

    int *indptr_ptr; 
    int *indices_ptr;
    
    unsigned int *Degree;

    int indptr_size; 
    int indices_size;
    int NUMTHREAD;
    int num_nodes;
    uint32_t seed;

    Graph(py::array_t<int> indptr, py::array_t<int> indices, int seed_)
    {
        py::buffer_info indptr_bf = indptr.request(), indices_bf = indices.request();
        indptr_ptr = static_cast<int *>(indptr_bf.ptr);
        indices_ptr = static_cast<int *>(indices_bf.ptr);
        indices_size = indices_bf.size;
        indptr_size = indptr_bf.size;
        num_nodes = indptr_size - 1; 
        seed = seed_;
        NUMTHREAD = 40;
        Degree = new unsigned int [num_nodes];
        for(int i = 0; i< num_nodes; i++){
            Degree[i] = indptr_ptr[i+1] - indptr_ptr[i];
        }   
        omp_set_num_threads(NUMTHREAD);
    }
    
    static bool cmp(std::pair<int, double> a, std::pair<int, double> b){
        return a.second > b.second;
    }

    void gfpush_omp(py::array_t<int>& node_idx, py::array_t<int>& row_idx, py::array_t<int> & col_idx, py::array_t<double> & value, py::array_t<double> & coef, const double & rmax, const int & K){
        struct timeval t_start,t_end, t_1, t_2, t_3; 
        double timeCost;
        gettimeofday(&t_start, NULL); 
        

        py::buffer_info node_idx_bf = node_idx.request();
        int node_idx_size = node_idx_bf.size;
        int* node_idx_ptr = static_cast<int *>(node_idx_bf.ptr);
        py::buffer_info coef_bf = coef.request();
        py::buffer_info row_idx_bf = row_idx.request();
        py::buffer_info col_idx_bf = col_idx.request();
        py::buffer_info value_bf = value.request();

        double* coef_ptr = static_cast<double*>(coef_bf.ptr);
        int* row_idx_ptr = static_cast<int*> (row_idx_bf.ptr);
        int* col_idx_ptr = static_cast<int*> (col_idx_bf.ptr);
        double* value_ptr = static_cast<double *>(value_bf.ptr);
        int order = coef_bf.size;
        
        #pragma omp parallel for schedule(dynamic)
        for(int it = 0; it < node_idx_size ; it++ ){
            int thread_num =  omp_get_thread_num();
            unordered_map<int, double> residue_value = unordered_map<int, double>();
            unordered_map<int, double> reserve_value = unordered_map<int, double>();

            int node_id = node_idx_ptr[it]; 
            double r_sum = 1.0;
            residue_value[node_id] = 1.0;
            reserve_value[node_id] = 0.;
            for(int i = 0; i < order - 1; i++){
                unordered_map<int, double>  residue_value_tmp = unordered_map<int, double>();// residue_value;
                while(!residue_value.empty()){
                    auto iter = residue_value.begin();
                    int oldNode = iter->first;
                    residue_value.erase(oldNode);
                    double r = iter->second;
                    reserve_value[oldNode] += coef_ptr[i] * r;
                    if (Degree[oldNode] == 0){
                        residue_value_tmp[node_id] += r;
                    }
                    else if(r >= rmax * Degree[oldNode]){
                        double val_ = r / Degree[oldNode];
                        for(int j = indptr_ptr[oldNode]; j< indptr_ptr[oldNode + 1]; j++){
                            int newNode = indices_ptr[j];
                            residue_value_tmp[newNode] += val_;
                        }
                    }
                }
                residue_value = residue_value_tmp;
            }
            while(!residue_value.empty()){
                auto iter = residue_value.begin();
                int oldNode = iter->first;
                residue_value.erase(oldNode);
                double r = iter->second;
                reserve_value[oldNode] += coef_ptr[order-1] * r;
            }
            vector<pair<int, double>> res(reserve_value.begin(), reserve_value.end());
            
            int k = res.size()> K ? K : res.size();
        
            std::nth_element(res.begin(), res.begin() + k - 1, res.end(), cmp);

            for(int i =0; i< k; i++){
	        int col_id = res[i].first;
                double v_ = res[i].second;
                int idx = it * K + i;
		if (v_ > 0.0){
                    row_idx_ptr[idx] = node_id;
                    col_idx_ptr[idx] = col_id;
                    value_ptr[idx] = v_;
		}
            }
        }
	
        gettimeofday(&t_end, NULL); 
        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
    }

};

#endif
