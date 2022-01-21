#ifndef GRAPH_H
#define GRAPH_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
#include <unordered_map>
#include <unordered_set>
#include "Random.h"
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
    int prune_step = 10;
    // py::array_t<int> row_idx;
    // py::array_t<int> col_idx;
    // py::array_t<double> ppr_value;
    vector<vector<int>> ppr_walk;
    Random R;
    uint32_t seed;

    Graph(py::array_t<int> indptr, py::array_t<int> indices, int seed_)
    {
        py::buffer_info indptr_bf = indptr.request(), indices_bf = indices.request();
        indptr_ptr = static_cast<int *>(indptr_bf.ptr);
        indices_ptr = static_cast<int *>(indices_bf.ptr);
        indices_size = indices_bf.size;
        indptr_size = indptr_bf.size;
        // reserve_ptr =  static_cast<float *>(reserve_bf.ptr);
        // residue_ptr = static_cast<float *>(residue_bf.ptr);
        num_nodes = indptr_size - 1; 
        seed = seed_;
        // copy(positive_feat, positive_feat + feat_dim * num_nodes, reserve_ptr);
        NUMTHREAD = 40;
        Degree = new unsigned int [num_nodes];
        for(int i = 0; i< num_nodes; i++){
            Degree[i] = indptr_ptr[i+1] - indptr_ptr[i];
        }   
        // row_idx = py::array_t<int>(num_nodes * 32);
        // col_idx= py::array_t<int>(num_nodes * 32);
        // ppr_value=py::array_t<double>(num_nodes * 32);
        ppr_walk = vector<vector<int>>(num_nodes);
        R = Random(unsigned(seed_));
        omp_set_num_threads(NUMTHREAD);
    }
    
    static bool cmp(std::pair<int, double> a, std::pair<int, double> b){
        return a.second > b.second;
    }
    inline int rw_step(const int & inode){ 
        int i = R.generateRandom() % Degree[inode];
        return indices_ptr[indptr_ptr[inode] + i];
    } 
    
    void random_walk_omp(const double & alpha, const int & walk_num){
        #pragma omp parallel for schedule(dynamic)
        for(int it = 0; it < num_nodes; it++){
            for(int iw =0; iw < walk_num; iw++){
                if(Degree[it]>0){
                    int cur_node = rw_step(it);
                    int is = 0;
		    for(; (is<prune_step) && (R.drand() > alpha); is++){
                        if(Degree[cur_node] == 0){
                            cur_node = it;
                        }
                        else{
                            cur_node = rw_step(cur_node);
                        }
                    }
                    if (is >= prune_step)
                        ppr_walk[it].push_back(cur_node);
                    else
                        ppr_walk[it].push_back(it);
                }
                else{
                    ppr_walk[it].push_back(it);
                }
            }
        }
    }
    
    inline int random_walk(const double & alpha, const int & node_id){
        
        if(Degree[node_id]>0){
            int cur_node = rw_step(node_id);
            int is = 0;
	    for(; (is<prune_step) && (R.drand() > alpha); is++){
                if(Degree[cur_node] == 0){
                    cur_node = node_id;
                }
                else{
                    cur_node = rw_step(cur_node);
                }
            }
            if(is >= prune_step)
                return node_id;
            else
                return cur_node;
        }
        else{
            return node_id;
        }
    }

    void forward_ppr_omp_map(py::array_t<int>& node_idx, py::array_t<int>& row_idx, py::array_t<int> & col_idx, py::array_t<double> & ppr_value, const double & rmax, const double & alpha, const int & K, const int & walk_num){
        struct timeval t_start,t_end, t_1, t_2, t_3; 
        double timeCost;
        gettimeofday(&t_start, NULL); 
        

        py::buffer_info node_idx_bf = node_idx.request();
        int node_idx_size = node_idx_bf.size;
        int* node_idx_ptr = static_cast<int *>(node_idx_bf.ptr);

        py::buffer_info row_idx_bf = row_idx.request();
        py::buffer_info col_idx_bf = col_idx.request();
        py::buffer_info ppr_value_bf = ppr_value.request();

        int* row_idx_ptr = static_cast<int*> (row_idx_bf.ptr);
        int* col_idx_ptr = static_cast<int*> (col_idx_bf.ptr);
        double* ppr_value_ptr = static_cast<double *>(ppr_value_bf.ptr);
        memset(row_idx_ptr, 0, node_idx_size * K * sizeof(int));
        memset(col_idx_ptr, 0, node_idx_size * K * sizeof(int));
        memset(ppr_value_ptr, 0, node_idx_size * K * sizeof(double));

        double alpha_rmax = alpha * rmax;
        // cout<< num_nodes * node_idx_size<< endl;
        vector<unordered_map<int, double>> residue_value_all(NUMTHREAD,unordered_map<int, double>());
        vector<unordered_map<int, double>> reserve_value_all(NUMTHREAD,unordered_map<int, double>());
        vector<unordered_set<int>> cand_set_all(NUMTHREAD, unordered_set<int>());
        
        if(walk_num > 0) 
            random_walk_omp(alpha, walk_num);
        
        #pragma omp parallel for schedule(dynamic)
        for(int it = 0; it < node_idx_size ; it++ ){
            int thread_num =  omp_get_thread_num();
            unordered_map<int, double>& residue_value = residue_value_all[thread_num];
            unordered_map<int, double>& reserve_value = reserve_value_all[thread_num];

            unordered_set<int> & cand_set = cand_set_all[thread_num]; //= Node_Set(num_nodes);//cand_sets[thread_num];
            
            int node_id = node_idx_ptr[it]; //w = random_w[it]
            double r_sum = 1.0;
            cand_set.insert(node_id);
            residue_value[node_id] = 1.0;
            reserve_value[node_id] = 0.;
            
            while(!cand_set.empty()){
                auto iter = cand_set.begin();
                int oldNode = *iter;
                cand_set.erase(iter);
                double r = residue_value[oldNode];
                reserve_value[oldNode] += r * alpha;
                residue_value[oldNode] = 0.;
                r_sum -= r * alpha;
                if(Degree[oldNode]==0){
                    residue_value[node_id] += r * (1 - alpha);
                    if(Degree[node_id]>0 && residue_value[node_id] >= rmax *Degree[node_id]){
                        cand_set.insert(node_id);
                    }
                }
                else{
                    double val_ = (1. - alpha) * r/ Degree[oldNode];
                    for(int i = indptr_ptr[oldNode]; i<indptr_ptr[oldNode+1]; i++){
                        int newNode = indices_ptr[i];
                        residue_value[newNode] += val_;
                        double res_val = residue_value[newNode];
                        if(res_val >= rmax * Degree[newNode]){
                            cand_set.insert(newNode);
                        }
                    }
                }
            }
            for(auto & reserve_i: reserve_value){
                reserve_i.second += alpha * residue_value[reserve_i.first];
            }
            if(walk_num > 0){
                for(auto & residue_i: residue_value){
                    double res_iv = residue_i.second * (1 - alpha);
                    unsigned long w_i = ceil(residue_i.second * walk_num/ r_sum);
                    double a_i = res_iv/double(w_i);
                    for(int rw_i = 0; rw_i < w_i; rw_i ++){
                        int des = ppr_walk[residue_i.first][rw_i];
                        reserve_value[des] += a_i;
                    }
                }
            }
            // reserve_value[node_id] += alpha * residue_value[node_id];
            // residue_value.erase(node_id);
            
            // rw +
            
            vector<pair<int, double>> ppr_res(reserve_value.begin(), reserve_value.end());
            
            int k = ppr_res.size()> K ? K : ppr_res.size();
        
            std::nth_element(ppr_res.begin(), ppr_res.begin() + k - 1, ppr_res.end(), cmp);

            for(int i =0; i< k; i++){
                int col_id = ppr_res[i].first;
                double ppr_v = ppr_res[i].second;
                int idx = it * K + i;
                row_idx_ptr[idx] = node_id;
                col_idx_ptr[idx] = col_id;
                ppr_value_ptr[idx] = ppr_v;
            }
            residue_value.clear();
            reserve_value.clear();
            cand_set.clear();
        }        

        gettimeofday(&t_end, NULL); 
        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        // cout<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    }

    void forward_rw_omp(py::array_t<int>& node_idx, py::array_t<int>& row_idx, py::array_t<int> & col_idx, py::array_t<double> & value, py::array_t<double> & coef, const double & rmax, const int & K){
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
        // memset(row_idx_ptr, 0, node_idx_size * K * sizeof(int));
        // memset(col_idx_ptr, 0, node_idx_size * K * sizeof(int));
        // double alpha_rmax = alpha * rmax;
        // cout<< num_nodes * node_idx_size<< endl;
        int order = coef_bf.size;
        // vector<unordered_map<int, double>> residue_value_all(NUMTHREAD,unordered_map<int, double>());
        // vector<unordered_map<int, double>> reserve_value_all(NUMTHREAD,unordered_map<int, double>());
        // vector<unordered_set<int>> cand_set_all(NUMTHREAD, unordered_set<int>());
        
        // if(walk_num > 0) 
            // random_walk_omp(alpha, walk_num);
        
        #pragma omp parallel for schedule(dynamic)
        for(int it = 0; it < node_idx_size ; it++ ){
            int thread_num =  omp_get_thread_num();
            unordered_map<int, double> residue_value = unordered_map<int, double>();//residue_value_all[thread_num];
            unordered_map<int, double> reserve_value = unordered_map<int, double>();//reserve_value_all[thread_num];

            int node_id = node_idx_ptr[it]; //w = random_w[it]
            double r_sum = 1.0;
            // cand_set.insert(node_id);
            residue_value[node_id] = 1.0;
            reserve_value[node_id] = 0.;
            for(int i = 0; i < order; i++){
                unordered_map<int, double> & residue_value_tmp = residue_value;
                residue_value = unordered_map<int, double>();
                while(!residue_value_tmp.empty()){
                    auto iter = residue_value.begin();
                    int oldNode = iter->first;
                    residue_value_tmp.erase(oldNode);
                    double r = iter->second;
                    reserve_value[oldNode] += coef_ptr[i] * r;
                    if (Degree[oldNode] == 0){
                        residue_value[node_id] += r;
                    }
                    else if(r >= rmax * Degree[oldNode]){
                        double val_ = r / Degree[oldNode];
                        for(int j = indptr_ptr[oldNode]; j< indptr_ptr[oldNode + 1]; j++){
                            int newNode = indices_ptr[j];
                            residue_value[newNode] += val_;
                        }
                    }
                }
            }
            
            vector<pair<int, double>> res(reserve_value.begin(), reserve_value.end());
            
            int k = res.size()> K ? K : res.size();
        
            std::nth_element(res.begin(), res.begin() + k - 1, res.end(), cmp);

            for(int i =0; i< k; i++){
                int col_id = res[i].first;
                double v_ = res[i].second;
                int idx = it * K + i;
                row_idx_ptr[idx] = node_id;
                col_idx_ptr[idx] = col_id;
                value_ptr[idx] = v_;
            }
        }        

        gettimeofday(&t_end, NULL); 
        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        // cout<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    }
    
    void forward_ppr_omp_map_w(py::array_t<int>& node_idx, py::array_t<int>& row_idx, py::array_t<int> & col_idx, py::array_t<double> & ppr_value, const double & rmax, const double & alpha, const int & K, const int & walk_num){
        struct timeval t_start,t_end, t_1, t_2, t_3; 
        double timeCost;
        gettimeofday(&t_start, NULL); 
        

        py::buffer_info node_idx_bf = node_idx.request();
        int node_idx_size = node_idx_bf.size;
        int* node_idx_ptr = static_cast<int *>(node_idx_bf.ptr);

        py::buffer_info row_idx_bf = row_idx.request();
        py::buffer_info col_idx_bf = col_idx.request();
        py::buffer_info ppr_value_bf = ppr_value.request();

        int* row_idx_ptr = static_cast<int*> (row_idx_bf.ptr);
        int* col_idx_ptr = static_cast<int*> (col_idx_bf.ptr);
        double* ppr_value_ptr = static_cast<double *>(ppr_value_bf.ptr);
        memset(row_idx_ptr, 0, node_idx_size * K * sizeof(int));
        memset(col_idx_ptr, 0, node_idx_size * K * sizeof(int));
        memset(ppr_value_ptr, 0, node_idx_size * K * sizeof(double));

        double alpha_rmax = alpha * rmax;
        // cout<< num_nodes * node_idx_size<< endl;
        vector<unordered_map<int, double>> residue_value_all(NUMTHREAD,unordered_map<int, double>());
        vector<unordered_map<int, double>> reserve_value_all(NUMTHREAD,unordered_map<int, double>());
        vector<unordered_set<int>> cand_set_all(NUMTHREAD, unordered_set<int>());
        
        // if(walk_num > 0) 
            // random_walk_omp(alpha, walk_num);
        
        #pragma omp parallel for schedule(dynamic)
        for(int it = 0; it < node_idx_size ; it++ ){
            int thread_num =  omp_get_thread_num();
            unordered_map<int, double>& residue_value = residue_value_all[thread_num];
            unordered_map<int, double>& reserve_value = reserve_value_all[thread_num];

            unordered_set<int> & cand_set = cand_set_all[thread_num]; //= Node_Set(num_nodes);//cand_sets[thread_num];
            
            int node_id = node_idx_ptr[it]; //w = random_w[it]
            double r_sum = 1.0;
            cand_set.insert(node_id);
            residue_value[node_id] = 1.0;
            reserve_value[node_id] = 0.;
            
            while(!cand_set.empty()){
                auto iter = cand_set.begin();
                int oldNode = *iter;
                cand_set.erase(iter);
                double r = residue_value[oldNode];
                reserve_value[oldNode] += r * alpha;
                residue_value[oldNode] = 0.;
                r_sum -= r * alpha;
                if(Degree[oldNode]==0){
                    residue_value[node_id] += r * (1 - alpha);
                    if(Degree[node_id]>0 && residue_value[node_id] >= rmax *Degree[node_id]){
                        cand_set.insert(node_id);
                    }
                }
                else{
                    double val_ = (1. - alpha) * r/ Degree[oldNode];
                    for(int i = indptr_ptr[oldNode]; i<indptr_ptr[oldNode+1]; i++){
                        int newNode = indices_ptr[i];
                        residue_value[newNode] += val_;
                        double res_val = residue_value[newNode];
                        if(res_val >= rmax * Degree[newNode]){
                            cand_set.insert(newNode);
                        }
                    }
                }
            }
            for(auto & reserve_i: reserve_value){
                reserve_i.second += alpha * residue_value[reserve_i.first];
            }
            if(walk_num > 0){
                for(auto & residue_i: residue_value){
                    double res_iv = residue_i.second * (1 - alpha);
                    unsigned long w_i = ceil(residue_i.second * walk_num/ r_sum);
                    double a_i = res_iv/double(w_i);
                    for(int rw_i = 0; rw_i < w_i; rw_i ++){
                        int des = random_walk(alpha, residue_i.first);
                        reserve_value[des] += a_i;
                    }
                }
            }
            // reserve_value[node_id] += alpha * residue_value[node_id];
            // residue_value.erase(node_id);
            
            // rw +
            
            vector<pair<int, double>> ppr_res(reserve_value.begin(), reserve_value.end());
            
            int k = ppr_res.size()> K ? K : ppr_res.size();
        
            std::nth_element(ppr_res.begin(), ppr_res.begin() + k - 1, ppr_res.end(), cmp);

            for(int i =0; i< k; i++){
                int col_id = ppr_res[i].first;
                double ppr_v = ppr_res[i].second;
                int idx = it * K + i;
                row_idx_ptr[idx] = node_id;
                col_idx_ptr[idx] = col_id;
                ppr_value_ptr[idx] = ppr_v;
            }
            residue_value.clear();
            reserve_value.clear();
            cand_set.clear();
        }        

        gettimeofday(&t_end, NULL); 
        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        // cout<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    }



    void forward_ppr_omp(py::array_t<int>& node_idx, py::array_t<int>& row_idx, py::array_t<int> & col_idx, py::array_t<double> & ppr_value, const double & rmax, const double & alpha, const int & K, const int & walk_num){
        struct timeval t_start,t_end, t_1, t_2, t_3; 
        double timeCost;
        gettimeofday(&t_start, NULL); 
        // Node_Set* residue_set;
        // Node_Set* reserve_set;
        // double * residue_value;
        // double * reserve_value;
        

        py::buffer_info node_idx_bf = node_idx.request();
        int node_idx_size = node_idx_bf.size;
        int* node_idx_ptr = static_cast<int *>(node_idx_bf.ptr);
        // row_idx = new py::array_t<int>(node_idx_size * K);
        // col_idx = new py::array_t<int>(node_idx_size * K);
        // ppr_value = new py::array_t<double>(node_idx_size * K);

        py::buffer_info row_idx_bf = row_idx.request();
        py::buffer_info col_idx_bf = col_idx.request();
        py::buffer_info ppr_value_bf = ppr_value.request();

        int* row_idx_ptr = static_cast<int*> (row_idx_bf.ptr);
        int* col_idx_ptr = static_cast<int*> (col_idx_bf.ptr);
        double* ppr_value_ptr = static_cast<double *>(ppr_value_bf.ptr);
        memset(row_idx_ptr, 0, node_idx_size * K * sizeof(int));
        memset(col_idx_ptr, 0, node_idx_size * K * sizeof(int));
        memset(ppr_value_ptr, 0, node_idx_size * K * sizeof(double));
        // cout << *std::min_element(col_idx_ptr, col_idx_ptr + node_idx_size * K) << endl;

        omp_set_num_threads(NUMTHREAD);
        double alpha_rmax = alpha * rmax;
        cout<< num_nodes * node_idx_size<< endl;
        vector<vector<double>> residue_value_all(NUMTHREAD,vector<double>(num_nodes, 0.));
        vector<vector<double>> reserve_value_all(NUMTHREAD,vector<double>(num_nodes, 0.));
        #pragma omp parallel for schedule(dynamic)
        for(int it = 0; it < node_idx_size ; it++ ){
            Node_Set residue_set(num_nodes);
            Node_Set reserve_set(num_nodes);
            // std::array<double, num_nodes>  residue_value = {0.} 
            // std::array<double, num_nodes>  reserve_value = {0.} 
            int thread_num =  omp_get_thread_num();
            vector<double> & residue_value = residue_value_all[thread_num];//new double[num_nodes]{0.};
            vector<double> & reserve_value = reserve_value_all[thread_num];//new double[num_nodes]{0.};
            // double * ppr_res_value = new double[num_nodes]{0.};
            // int* ppr_res_idx = new int[num_nodes]{0};
            int node_id = node_idx_ptr[it]; //w = random_w[it];
            // cout << node_id << endl;
            residue_set.Push(node_id);
            residue_value[node_id] = alpha;
            while(residue_set.KeyNumber > 0){
                int oldNode = residue_set.Pop();
                double r = residue_value[oldNode];
                reserve_set.Push(oldNode);
                reserve_value[oldNode] += r;
                residue_value[oldNode] = 0.;
                
                for(int i = indptr_ptr[oldNode]; i<indptr_ptr[oldNode+1]; i++){
                    int newNode = indices_ptr[i];
                    residue_value[newNode]+= (1. - alpha) * r/Degree[oldNode];
                    if(residue_value[newNode] >= alpha_rmax * Degree[newNode]){
                        residue_set.Push(newNode);
                    }
                }
            }
            // cout<< "123456"<<endl;
            gettimeofday(&t_1, NULL); 
            timeCost = t_1.tv_sec - t_start.tv_sec + (t_1.tv_usec - t_start.tv_usec)/1000000.0;
            // cout<<" fwd pussh cost: "<<timeCost<<" s"<<endl;
            
            std::vector<std::pair<int, double>> ppr_res;
            int res_ppr_size = reserve_set.KeyNumber;
            ppr_res.reserve(res_ppr_size);
            while(reserve_set.KeyNumber > 0){
                int node_id = reserve_set.Pop();
                double ppr_v = reserve_value[node_id] + alpha * residue_value[node_id];
                ppr_res.emplace_back(std::pair<int,double>(node_id, ppr_v));
            }
            int k = ppr_res.size()> K ? K : ppr_res.size();
            // cout << it << k << endl;
            gettimeofday(&t_2, NULL); 
            timeCost = t_2.tv_sec - t_1.tv_sec + (t_2.tv_usec - t_1.tv_usec)/1000000.0;
            // cout<<" copy cost: "<<timeCost<<" s"<<endl;
        
            
            std::nth_element(ppr_res.begin(), ppr_res.begin() + k - 1, ppr_res.end(), cmp);

            gettimeofday(&t_3, NULL); 
            timeCost = t_3.tv_sec - t_2.tv_sec + (t_3.tv_usec - t_2.tv_usec)/1000000.0;
            // cout<<" find top n cost: "<<timeCost<<" s"<<endl;

            for(int i =0; i< k; i++){
                int col_id = ppr_res[i].first;
                double ppr_v = ppr_res[i].second;
                int idx = it * K + i;
                row_idx_ptr[idx] = node_id;
                col_idx_ptr[idx] = col_id;
                ppr_value_ptr[idx] = ppr_v;
            }
            vector<double>(num_nodes, 0.).swap(reserve_value);
            vector<double>(num_nodes, 0.).swap(residue_value);
            // delete [] residue_value;s
            // delete [] reserve_value;
        }
        // delete [] residue_value;
        // delete [] reserve_value;
        // cout << *std::min_element(col_idx_ptr, col_idx_ptr + node_idx_size * K) << endl;
        gettimeofday(&t_end, NULL); 
        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        cout<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    }

    // void BackwardPush_ppr(int start, int end){
    //     Node_Set* candidate_set1 = new Node_Set(num_nodes);
    //     Node_Set* candidate_set2 = new Node_Set(num_nodes);
    //     for(int it = start ; it < end ; it++ ){
    //         int w = random_w[it];
    //         float rowsum_pos = positive_sum[w];
    //         float rowsum_neg = negative_sum[w];

    //         for(int ik = 0 ; ik < num_nodes; ik++ ){
    //             int idx = ik + w * num_nodes;
    //             if( positive_feat[idx] > rmax*rowsum_pos ){
    //                 candidate_set1->Push(ik);
    //             }
    //             if( negative_feat[idx] > rmax*rowsum_neg ){
    //                 candidate_set2->Push(ik);
    //             }
    //         }

    //         while(candidate_set1->KeyNumber !=0){
    //             int oldNode = candidate_set1->Pop();
    //             int idx = oldNode + w * num_nodes;
    //             float oldresidue = positive_feat[idx];
    //             float rpush = (1-alpha)*oldresidue;
    //             reserve_ptr[idx]+=alpha*oldresidue;
    //             positive_feat[idx] = 0;
    //             for(int n_i = indptr_ptr[oldNode]; n_i < indptr_ptr[oldNode+1]; n_i++){
    //                 int newNode = indices_ptr[n_i];
    //                 int new_idx = newNode + w * num_nodes;
    //                 positive_feat[new_idx]+=rpush/Degree[newNode];
    //                 if(positive_feat[new_idx] > rmax*rowsum_pos){
    //                     candidate_set1->Push(newNode);
    //                 }
    //             }
    //         }
    //         while(candidate_set2->KeyNumber !=0){
    //             int oldNode = candidate_set2->Pop();
    //             int idx = oldNode + w * num_nodes;
    //             float oldresidue = negative_feat[idx];
    //             float rpush = (1-alpha)*oldresidue;
    //             reserve_ptr[oldNode]-=alpha*oldresidue;
    //             negative_feat[idx] = 0;
    //             for(int n_i = indptr_ptr[oldNode]; n_i < indptr_ptr[oldNode+1]; n_i++){
    //                 int newNode = indices_ptr[n_i];
    //                 int new_idx = newNode + w * num_nodes;
    //                 negative_feat[new_idx] +=rpush/Degree[newNode];
    //                 if(negative_feat[new_idx] > rmax*rowsum_neg){
    //                     candidate_set2->Push(newNode);
    //                 }
    //             }
    //         }
    //         candidate_set1->Clean();
    //         candidate_set2->Clean();
    //     }
    // }
    // void random_walk(py::array_t<int> nodes){
    //     struct timeval t_start,t_end; 
    //     float timeCost;
    //     gettimeofday(&t_start, NULL); 

    //     py::buffer_info nodes_bf = nodes.request();
    //     int* nodes_ptr = static_cast<int *>(nodes_bf.ptr);
    //     int nodes_num = nodes_bf.size;
    //     rw_res = new py::array_t<int>(nodes_num * walk_num);
    //     py::buffer_info rw_res_bf = rw_res->request();
    //     int * rw_res_ptr = static_cast<int *>(rw_res_bf.ptr);
    //     #pragma omp parallel for shared(nodes_ptr, rw_res)
    //     for (int i = 0; i < nodes_num; i++) {
    //         int n_id = nodes_ptr[i];
    //         for(int j = 0; j < walk_num; j++){
    //             int cur_id = n_id;
    //             while(true){
    //                 float ber_s = rand_0_1();
    //                 if(ber_s < alpha)
    //                     break;
    //                 else{
    //                     if(Degree[cur_id])
    //                         cur_id = indices_ptr[indptr_ptr[cur_id] + rand_max()%int(Degree[cur_id])];
    //                     else
    //                         cur_id = n_id;
    //                 }
    //             }
    //             rw_res_ptr[i * walk_num + j] = cur_id;
    //         }
    //     }
    //     gettimeofday(&t_end, NULL); 
    //     timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
    //     cout<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    // }

    // py::array_t<float>* augmentation(py::array_t<int> nodes, int walk_num_batch){
    //     struct timeval t_start,t_end; 
    //     float timeCost;
    //     gettimeofday(&t_start, NULL); 

    //     py::buffer_info nodes_bf = nodes.request();
    //     int* nodes_ptr = static_cast<int *>(nodes_bf.ptr);
    //     int batch_num = nodes_bf.size;
    //     rw_res = new py::array_t<int>(batch_num * walk_num_batch);
    //     py::buffer_info rw_res_bf = rw_res->request();
    //     int * rw_res_ptr = static_cast<int *>(rw_res_bf.ptr);

    //     py::array_t<float>* res_arr = new py::array_t<float>(batch_num * feat_dim);
        
    //     py::buffer_info res_arr_bf = res_arr->request();
    //     // cout<< res_arr_bf.size << endl;
    //     float* res_arr_ptr = static_cast<float *>(res_arr_bf.ptr);

    //     #pragma omp parallel for shared(nodes_ptr, rw_res, res_arr_ptr)
    //     for (int i = 0; i < batch_num; i++) {
    //         int n_id = nodes_ptr[i];
    //         for(int k=0; k < feat_dim; k++){
    //             res_arr_ptr[i * feat_dim + k] = reserve_ptr[k * num_nodes + n_id];
    //         }
    //         for(int j = 0; j < walk_num_batch; j++){
    //             int cur_id = n_id;
    //             while(true){
    //                 float ber_s = rand_0_1();
    //                 if(ber_s < alpha)
    //                     break;
    //                 else{
    //                     if(Degree[cur_id])
    //                         cur_id = indices_ptr[indptr_ptr[cur_id] + rand_max()%int(Degree[cur_id])];
    //                     else
    //                         cur_id = n_id;
    //                     }
    //             }
    //             for(int k=0; k < feat_dim; k++){
    //                 res_arr_ptr[i * feat_dim + k] += residue_ptr[k * num_nodes + cur_id]/ float(walk_num_batch);
    //             }
    //             //rw_res_ptr[i * walk_num + j] = cur_id;
    //         }
    //     }
    //     gettimeofday(&t_end, NULL); 
    //     timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
    //     // cout<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    //     return res_arr;
    // }
    // void ppr_push(){
    //     struct timeval t_start,t_end; 
    //     float timeCost;
    //     gettimeofday(&t_start, NULL); 
    //     vector<thread> threads;
    //     int ti;
    //     int start;
    //     int end = 0;
    //     for( ti=1 ; ti <= feat_dim%NUMTHREAD ; ti++ ){
    //         start = end;
    //         end+=ceil((float)feat_dim/NUMTHREAD);
    //         threads.push_back(thread(&Graph::BackwardPush_ppr,this, start, end));
    //     }
    //     for( ; ti<=NUMTHREAD ; ti++ ){
    //         start = end;
    //         end+=feat_dim/NUMTHREAD;
    //         threads.push_back(thread(&Graph::BackwardPush_ppr,this, start, end));
    //     }
    //     for (int t = 0; t < NUMTHREAD ; t++){
    //         threads[t].join();
    //     }
    //     vector<thread>().swap(threads);
    //     gettimeofday(&t_end, NULL); 
    //     timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
    //     cout<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    // }


    double rand_0_1(){
        return rand_r(&seed)%RAND_MAX/(double)RAND_MAX;
    }
    int rand_max(){
        return rand_r(&seed);
    }

};

#endif
