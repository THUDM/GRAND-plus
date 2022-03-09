all:
	g++ -g -O3 -Wall -fopenmp -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I precompute/SFMT-src-1.4/ -o precompute/propagation`python3-config --extension-suffix` precompute/propagation.cpp
clean:
	rm -rf precompute/*.so
