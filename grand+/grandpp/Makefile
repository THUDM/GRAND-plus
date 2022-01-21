all:
	g++ -g -O3 -Wall -fopenmp -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I SFMT-src-1.4/ -o propagation`python3-config --extension-suffix` propagation.cpp SFMT.c
clean:
	rm -rf *.so
