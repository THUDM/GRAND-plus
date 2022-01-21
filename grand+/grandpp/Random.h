#ifndef RANDOM_H
#define RANDOM_H

#include<random>
#include<algorithm>
#include"SFMT.h"


using namespace std;

class Random{
public:
	sfmt_t sfmt;
	unsigned seed_r;
	Random(){
		Random(unsigned(time(0)));
	}
	Random(unsigned seed){
		srand(unsigned(seed));
		seed_r = seed;
		sfmt_init_gen_rand(&sfmt, rand());
	}
	unsigned int generateRandom() {
		return sfmt_genrand_uint32(&sfmt);
	}
	unsigned int generateRandom_t() {
		return rand_r(&seed_r);
	}
	double drand(){
		return generateRandom() % RAND_MAX / (double) RAND_MAX;
	}
	double drand_t(){
		return generateRandom_t() % RAND_MAX / (double) RAND_MAX;
	}

};

#endif