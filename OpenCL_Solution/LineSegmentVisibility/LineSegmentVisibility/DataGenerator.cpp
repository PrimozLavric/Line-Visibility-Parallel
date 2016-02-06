#include "DataGenerator.h"

//32 bit random number generator
unsigned int rand32(){
	unsigned int r = (rand() & 0x7fff) | ((rand() & 0x7fff) << 15) | ((rand() & 0x3) << 30);
	return r;
}

//Returns the line data structure filled with x and y coordinates of lines
//N - number of lines
//p - proportion of visible lines in the data
//maxAngle - maximum LOS angle to the line top
lines generateData(long long N, float p, float maxAngle){


	//Allocate memory for the data
	lines data;
	data.y = (float *)malloc(N * sizeof(float));
	data.x = (float *)malloc(N * sizeof(float));

	//Compute the maximum LOS coefficient
	float maxk = tan(maxAngle / 180 * PI);

	//Compute the approximate number of visible line sections
	float num_visible = round(N*p);

	//coefficient step
	float kstep = maxk / num_visible;

	//min x-coordinate step
	float xstep = 1000.0 / N;

	//generate ascending x-coordinates of lines
	for (long long i = 0; i < N; i++){
		data.x[i] = xstep + i*xstep;
	}


	//generate heights of lines according to the probability of a visible line p
	float k = kstep;
	for (long long i = 0; i < N; i += 1){
		if ((1.0*rand32() / UINT_MAX) < p){
			//section of a line visible
			data.y[i] = data.x[i] * k;
			k = k + kstep;
		}
		else
			//line not visible
			data.y[i] = data.x[i] * ((1.0*rand32() / UINT_MAX)*(k - kstep));
	}

	return data;

}