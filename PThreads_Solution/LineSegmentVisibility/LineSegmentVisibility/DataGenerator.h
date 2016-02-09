#ifndef DATA_GENERATOR_HPP
#define DATA_GENERATOR_HPP

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>

#define PI 3.14159265358979323846


typedef struct{
	float *x;
	float *y;
}lines;

lines generateData(long long N, float p, float maxAngle);

#endif