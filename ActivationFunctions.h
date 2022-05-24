#pragma once
#include <algorithm>
#include <math.h>

double relu(double input) {
	return std::max(0.0, input);
}
double reluDerivative(double input) {
	if (input >= 0) { return 1.0; }
	else { return 0.0; }
}

double identity(double input) {
	return input;
}
double identityDerivative(double input) {
	return 1.0;
}


double softMaxDerivative(double z) {
	return z * (1 - z);
}

// need to write a function which takes in a vector and returns the corrected softmax vector
double* arraySoftMax(double* ary, int length) {
	double* output = new double[length];
	for (int i = 0; i < length; i++) {
		output[i] = ary[i];
	}
	// find max
	double maxValue = 0;
	for (int i = 0; i < length; i++) {
		if (output[i] > maxValue) { maxValue = output[i]; }
	}

	// minus max value from every element and add to sum
	double sum = 0;
	for (int i = 0; i < length; i++) {
		output[i] -= maxValue;
		sum += exp(output[i]);
	}

	// compute softmax for each element
	for (int i = 0; i < length; i++) {
		output[i] = exp(output[i]) / sum;
	}
	return output;
}

