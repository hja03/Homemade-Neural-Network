#pragma once
#include <algorithm>

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