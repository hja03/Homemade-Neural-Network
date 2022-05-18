#pragma once

class Example {
	double* inputs;
	double* outputs;

public:
	Example(double ins[], double outs[]) {
		inputs = ins;
		outputs = outs;
	}

	double x(int i) {
		return inputs[i];
	}
	double y(int i) {
		return outputs[i];
	}
	
};