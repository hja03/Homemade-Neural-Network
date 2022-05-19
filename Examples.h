#pragma once

class Example {
	double* inputs;
	double* outputs;

	public:
		// once example is constructed its vectors cannot be eddited
		Example(double ins[], double outs[]) {
			inputs = ins;
			outputs = outs;
		}

		// methods to return whole input or output vector
		double* x() {
			return inputs;
		}
		double* y() {
			return outputs;
		}

		// methods to return specific elements of input or output vector with index i
		double x(int i) {
			return inputs[i];
		}
		double y(int i) {
			return outputs[i];
		}
	
};