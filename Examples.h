#pragma once
#include <vector>

class Example {
	double* inputs;
	double* outputs;
	
	int inSize;
	int outSize;


	public:
		// once example is constructed its vectors cannot be eddited
		Example(double ins[], double outs[], int inputSize, int outputSize) {
			inputs = ins;
			outputs = outs;

			inSize = inputSize;
			outSize = outputSize;
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

		void print() {
			std::cout << "Input: ";
			for (int i = 0; i < inSize; i++) {
				std::cout << inputs[i] << " ";
			}
			std::cout << std::endl;
			std::cout << "Output: ";
			for (int i = 0; i < outSize; i++) {
				std::cout << outputs[i] << " ";
			}
			std::cout << std::endl;
		}
};

// turns a (output, input) set of vectors into a vector of examples ready to be used
// specifically for binary labels
std::vector<Example> csvToExamples(std::vector<std::vector<double>> vectors, int outputSize) {
	std::vector<Example> output =  std::vector<Example>();

	for (std::vector<double> v : vectors) {
		double *inp = new double[vectors[0].size() - outputSize];
		double* out = new double[outputSize];

		for (int i = 0; i < v.size(); i++) {
			if (i == 0) {
				if (v[i] == 1) {
					out[0] = 1;
					out[1] = 0;
				}
				else {
					out[0] = 0;
					out[1] = 1;
				}
			}
			else {
				inp[i - 1] = v[i];
			}
		}
		output.push_back(Example(inp, out, vectors[0].size() - outputSize, outputSize));
	}
	return output;
}