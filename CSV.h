#pragma once
#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <stdlib.h>

// reads a .csv file into a 2d vector
std::vector<std::vector<double>> readCSV(std::string fileName) {
	std::ifstream file(fileName);
	std::vector<std::vector<double>> output;

	std::string line;
	while (std::getline(file, line)) {
		std::vector<double> vector;
		std::stringstream ss(line);

		for (double i; ss >> i;) {
			vector.push_back(i);
			if (ss.peek() == ',') {
				ss.ignore();
			}
		}

		output.push_back(vector);
	}

	file.close();
	return output;
}