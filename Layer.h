#pragma once
#include <iostream>
#include "ActivationFunctions.h"

class Layer {
    double** weights;
    double* inputs;
    double* nodes;
    double* errors;
    int layerNumber;

public:
    int numNodes;
    int nextNumNodes;

    double (*activation)(double);
    double (*derivative)(double);

    // Default layer (Where there exists a next layer)
    Layer(int num, int size, int nextLayerSize) {
        // Set id number
        layerNumber = num;
        // Init nodes
        nodes = new double[size];
        inputs = new double[size];
        numNodes = size;
        nextNumNodes = nextLayerSize;

        // Init weights going to next layer
        weights = new double* [size];
        for (int i = 0; i < size; i++) {
            weights[i] = new double[nextLayerSize];
        }

        errors = new double[size];

        // By default set activation to ReLU
        activation = &relu;
        derivative = &reluDerivative;
    }

    // End output layer
    Layer(int num, int size) {
        // Set id number
        layerNumber = num;
        // Init nodes
        nodes = new double[size];
        inputs = new double[size];
        numNodes = size;
        nextNumNodes = 0;

        // Set weights to be null
        weights = nullptr;

        errors = new double[size];

        // By default set activation to ReLU
        activation = &relu;
        derivative = &reluDerivative;
    }

    void printNodes() {
        std::cout << "Layer " << layerNumber << " nodes: ";
        for (int i = 0; i < numNodes; i++) {
            std::cout << nodes[i] << " ";
        }
        std::cout << std::endl;
    }

    void printErrors() {
        std::cout << "Layer " << layerNumber << " errors: ";
        for (int i = 0; i < numNodes; i++) {
            std::cout << errors[i] << " ";
        }
        std::cout << std::endl;
    }

    void printWeights() {
        if (nextNumNodes == 0) {
            std::cout << "Layer " << layerNumber << " weights: Output\n";
            return;
        }
        std::cout << "Layer " << layerNumber << " weights:\n";
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < nextNumNodes; j++) {
                std::cout << weights[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void printSummary() {
        std::cout << "Layer " << layerNumber << std::endl;
        std::cout << "# of nodes: " << numNodes << std::endl;
        std::cout << "Weight Dimensions: " << numNodes << ", " << nextNumNodes << std::endl;
        std::cout << "# of weights: " << numNodes * nextNumNodes << "\n\n";
    }

    // Setters and Getters
    void setWeight(int i, int j, double value) {
        if (i < 0 || j < 0) {
            std::cout << "Tried to access an element in an array below 0";
            std::exit(EXIT_FAILURE);
        }
        if (i >= numNodes || j >= nextNumNodes) {
            std::cout << "Tried to access an element in an array above max size";
            std::exit(EXIT_FAILURE);
        }
        weights[i][j] = value;
    }

    void setNode(int i, double value) {
        if (i < 0 || i >= numNodes) {
            std::cout << "Tried to set value for node with index out of range";
            std::exit(EXIT_FAILURE);
        }
        nodes[i] = value;
    }

    double getWeight(int i, int j) {
        if (i < 0 || j < 0) {
            std::cout << "Tried to access an element in an array below 0";
            std::exit(EXIT_FAILURE);
        }
        if (i >= numNodes || j >= nextNumNodes) {
            std::cout << "Tried to access an element in an array above max size";
            std::exit(EXIT_FAILURE);
        }
        return weights[i][j];
    }

    double getNode(int i) {
        if (i < 0 || i >= numNodes) {
            std::cout << "Tried to set value for node with index out of range";
            std::exit(EXIT_FAILURE);
        }
        return nodes[i];
    }

    void setError(int j, double value) {
        if (j < 0 || j >= numNodes) {
            std::cout << "Tried to set value for node with index out of range";
            std::exit(EXIT_FAILURE);
        }
        errors[j] = value;
    }

    double getError(int j) {
        if (j < 0 || j >= numNodes) {
            std::cout << "Tried to set value for node with index out of range";
            std::exit(EXIT_FAILURE);
        }
        return errors[j];
    }

    void setInput(int j, double value) {
        if (j < 0 || j >= numNodes) {
            std::cout << "Tried to set value for node with index out of range";
            std::exit(EXIT_FAILURE);
        }
        inputs[j] = value;
    }

    double getInput(int j) {
        if (j < 0 || j >= numNodes) {
            std::cout << "Tried to set value for node with index out of range";
            std::exit(EXIT_FAILURE);
        }
        return inputs[j];
    }

};