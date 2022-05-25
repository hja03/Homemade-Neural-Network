#pragma once
#include <iostream>
#include "ActivationFunctions.h"

class Layer {
    double* inputs;
    double* nodes;
    double** weights;
    double* errors;

    int layerNumber;

public:
    int numNodes;
    int nextNumNodes;

    double (*activation)(double);
    double (*derivative)(double);
    static int layerCounter;

    // layer constructor
    Layer(int size) {
        // Set id number
        layerNumber = layerCounter;
        layerCounter++;

        // Init nodes
        inputs = new double[size];
        nodes = new double[size];
        weights = nullptr;
        errors = new double[size];

        numNodes = size;
        nextNumNodes = 0;

        // By default set activation to ReLU
        activation = &relu;
        derivative = &reluDerivative;
    }

    // sets connection to next layer in network and constructs weights matrix
    void setNextNumNodes(int nextLayerSize) {
        nextNumNodes = nextLayerSize;

        // Init weights going to next layer
        weights = new double* [numNodes];
        for (int i = 0; i < numNodes; i++) {
            weights[i] = new double[nextLayerSize];
        }
    }

    // debug print functions
    void printNodes() {
        std::cout << "Layer " << layerNumber << " nodes: ";
        for (int i = 0; i < numNodes; i++) {
            std::cout << nodes[i] << " ";
        }
        std::cout << std::endl;
    }

    void printInputs() {
        std::cout << "Layer " << layerNumber << " inputs: ";
        for (int i = 0; i < numNodes; i++) {
            std::cout << inputs[i] << " ";
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
        if (layerNumber == 1) {
            std::cout << "Layer " << layerNumber << " - Input" << std::endl;
        }
        else if (nextNumNodes == 0) {
            std::cout << "Layer " << layerNumber << " - Output" << std::endl;
        }
        else {
            std::cout << "Layer " << layerNumber << std::endl;
        }
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

    void setNode(int i, double value) {
        if (i < 0 || i >= numNodes) {
            std::cout << "Tried to set value for node with index out of range";
            std::exit(EXIT_FAILURE);
        }
        nodes[i] = value;
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

    // should probably return a copy of these arrays 
    double* getInputsArray() {
        return inputs;
    }

    double* getNodesArray() {
        return nodes;
    }

    // again this is pretty sloppy code practice
    void setNodesArray(double* ary) {
        nodes = ary;
    }
};

// sets static member layer counter to start at 1 
int Layer::layerCounter = 1;