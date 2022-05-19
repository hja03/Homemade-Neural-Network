#pragma once
#include <iostream>
#include <vector>
#include "Layer.h"

class Network {
    std::vector<Layer> layers;

public:
    int numLayers = 0;
    double learningRate;

    Network(double learnRate) {
        learningRate = learnRate;
    }

    void addLayer(Layer l) {
        layers.push_back(l);
        numLayers++;
    }

    void printSummary() {
        std::cout << "--- Network Summary ---\n";
        for (Layer l : layers) {
            l.printSummary();
        }
    }

    std::vector<Layer> getLayers() {
        return layers;
    }

    Layer getLayer(int l) {
        return layers.at(l);
    }

};