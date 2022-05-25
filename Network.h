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
            if (layers.size() > 0) {
                layers.back().setNextNumNodes(l.numNodes);
            }
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

        int predict(double* input) {
            // load input vector to first layers nodes
            Layer inputLayer = getLayer(0);
            for (int i = 0; i < inputLayer.numNodes; i++) {
                inputLayer.setNode(i, input[i]);
                inputLayer.setInput(i, input[i]);
            }

            // compute hidden layers
            for (int l = 1; l < numLayers - 1; l++) {
                Layer layer = getLayer(l);
                Layer layerBefore = getLayer(l - 1);
                for (int j = 0; j < layer.numNodes; j++) {
                    double in_j = 0;
                    for (int i = 0; i < layerBefore.numNodes; i++) {
                        in_j += layerBefore.getNode(i) * layerBefore.getWeight(i, j);
                    }
                    layer.setInput(j, in_j);
                    layer.setNode(j, layer.activation(in_j));
                }
            }

            Layer outputLayer = getLayer(numLayers - 1);
            Layer outputLayerBefore = getLayer(numLayers - 2);
            for (int j = 0; j < outputLayer.numNodes; j++) {
                double in_j = 0;
                for (int i = 0; i < outputLayerBefore.numNodes; i++) {
                    in_j += outputLayerBefore.getNode(i) * outputLayerBefore.getWeight(i, j);
                }
                outputLayer.setInput(j, in_j);

            }

            // softmax activaion
            outputLayer.setNodesArray(arraySoftMax(outputLayer.getInputsArray(), outputLayer.numNodes));

            int maxIndex = std::distance(outputLayer.getNodesArray(),
                std::max_element(outputLayer.getNodesArray(), outputLayer.getNodesArray() + 2));
            return maxIndex;
        }

        double accuracyTest(std::vector<Example> examples) {
            int correct = 0;
            for (Example ex : examples) {
                int actualIndex = 0;
                if (ex.y(0) == 1) {
                    actualIndex = 0;
                }
                else {
                    actualIndex = 1;
                }


                if (predict(ex.x()) == actualIndex) {
                    correct += 1;
                    
                }

            }

            std::cout << "Number correct: " << correct;
            double accuracy = ((double)correct / examples.size()) * 100;
            std::cout << "\nAccuracy: " << accuracy << "%\n";
            return accuracy;
        }

};