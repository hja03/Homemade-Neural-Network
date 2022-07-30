#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "Examples.h"
#include "ActivationFunctions.h"
#include "Layer.h"
#include "Network.h"
#include "CSV.h"


void back_prop_learning(Network network, std::vector<Example> examples, int trainingCycles) {
    if (trainingCycles < 1) {
        std::cout << "At least 1 training cycle must be set!\n";
        return;
    }

    std::cout << "--- Starting backpropagation learning! ---\n";

    // init all weights in network
    for (Layer l : network.getLayers()) {
        for (int i = 0; i < l.numNodes; i++) {
            for (int j = 0; j < l.nextNumNodes; j++) {
                l.setWeight(i, j, ((double)rand() / (RAND_MAX)));
            }
        }
    }

    for (int a = 0; a < trainingCycles; a++) {
        // for each input output pair 
        double totalCrossEntropyLoss = 0;
        for (Example example : examples) {
            // load input vector to first layers nodes
            Layer inputLayer = network.getLayer(0);
            for (int i = 0; i < inputLayer.numNodes; i++) {
                inputLayer.setNode(i, example.x(i));
                inputLayer.setInput(i, example.x(i));
            }


            // compute hidden layers
            for (int l = 1; l < network.numLayers - 1; l++) {
                Layer layer = network.getLayer(l);
                Layer layerBefore = network.getLayer(l - 1);
                for (int j = 0; j < layer.numNodes; j++) {
                    double in_j = 0;
                    for (int i = 0; i < layerBefore.numNodes; i++) {
                        in_j += layerBefore.getNode(i) * layerBefore.getWeight(i, j);
                    }
                    layer.setInput(j, in_j);
                    layer.setNode(j, layer.activation(in_j));
                }
            }

            Layer outputLayer = network.getLayer(network.numLayers - 1);
            Layer outputLayerBefore = network.getLayer(network.numLayers - 2);
            for (int j = 0; j < outputLayer.numNodes; j++) {
                double in_j = 0;
                for (int i = 0; i < outputLayerBefore.numNodes; i++) {
                    in_j += outputLayerBefore.getNode(i) * outputLayerBefore.getWeight(i, j);
                }
                outputLayer.setInput(j, in_j);

            }

            // softmax activaion
            outputLayer.setNodesArray(arraySoftMax(outputLayer.getInputsArray(), outputLayer.numNodes));

            // front propogation done at this point
            // calculate cross entropy loss
            double crossEntropyLoss = 0;
            for (int i = 0; i < outputLayer.numNodes; i++) {
                double nodeValue = outputLayer.getNode(i);
                if (nodeValue == 0) { nodeValue = 0.001; }
                crossEntropyLoss += example.y(i) * log(nodeValue);

            }
            totalCrossEntropyLoss += -crossEntropyLoss;;

            // time to backpropagate errors through the network

            // calculate error at output nodes from difference to expected output
            for (int j = 0; j < outputLayer.numNodes; j++) {
                outputLayer.setError(j, 
                    //softMaxDerivative(outputLayer.getNode(j)) * 
                    (example.y(j) - outputLayer.getNode(j)));
            }


            // propogate error through network
            for (int l = network.numLayers - 2; l >= 0; l--) {
                Layer layer = network.getLayer(l);
                Layer nextLayer = network.getLayer(l + 1);
                for (int i = 0; i < layer.numNodes; i++) {
                    double sum = 0;
                    for (int j = 0; j < nextLayer.numNodes; j++) {
                        sum += layer.getWeight(i, j) * nextLayer.getError(j);
                    }
                    layer.setError(i, layer.derivative(layer.getInput(i)) * sum);
                }
            }

            // update every weight based on errors
            for (int l = 0; l < network.numLayers - 1; l++) {
                Layer layer = network.getLayer(l);
                Layer nextLayer = network.getLayer(l + 1);
                for (int i = 0; i < layer.numNodes; i++) {
                    for (int j = 0; j < nextLayer.numNodes; j++) {
                        double weight = layer.getWeight(i, j);
                        weight += network.learningRate * layer.getNode(i) * nextLayer.getError(j);

                        layer.setWeight(i, j, weight);
                    }
                }
            }
        }
        totalCrossEntropyLoss /= examples.size();
        std::cout << "Avg X Entropy Loss: " << totalCrossEntropyLoss << std::endl;
    }
    std::cout << "Done training!\n";
}


int main() {
    std::cout << "--- Starting Program ---\n\n";

    const double LEARNING_RATE = 0.001;
    Network net = Network(LEARNING_RATE);
    net.addLayer(Layer(54));
    net.addLayer(Layer(30));
    net.addLayer(Layer(2));

    // loads spam filtering dataset to examples vector
    std::string fileName = "test.csv";
    std::string fileName = "train.csv";
    std::vector<std::vector<double>> csv = readCSV(fileName);
    std::vector<Example> training = csvToExamples(csv, 2);

    std::string fileName2 = "test.csv";
    std::vector<std::vector<double>> csv2 = readCSV(fileName2);
    std::vector<Example> testing = csvToExamples(csv2, 2);

    net.printSummary();
    const int EPOCHS = 500;
    back_prop_learning(net, training, EPOCHS);
 
    net.accuracyTest(testing);
}