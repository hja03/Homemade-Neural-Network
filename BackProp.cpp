#include <iostream>
#include <vector>

#include "Examples.h"
#include "ActivationFunctions.h"
#include "Layer.h"
#include "Network.h"



void back_prop_learning(std::vector<Example> examples,  Network network) {
    // init all weights in network
    for (Layer l : network.getLayers()) {
        for (int i = 0; i < l.numNodes; i++) {
            for (int j = 0; j < l.nextNumNodes; j++) {
                l.setWeight(i, j, ((double)rand() / (RAND_MAX)));
            }
        }
        l.printWeights();
    }

    for (int a = 0; a < 10; a++) {
        // for each input output pair 
        for (Example example : examples) {
            // load input vector to first layers nodes
            Layer inputLayer = network.getLayer(0);
            for (int i = 0; i < inputLayer.numNodes; i++) {
                inputLayer.setNode(i, example.x(i));
                inputLayer.setInput(i, example.x(i));
            }
            inputLayer.printNodes();

            for (int l = 1; l < network.numLayers; l++) {
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
                layer.printNodes();
            }
            // front propogation done at this point
            // time to backpropagate errors through the network

            // calculate error at output nodes from difference to expected output
            Layer outputLayer = network.getLayer(network.numLayers - 1);
            for (int j = 0; j < outputLayer.numNodes; j++) {
                outputLayer.setError(j, outputLayer.derivative(outputLayer.getInput(j)) * (example.y(j) - outputLayer.getNode(j)));
            }
            outputLayer.printErrors();

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
                layer.printErrors();
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
                layer.printWeights();
            }
        }
    }
}

void testNetwork(std::vector<Example> examples, Network network) {
    double totalError = 0;
    for (Example example : examples) {
        // need to write network forward feed method



    }



}


int main() {
    std::cout << "--- Starting Program ---\n\n";

    Layer one = Layer(1, 2, 2);
    Layer two = Layer(2, 2);

    Network net = Network(0.8);
    net.addLayer(one);
    net.addLayer(two);

    std::vector<Example> examples;
    double test1in[] = { 1,0 };
    double test1out[] = { 1,0 };
    double test2in[] = { 0,1 };
    double test2out[] = { 0,1 };
    double test3in[] = { 1,1 };
    double test3out[] = { 0.5,0.5 };

    examples.push_back(Example(test1in, test1out));
    examples.push_back(Example(test2in, test2out));
    examples.push_back(Example(test3in, test3out));

    back_prop_learning(examples, net);
}


