#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "Examples.h"
#include "ActivationFunctions.h"
#include "Layer.h"
#include "Network.h"
#include "CSV.h"


void back_prop_learning(std::vector<Example> examples,  Network network) {
    // init all weights in network
    for (Layer l : network.getLayers()) {
        for (int i = 0; i < l.numNodes; i++) {
            for (int j = 0; j < l.nextNumNodes; j++) {
                l.setWeight(i, j, ((double)rand() / (RAND_MAX)));
            }
        }
    }

    for (int a = 0; a < 1000; a++) {
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

            // need to do softmax properly after assigning all values
            //outputLayer.printInputs();
            outputLayer.setNodesArray(arraySoftMax(outputLayer.getInputsArray(), outputLayer.numNodes));
            //outputLayer.printNodes();
            //example.print();


            // front propogation done at this point
            // calculate cross entropy loss
            double crossEntropyLoss = 0;
            for (int i = 0; i < outputLayer.numNodes; i++) {
                //std::cout << " i - " << example.y(i) * log(outputLayer.getNode(i));
                double nodeValue = outputLayer.getNode(i);
                if (nodeValue == 0) { nodeValue = 0.001; }
                crossEntropyLoss += example.y(i) * log(nodeValue);

            }
            totalCrossEntropyLoss += -crossEntropyLoss;;

            //std::cout << totalCrossEntropyLoss << "\n";


            // time to backpropagate errors through the network

            // calculate error at output nodes from difference to expected output
            for (int j = 0; j < outputLayer.numNodes; j++) {
                outputLayer.setError(j, softMaxDerivative(outputLayer.getNode(j)) * (example.y(j) - outputLayer.getNode(j)));
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
    
}




int main() {
    std::cout << "--- Starting Program ---\n\n";

    //Layer one = Layer(1, 2, 2);
    //Layer onepointfive = Layer(2, 2, 2);
    //Layer two = Layer(3, 2);

    //Network net = Network(0.8);
    //net.addLayer(one);
    //net.addLayer(onepointfive);
    //net.addLayer(two);


    //std::vector<Example> examples;
    //double test1in[] = { 1,0 };
    //double test1out[] = { 1,0 };

    //double test2in[] = { 0,1 };
    //double test2out[] = { 0,1 };

    //double test3in[] = { 1,1 };
    //double test3out[] = { 0.5,0.5 };

    //examples.push_back(Example(test1in, test1out, 2, 2));
    //examples.push_back(Example(test2in, test2out, 2, 2));
    //examples.push_back(Example(test3in, test3out, 2, 2));

    //back_prop_learning(examples, net);

    Layer one = Layer(1, 54, 20);
    Layer two = Layer(2, 20, 2);
    Layer three = Layer(3, 2, 0);

    Network net = Network(0.1);
    net.addLayer(one);
    net.addLayer(two);
    net.addLayer(three);


    std::string fileName = "test.csv";
    std::vector<std::vector<double>> csv = readCSV(fileName);
    std::vector<Example> examples2 = csvToExamples(csv, 2);



    back_prop_learning(examples2, net);


}



