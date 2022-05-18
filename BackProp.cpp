#include <iostream>
#include <vector>
#include "Examples.h"

class Layer {
    double** weights;
    double* nodes;
    double* errors;
    int layerNumber;

    public:
        int numNodes;
        int nextNumNodes;

        // Default layer (Where there exists a next layer)
        Layer(int num, int size, int nextLayerSize) {
            // Set id number
            layerNumber = num;
            // Init nodes
            nodes = new double[size];
            numNodes = size;
            nextNumNodes = nextLayerSize;

            // Init weights going to next layer
            weights = new double* [size];
            for (int i = 0; i < size; i++) {
                weights[i] = new double[nextLayerSize];
            }

            errors = new double[size];
        }

        // End output layer
        Layer(int num, int size) {
            // Set id number
            layerNumber = num;
            // Init nodes
            nodes = new double[size];
            numNodes = size;
            nextNumNodes = 0;

            // Set weights to be null
            weights = nullptr;

            errors = new double[size];
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

};

class Network {
    std::vector<Layer> layers;

    public:
        int numLayers = 0;
        Network() {

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



void back_prop_learning(std::vector<Example> examples,  Network network) {
    //repeat
    // init all weights in network
    for (Layer l : network.getLayers()) {
        for (int i = 0; i < l.numNodes; i++) {
            for (int j = 0; j < l.nextNumNodes; j++) {
                l.setWeight(i, j, ((double)rand() / (RAND_MAX)));
            }
        }
        l.printWeights();
    }

    // for each input output pair 
    for (Example example : examples) {
        // load input vector to first layers nodes
        Layer inputLayer = network.getLayer(0);
        for (int i = 0; i < inputLayer.numNodes; i++) {
            inputLayer.setNode(i, example.x(i));
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
                //TODO need to pass in_j through activation function first ---------------------------!!!!!!
                layer.setNode(j, in_j);
            }
            layer.printNodes();
        }

        // front propogation done at this point
        Layer outputLayer = network.getLayer(network.numLayers - 1);
        for (int j = 0; j < outputLayer.numNodes; j++) {
            
        }

    }
}



int main() {
    std::cout << "--- Starting Program ---\n\n";

    Layer one = Layer(1, 3, 2);
    Layer two = Layer(2, 2);

    Network net = Network();
    net.addLayer(one);
    net.addLayer(two);

    std::vector<Example> examples;
    double test1in[] = { 1,0,0 };
    double test1out[] = { 1,0 };
    examples.push_back(Example(test1in, test1out));

    back_prop_learning(examples, net);
}


