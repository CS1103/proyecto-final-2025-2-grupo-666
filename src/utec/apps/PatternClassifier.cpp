//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/PatternClassifier.h"
#include <fstream>
#include <array>
#include <cstdlib>

using namespace utec::neural_network;

PatternClassifier::PatternClassifier() {
    build_dataset();

    // Inicializar generador aleatorio
    srand((unsigned) time(nullptr));

    // Capa oculta 1
    nn.add_layer(std::make_unique<Dense<double>>(3, 8,
        [](auto& W){
            for (size_t i = 0; i < W.shape()[0]; i++)
                for (size_t j = 0; j < W.shape()[1]; j++)
                    W(i,j) = ((double) rand() / RAND_MAX) * 0.2 - 0.1; // [-0.1, 0.1]
        },
        [](auto& b){
            for (size_t i = 0; i < b.shape()[0]; i++)
                for (size_t j = 0; j < b.shape()[1]; j++)
                    b(i,j) = 0;
        }
    ));

    nn.add_layer(std::make_unique<ReLU<double>>());

    // Capa de salida
    nn.add_layer(std::make_unique<Dense<double>>(8, 3,
        [](auto& W){
            for (size_t i = 0; i < W.shape()[0]; i++)
                for (size_t j = 0; j < W.shape()[1]; j++)
                    W(i,j) = ((double) rand() / RAND_MAX) * 0.2 - 0.1;
        },
        [](auto& b){
            for (size_t i = 0; i < b.shape()[0]; i++)
                for (size_t j = 0; j < b.shape()[1]; j++)
                    b(i,j) = 0;
        }
    ));

    nn.add_layer(std::make_unique<Sigmoid<double>>());
}

void PatternClassifier::build_dataset() {
    // Tres patrones sintéticos
    X = {
        {0.9, 0.1, 0.1}, // círculo
        {0.1, 0.9, 0.1}, // cuadrado
        {0.1, 0.1, 0.9}  // triángulo
    };

    Y = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
}

void PatternClassifier::train(size_t epochs, double lr) {

    Tensor<double,2> X_tensor(std::array<size_t,2>{3,3});
    Tensor<double,2> Y_tensor(std::array<size_t,2>{3,3});

    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            X_tensor(i,j) = X[i][j];
            Y_tensor(i,j) = Y[i][j];
        }
    }

    nn.train<MSELoss, SGD>(X_tensor, Y_tensor, epochs, 1, lr);
}

int PatternClassifier::predict(const std::vector<double>& input) {

    Tensor<double,2> x(std::array<size_t,2>{1,3});

    for (int i = 0; i < 3; i++)
        x(0,i) = input[i];

    Tensor<double,2> out = nn.predict(x);

    int best = 0;
    double best_val = out(0,0);

    for (int i = 1; i < 3; i++){
        if (out(0,i) > best_val){
            best = i;
            best_val = out(0,i);
        }
    }

    return best;
}

void PatternClassifier::save_model(const std::string& filename) {
    nn.save_model(filename);
}

void PatternClassifier::load_model(const std::string& filename) {
    nn.load_model(filename);
}

