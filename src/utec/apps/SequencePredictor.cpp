//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/SequencePredictor.h"
#include <fstream>
#include <array>
#include <cstdlib>
#include <ctime>

using namespace utec::neural_network;

SequencePredictor::SequencePredictor() {
    build_dataset();

    srand((unsigned) time(nullptr));

    nn.add_layer(std::make_unique<Dense<double>>(3, 8,
        [](auto& W){
            for (size_t i = 0; i < W.shape()[0]; i++)
                for (size_t j = 0; j < W.shape()[1]; j++)
                    W(i,j) = ((double) rand() / RAND_MAX) * 0.2 - 0.1;
        },
        [](auto& b){
            for (size_t i = 0; i < b.shape()[0]; i++)
                for (size_t j = 0; j < b.shape()[1]; j++)
                    b(i,j) = 0.0;
        }
    ));

    nn.add_layer(std::make_unique<ReLU<double>>());

    nn.add_layer(std::make_unique<Dense<double>>(8, 1,
        [](auto& W){
            for (size_t i = 0; i < W.shape()[0]; i++)
                for (size_t j = 0; j < W.shape()[1]; j++)
                    W(i,j) = ((double) rand() / RAND_MAX) * 0.2 - 0.1;
        },
        [](auto& b){
            for (size_t i = 0; i < b.shape()[0]; i++)
                for (size_t j = 0; j < b.shape()[1]; j++)
                    b(i,j) = 0.0;
        }
    ));
}

void SequencePredictor::build_dataset() {
    X.clear();
    Y.clear();
    for (int a = 1; a <= 20; ++a) {
        X.push_back({
            (double)a / 20.0,
            (double)(a+1) / 20.0,
            (double)(a+2) / 20.0
        });

        Y.push_back((double)(a+3) / 20.0);
    }
}

void SequencePredictor::train(size_t epochs, double lr) {
    const size_t n_samples = X.size();
    const size_t n_features = 3;

    Tensor<double,2> X_tensor(std::array<size_t,2>{n_samples, n_features});
    Tensor<double,2> Y_tensor(std::array<size_t,2>{n_samples, 1});

    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X_tensor(i, j) = X[i][j];
        }
        Y_tensor(i, 0) = Y[i];
    }

    nn.train<MSELoss, SGD>(X_tensor, Y_tensor, epochs, 1, lr);
}

double SequencePredictor::predict(const std::vector<double>& input) {
    if (input.size() != 3) return 0.0;

    Tensor<double,2> x(std::array<size_t,2>{1,3});
    for (size_t j = 0; j < 3; ++j)
        x(0,j) = input[j] / 20.0;

    Tensor<double,2> out = nn.predict(x);

    return out(0,0) * 20.0;
}

void SequencePredictor::save_model(const std::string& filename) {
    nn.save_model(filename);
}

void SequencePredictor::load_model(const std::string& filename) {
    nn.load_model(filename);
}

