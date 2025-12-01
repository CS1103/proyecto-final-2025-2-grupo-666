//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/PatternClassifier.h"
#include <fstream>
#include <array>
#include <cstdlib>
#include <random>

using namespace utec::neural_network;

PatternClassifier::PatternClassifier(bool use_noise)
    : use_noise(use_noise)
{
    build_dataset();

    srand((unsigned) time(nullptr));

    auto initW = [](auto& W){
        for (size_t i = 0; i < W.shape()[0]; i++)
            for (size_t j = 0; j < W.shape()[1]; j++)
                W(i,j) = ((double) rand() / RAND_MAX) * 0.2 - 0.1;
    };
    auto initB = [](auto& b){
        for (size_t j = 0; j < b.shape()[1]; j++)
            b(0,j) = 0;
    };

    nn.add_layer(std::make_unique<Dense<double>>(3, 12, initW, initB));
    nn.add_layer(std::make_unique<ReLU<double>>());

    nn.add_layer(std::make_unique<Dense<double>>(12, 6, initW, initB));
    nn.add_layer(std::make_unique<ReLU<double>>());

    nn.add_layer(std::make_unique<Dense<double>>(6, 3, initW, initB));
    nn.add_layer(std::make_unique<Softmax<double>>());
}

void PatternClassifier::build_dataset() {
    X.clear();
    Y.clear();

    if (!use_noise) {
        X = {
            {0.9,0.1,0.1},
            {0.1,0.9,0.1},
            {0.1,0.1,0.9}
        };
        Y = {
            {1,0,0},
            {0,1,0},
            {0,0,1}
        };
        return;
    }

    std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 0.05);

    std::vector<double> circle   = {0.9,0.1,0.1};
    std::vector<double> square   = {0.1,0.9,0.1};
    std::vector<double> triangle = {0.1,0.1,0.9};

    for (int i = 0; i < 300; i++) {
        X.push_back({ circle[0]+dist(gen), circle[1]+dist(gen), circle[2]+dist(gen) });
        Y.push_back({1,0,0});

        X.push_back({ square[0]+dist(gen), square[1]+dist(gen), square[2]+dist(gen) });
        Y.push_back({0,1,0});

        X.push_back({ triangle[0]+dist(gen), triangle[1]+dist(gen), triangle[2]+dist(gen) });
        Y.push_back({0,0,1});
    }
}

void PatternClassifier::train(size_t epochs, double lr) {
    size_t N = X.size();

    Tensor<double,2> X_tensor( std::array<size_t,2>{N,3} );
    Tensor<double,2> Y_tensor( std::array<size_t,2>{N,3} );

    for (size_t i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++)
            X_tensor(i,j) = X[i][j];

        for (int j = 0; j < 3; j++)
            Y_tensor(i,j) = Y[i][j];
    }

    nn.train<CrossEntropyLoss, SGD>(X_tensor, Y_tensor, epochs, 1, lr);
}

int PatternClassifier::predict(const std::vector<double>& input) {
    Tensor<double,2> x( std::array<size_t,2>{1,3} );
    for (int i = 0; i < 3; i++) x(0,i) = input[i];

    auto out = nn.predict(x);
    int best = 0;
    double bv = out(0,0);

    for (int i = 1; i < 3; i++)
        if (out(0,i) > bv) { best = i; bv = out(0,i); }

    return best;
}

void PatternClassifier::save_model(const std::string& filename) {
    nn.save_model(filename);
}
void PatternClassifier::load_model(const std::string& filename) {
    nn.load_model(filename);
}
