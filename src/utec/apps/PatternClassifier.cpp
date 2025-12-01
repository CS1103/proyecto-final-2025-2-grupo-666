//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/PatternClassifier.h"
#include <fstream>
#include <array>
#include <cstdlib>
#include <random>
#include <iostream>

using namespace utec::neural_network;

PatternClassifier::PatternClassifier() {
    std::srand((unsigned) time(nullptr));
    build_dataset();

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

    std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> noise(0.0, 0.05);

    std::vector<double> circle    = {0.9, 0.1, 0.1};
    std::vector<double> square    = {0.1, 0.9, 0.1};
    std::vector<double> triangle  = {0.1, 0.1, 0.9};

    const int N = 300;

    for (int i = 0; i < N; i++) {
        // círculo
        X.push_back({
            circle[0] + noise(gen),
            circle[1] + noise(gen),
            circle[2] + noise(gen)
        });
        Y.push_back({1,0,0});

        // cuadrado
        X.push_back({
            square[0] + noise(gen),
            square[1] + noise(gen),
            square[2] + noise(gen)
        });
        Y.push_back({0,1,0});

        // triángulo
        X.push_back({
            triangle[0] + noise(gen),
            triangle[1] + noise(gen),
            triangle[2] + noise(gen)
        });
        Y.push_back({0,0,1});
    }
}

void PatternClassifier::train(size_t epochs, double lr) {

    const size_t N = X.size();

    Tensor<double,2> X_tensor(std::array<size_t,2>{N,3});
    Tensor<double,2> Y_tensor(std::array<size_t,2>{N,3});

    for (size_t i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++)
            X_tensor(i,j) = X[i][j];

        for (int j = 0; j < 3; j++)
            Y_tensor(i,j) = Y[i][j];
    }

    nn.train<CrossEntropyLoss, SGD>(X_tensor, Y_tensor, epochs, 1, lr);

    int correct = 0;
    int confusion[3][3] = {0};

    for (size_t i = 0; i < N; i++) {
        int pred = predict(X[i]);
        int real = (Y[i][0] == 1 ? 0 : (Y[i][1] == 1 ? 1 : 2));

        if (pred == real) correct++;

        confusion[real][pred]++;
    }

    double accuracy = (double)correct / N;

    std::cout << "\n=== PatternClassifier metricas ===\n";
    std::cout << "Precision: " << accuracy*100.0 << "%\n\n";
    std::cout << "Matriz de confusion:\n";
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++)
            std::cout << confusion[i][j] << " ";
        std::cout << "\n";
    }
    std::cout << "=================================\n\n";
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
