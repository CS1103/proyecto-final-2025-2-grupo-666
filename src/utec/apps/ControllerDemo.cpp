//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/ControllerDemo.h"
#include <array>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>

using namespace utec::neural_network;

ControllerDemo::ControllerDemo() {
    std::srand((unsigned) std::time(nullptr));
    build_dataset();

    auto initW = [](auto& W) {
        for (size_t i = 0; i < W.shape()[0]; i++)
            for (size_t j = 0; j < W.shape()[1]; j++)
                W(i,j) = ((double) std::rand() / RAND_MAX)*0.2 - 0.1;
    };

    auto initB = [](auto& b) {
        for (size_t j = 0; j < b.shape()[1]; j++)
            b(0,j) = 0.0;
    };

    nn.add_layer(std::make_unique<Dense<double>>(2, 16, initW, initB));
    nn.add_layer(std::make_unique<ReLU<double>>());

    nn.add_layer(std::make_unique<Dense<double>>(16, 8, initW, initB));
    nn.add_layer(std::make_unique<ReLU<double>>());

    nn.add_layer(std::make_unique<Dense<double>>(8, 3, initW, initB));

    nn.add_layer(std::make_unique<Softmax<double>>());
}

void ControllerDemo::build_dataset() {
    X.clear();
    Y.clear();

    std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 0.4);

    const int N = 4000;

    for (int i = 0; i < N; i++) {

        double pos = dist(gen);
        if (pos < -1) pos = -1;
        if (pos > 1)  pos = 1;

        double vel = dist(gen);
        if (vel < -1) vel = -1;
        if (vel > 1)  vel = 1;

        X.push_back({pos, vel});

        int action = 1;

        if (pos < -0.2)
            action = 2;
        else if (pos > 0.2)
            action = 0;
        else if (std::fabs(vel) > 0.1)
            action = (vel > 0 ? 0 : 2);
        else
            action = 1;

        Y.push_back(action);
    }
}

void ControllerDemo::train(size_t epochs, double lr) {
    const size_t N = X.size();

    Tensor<double,2> X_tensor(std::array<size_t,2>{N, 2});
    Tensor<double,2> Y_tensor(std::array<size_t,2>{N, 3});

    for (size_t i = 0; i < N; i++) {
        X_tensor(i,0) = X[i][0];
        X_tensor(i,1) = X[i][1];

        Y_tensor(i,0) = (Y[i] == 0 ? 1.0 : 0.0);
        Y_tensor(i,1) = (Y[i] == 1 ? 1.0 : 0.0);
        Y_tensor(i,2) = (Y[i] == 2 ? 1.0 : 0.0);
    }

    nn.train<CrossEntropyLoss, SGD>(X_tensor, Y_tensor, epochs, 1, lr);
}

int ControllerDemo::predict(double position, double velocity) {
    Tensor<double,2> x(std::array<size_t,2>{1,2});
    x(0,0) = position;
    x(0,1) = velocity;

    auto out = nn.predict(x);

    int best = 0;
    double val = out(0,0);

    for (int i = 1; i < 3; i++)
        if (out(0,i) > val) { best = i; val = out(0,i); }

    return best;
}

void ControllerDemo::run_simulation(int steps) {
    auto [p, v] = env.reset();

    std::cout << "Simulacion del controlador:\n";

    for (int t = 0; t < steps; t++) {

        Tensor<double,2> x(std::array<size_t,2>{1,2});
        x(0,0)=p;
        x(0,1)=v;

        auto probs = nn.predict(x);
        int action = predict(p, v);

        auto result = env.step(action);

        p = result.first;
        v = result.second;

        std::cout << "t=" << t
                  << "  pos=" << p
                  << "  vel=" << v
                  << "  probs=["
                  << probs(0,0) << ", "
                  << probs(0,1) << ", "
                  << probs(0,2) << "]"
                  << "  action=" << action << "\n";
    }
}

void ControllerDemo::save_model(const std::string& filename) {
    nn.save_model(filename);
}

void ControllerDemo::load_model(const std::string& filename) {
    nn.load_model(filename);
}
