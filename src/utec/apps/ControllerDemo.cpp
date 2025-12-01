//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/ControllerDemo.h"
#include <array>
#include <iostream>

using namespace utec::neural_network;

ControllerDemo::ControllerDemo() {
    srand((unsigned) time(nullptr));
    build_dataset();

    // NN: entrada = 2 (pos, vel), oculta = 8, salida = 3 (acciones)
    nn.add_layer(std::make_unique<Dense<double>>(2, 8,
        [](auto& W){
            for (size_t i = 0; i < W.shape()[0]; i++)
                for (size_t j = 0; j < W.shape()[1]; j++)
                    W(i,j) = ((double) rand() / RAND_MAX)*0.2 - 0.1;
        },
        [](auto& b){
            for (size_t i = 0; i < b.shape()[0]; i++)
                for (size_t j = 0; j < b.shape()[1]; j++)
                    b(i,j) = 0;
        }
    ));

    nn.add_layer(std::make_unique<ReLU<double>>());

    nn.add_layer(std::make_unique<Dense<double>>(8, 3,
        [](auto& W){
            for (size_t i = 0; i < W.shape()[0]; i++)
                for (size_t j = 0; j < W.shape()[1]; j++)
                    W(i,j) = ((double) rand() / RAND_MAX)*0.2 - 0.1;
        },
        [](auto& b){
            for (size_t i = 0; i < b.shape()[0]; i++)
                for (size_t j = 0; j < b.shape()[1]; j++)
                    b(i,j) = 0;
        }
    ));

    nn.add_layer(std::make_unique<Sigmoid<double>>());
}

void ControllerDemo::build_dataset() {
    X.clear();
    Y.clear();

    // Regla base:
    // Si pos < -0.05 → mover derecha (2)
    // Si pos >  0.05 → mover izquierda (0)
    // Si cerca de 0 → detener (1)

    for (int i = 0; i < 200; i++) {
        double pos = ((rand() % 2000) / 1000.0) - 1.0; // [-1,1]
        double vel = ((rand() % 1000) / 1000.0) - 0.5; // [-0.5,0.5]

        X.push_back({pos, vel});

        if (pos < -0.05)      Y.push_back(2); // derecha
        else if (pos > 0.05)  Y.push_back(0); // izquierda
        else                  Y.push_back(1); // detener
    }
}

void ControllerDemo::train(size_t epochs, double lr) {

    const size_t N = X.size();

    Tensor<double,2> X_tensor(std::array<size_t,2>{N, 2});
    Tensor<double,2> Y_tensor(std::array<size_t,2>{N, 3});

    for (size_t i = 0; i < N; i++) {
        X_tensor(i,0) = X[i][0];
        X_tensor(i,1) = X[i][1];

        Y_tensor(i,0) = (Y[i] == 0 ? 1 : 0);
        Y_tensor(i,1) = (Y[i] == 1 ? 1 : 0);
        Y_tensor(i,2) = (Y[i] == 2 ? 1 : 0);
    }

    nn.train<MSELoss, SGD>(X_tensor, Y_tensor, epochs, 1, lr);
}

int ControllerDemo::predict(double position, double velocity) {
    Tensor<double,2> x(std::array<size_t,2>{1,2});
    x(0,0) = position;
    x(0,1) = velocity;

    auto out = nn.predict(x);

    int best = 0;
    double val = out(0,0);

    for (int i = 1; i < 3; i++){
        if (out(0,i) > val){
            best = i;
            val = out(0,i);
        }
    }
    return best;
}

void ControllerDemo::run_simulation(int steps) {
    auto [p, v] = env.reset();

    std::cout << "Simulacion del controlador:\n";

    for (int t = 0; t < steps; t++) {
        int action = predict(p, v);
        auto result = env.step(action);

        p = result.first;
        v = result.second;

        std::cout << "t=" << t
                  << "  pos=" << p
                  << "  vel=" << v
                  << "  action=" << action << "\n";
    }
}

void ControllerDemo::save_model(const std::string& filename) {
    nn.save_model(filename);
}

void ControllerDemo::load_model(const std::string& filename) {
    nn.load_model(filename);
}
