//
// Created by Usuario on 1/12/2025.
//

#include <cassert>
#include <iostream>
#include <cmath>

#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"

using namespace utec::neural_network;

int main() {

    // ============================================================
    // 1) CONSTRUCCIÓN BÁSICA
    // ============================================================
    NeuralNetwork<double> nn;

    nn.add_layer(std::make_unique<Dense<double>>(2, 4,
        [](auto& W){
            for(size_t i=0;i<2;i++)
                for(size_t j=0;j<4;j++)
                    W(i,j)=0.1;
        },
        [](auto& b){
            for(size_t j=0;j<4;j++) b(0,j)=0.0;
        }
    ));
    nn.add_layer(std::make_unique<ReLU<double>>());

    nn.add_layer(std::make_unique<Dense<double>>(4, 1,
        [](auto& W){
            for(size_t i=0;i<4;i++)
                for(size_t j=0;j<1;j++)
                    W(i,j)=0.2;
        },
        [](auto& b){ b(0,0)=0.0; }
    ));

    // ============================================================
    // 2) FORWARD
    // ============================================================
    Tensor<double,2> X(std::array<size_t,2>{1,2});
    X(0,0)=1; X(0,1)=-1;

    auto y = nn.predict(X);
    assert(y.shape()[1] == 1);

    // ============================================================
    // 3) ENTRENAMIENTO PEQUEÑO
    // ============================================================
    Tensor<double,2> Y(std::array<size_t,2>{1,1});
    Y(0,0)=0.5;

    nn.train<MSELoss, SGD>(X, Y, 100, 1, 0.1);
    auto y2 = nn.predict(X);

    assert(std::isfinite(y2(0,0)));

    // ============================================================
    // 4) SERIALIZACIÓN / DESERIALIZACIÓN
    // ============================================================
    nn.save_model("nn_test.model");

    NeuralNetwork<double> nn2;
    nn2.load_model("nn_test.model");

    auto y3 = nn2.predict(X);

    assert(std::abs(y3(0,0) - y2(0,0)) < 1e-6);

    // ============================================================
    // 5) TEST GRAFO — Verificamos orden correcto de capas
    // (Forward debe aumentar tamaño o mantenerlo según la capa).
    // ============================================================
    NeuralNetwork<double> nn_graph;
    nn_graph.add_layer(std::make_unique<Dense<double>>(3,5,
        [](auto& W){ for(size_t i=0;i<3;i++) for(size_t j=0;j<5;j++) W(i,j)=0.1; },
        [](auto& b){ for(size_t j=0;j<5;j++) b(0,j)=0; }
    ));
    nn_graph.add_layer(std::make_unique<ReLU<double>>());
    nn_graph.add_layer(std::make_unique<Dense<double>>(5,2,
        [](auto& W){ for(size_t i=0;i<5;i++) for(size_t j=0;j<2;j++) W(i,j)=0.1; },
        [](auto& b){ for(size_t j=0;j<2;j++) b(0,j)=0; }
    ));

    Tensor<double,2> Xg(std::array<size_t,2>{1,3});
    Xg(0,0)=1; Xg(0,1)=2; Xg(0,2)=3;

    auto Yg = nn_graph.predict(Xg);
    assert(Yg.shape()[1] == 2);

    std::cout << "test_neural_network.cpp OK\n";
    return 0;
}
