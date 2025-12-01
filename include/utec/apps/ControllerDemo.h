//
// Created by Usuario on 1/12/2025.
//

#ifndef PROYECTO_FINAL_2025_2_GRUPO_666_CONTROLLERDEMO_H
#define PROYECTO_FINAL_2025_2_GRUPO_666_CONTROLLERDEMO_H

#pragma once

#include <vector>
#include "utec/apps/EnvGym.h"
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"

class ControllerDemo {
public:
    ControllerDemo();

    /// Entrenamiento supervisado con reglas simples
    void train(size_t epochs = 2000, double lr = 0.05);

    /// Decide acción según la red
    int predict(double position, double velocity);

    /// Corre una simulación con la red
    void run_simulation(int steps = 30);

    void save_model(const std::string& filename);
    void load_model(const std::string& filename);

private:
    utec::neural_network::NeuralNetwork<double> nn;
    EnvGym env;

    std::vector<std::vector<double>> X;
    std::vector<int> Y;

    void build_dataset();
};

#endif //PROYECTO_FINAL_2025_2_GRUPO_666_CONTROLLERDEMO_H