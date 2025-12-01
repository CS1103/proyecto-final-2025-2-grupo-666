//
// Created by Usuario on 1/12/2025.
//

#ifndef PROYECTO_FINAL_2025_2_GRUPO_666_SEQUENCEPREDICTOR_H
#define PROYECTO_FINAL_2025_2_GRUPO_666_SEQUENCEPREDICTOR_H

#pragma once

#include <vector>
#include <string>
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"

class SequencePredictor {
public:
    SequencePredictor();

    /// Entrena el modelo con un dataset sintético de series numéricas
    void train(size_t epochs = 3000, double lr = 0.01);

    /// Predice el siguiente valor dado un vector de entrada [x1, x2, x3]
    double predict(const std::vector<double>& input);

    void save_model(const std::string& filename);
    void load_model(const std::string& filename);

private:
    utec::neural_network::NeuralNetwork<double> nn;

    // Dataset sintético: entradas y salidas
    std::vector<std::vector<double>> X;   // secuencias [a/20, (a+1)/20, (a+2)/20]
    std::vector<double> Y;                // siguientes valores (a+3)/20

    void build_dataset();
};

#endif //PROYECTO_FINAL_2025_2_GRUPO_666_SEQUENCEPREDICTOR_H