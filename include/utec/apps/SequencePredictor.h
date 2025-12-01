//
// Created by Usuario on 1/12/2025.
//

#ifndef PROYECTO_FINAL_2025_2_GRUPO_666_SEQUENCEPREDICTOR_H
#define PROYECTO_FINAL_2025_2_GRUPO_666_SEQUENCEPREDICTOR_H

#pragma once

#include <vector>
#include <string>
#include "utec/nn/neural_network.h"

class SequencePredictor {
public:
    SequencePredictor();

    void train(size_t epochs = 3000, double lr = 0.01);

    double predict(const std::vector<double>& input);

    void save_model(const std::string& filename);
    void load_model(const std::string& filename);

private:
    utec::neural_network::NeuralNetwork<double> nn;

    std::vector<std::vector<double>> X;
    std::vector<double> Y;

    void build_dataset();
};

#endif //PROYECTO_FINAL_2025_2_GRUPO_666_SEQUENCEPREDICTOR_H