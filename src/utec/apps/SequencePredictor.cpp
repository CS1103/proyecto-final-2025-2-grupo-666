//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/SequencePredictor.h"
#include <fstream>
#include <random>
#include <cmath>

SequencePredictor::SequencePredictor(bool use_noise)
    : use_noise(use_noise)
{
    build_dataset();
}

void SequencePredictor::build_dataset() {
    X.clear();
    Y.clear();

    std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> noise(0.0, 0.05);

    for (int a = 1; a <= 20; a++) {
        double x1 = a;
        double x2 = a + 1;
        double x3 = a + 2;
        double y  = a + 3;

        X.push_back({x1, x2, x3});
        Y.push_back(y);

        if (use_noise) {
            X.push_back({ x1 + noise(gen), x2 + noise(gen), x3 + noise(gen) });
            Y.push_back(y);
        }
    }
}

void SequencePredictor::train(size_t epochs, double lr) {

    double best_w = w_;
    double best_b = b_;
    double best_mse = 1e18;

    for (size_t it = 0; it < epochs; it++) {

        for (int step = 0; step < 20; step++) {

            double test_w = w_ + ((rand() / double(RAND_MAX)) - 0.5) * lr;
            double test_b = b_ + ((rand() / double(RAND_MAX)) - 0.5) * lr;

            double mse = 0;
            for (size_t i = 0; i < X.size(); i++) {
                double sum = (X[i][0] + X[i][1] + X[i][2]) / 3.0;
                double pred = sum * test_w + test_b;

                double diff = pred - Y[i];
                mse += diff * diff;
            }
            mse /= X.size();

            if (mse < best_mse) {
                best_mse = mse;
                best_w = test_w;
                best_b = test_b;
            }
        }

        w_ = best_w;
        b_ = best_b;
    }
}

double SequencePredictor::predict(const std::vector<double>& input) {
    double mean = (input[0] + input[1] + input[2]) / 3.0;
    return mean * w_ + b_;
}

void SequencePredictor::save_model(const std::string& filename) {
    std::ofstream f(filename);
    f << w_ << " " << b_;
}

void SequencePredictor::load_model(const std::string& filename) {
    std::ifstream f(filename);
    f >> w_ >> b_;
}