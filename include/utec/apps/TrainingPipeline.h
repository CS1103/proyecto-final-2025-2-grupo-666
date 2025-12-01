//
// Created by Usuario on 1/12/2025.
//

#ifndef PROYECTO_FINAL_2025_2_GRUPO_666_TRAININGPIPELINE_H
#define PROYECTO_FINAL_2025_2_GRUPO_666_TRAININGPIPELINE_H

#pragma once

#include <vector>
#include <functional>
#include <utility>
#include <random>

class TrainingPipeline {
public:

    using MetricFn = std::function<double()>;

    using TrainFn = std::function<void(size_t,size_t,double)>;

    static std::pair<double, std::pair<size_t,double>>
    random_search(TrainFn train_fn, MetricFn metric_fn,
                  int trials,
                  size_t min_epochs, size_t max_epochs,
                  double min_lr, double max_lr)
    {
        std::default_random_engine gen(std::random_device{}());
        std::uniform_int_distribution<size_t> epoch_dist(min_epochs, max_epochs);
        std::uniform_real_distribution<double> lr_dist(min_lr, max_lr);

        double best_mse = 1e18;
        size_t best_epochs = 0;
        double best_lr = 0;

        for (int i = 0; i < trials; i++) {
            size_t epochs = epoch_dist(gen);
            double lr = lr_dist(gen);

            train_fn(i, epochs, lr);
            double mse = metric_fn();

            if (mse < best_mse) {
                best_mse = mse;
                best_epochs = epochs;
                best_lr = lr;
            }
        }

        return { best_mse, { best_epochs, best_lr } };
    }

    static std::pair<double,double>
    hill_climb(TrainFn train_fn, MetricFn metric_fn,
               double start_lr, double step, int iterations,
               size_t epochs)
    {
        double lr = start_lr;
        double best_mse = 1e18;

        for (int i = 0; i < iterations; i++) {
            double lr_candidate = lr + ((i % 2 == 0) ? step : -step);

            train_fn(i, epochs, lr_candidate);
            double mse = metric_fn();

            if (mse < best_mse) {
                best_mse = mse;
                lr = lr_candidate;
            }
        }

        return { best_mse, lr };
    }

    static std::pair<double,std::pair<size_t,double>>
    grid_search(TrainFn train_fn, MetricFn metric_fn,
                const std::vector<size_t>& epoch_list,
                const std::vector<double>& lr_list)
    {
        double best_mse = 1e18;
        size_t best_epochs = 0;
        double best_lr = 0;

        for (size_t e : epoch_list) {
            for (double lr : lr_list) {
                train_fn(0, e, lr);
                double mse = metric_fn();

                if (mse < best_mse) {
                    best_mse = mse;
                    best_epochs = e;
                    best_lr = lr;
                }
            }
        }

        return { best_mse, { best_epochs, best_lr } };
    }
};

void run_sequence_pipeline();
void run_pattern_pipeline();

#endif //PROYECTO_FINAL_2025_2_GRUPO_666_TRAININGPIPELINE_H