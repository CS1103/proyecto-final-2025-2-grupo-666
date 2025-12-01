//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/TrainingPipeline.h"
#include "utec/apps/SequencePredictor.h"
#include "utec/apps/PatternClassifier.h"

#include <iostream>

using namespace std;

double sequence_mse(SequencePredictor& model,
                    const std::vector<std::vector<double>>& X,
                    const std::vector<double>& Y)
{
    double mse = 0;
    size_t N = X.size();

    for (size_t i = 0; i < N; i++) {
        double p = model.predict(X[i]);
        double diff = (p - Y[i]);
        mse += diff * diff;
    }
    return mse / N;
}

double pattern_acc(PatternClassifier& model,
                   const std::vector<std::vector<double>>& X,
                   const std::vector<std::vector<double>>& Y)
{
    size_t ok = 0;
    size_t N = X.size();

    for (size_t i = 0; i < N; i++) {
        int pred = model.predict(X[i]);

        int real_class = 0;
        for (int j = 0; j < 3; j++)
            if (Y[i][j] == 1) real_class = j;

        if (pred == real_class) ok++;
    }
    return (double)ok / N;
}


void run_sequence_pipeline()
{
    cout << "\n=============== PIPELINE: SequencePredictor ===============\n";

    SequencePredictor base_model(false);
    base_model.train(1, 0.01);

    auto X = base_model.get_X();
    auto Y = base_model.get_Y();


    cout << "\n[Random Search]\n";

    auto result_rs =
        TrainingPipeline::random_search(
            [&](size_t id, size_t epochs, double lr){
                SequencePredictor tmp(false);
                tmp.train(epochs, lr);

                base_model = tmp;
            },
            [&](){
                return sequence_mse(base_model, X, Y);
            },
            12,
            500, 2500,
            0.001, 0.2
        );

    cout << "  Mejor MSE = " << result_rs.first
         << " con epochs=" << result_rs.second.first
         << "  lr=" << result_rs.second.second << "\n";


    cout << "\n[Hill Climbing]\n";

    auto result_hc =
        TrainingPipeline::hill_climb(
            [&](size_t id, size_t epochs, double lr){
                SequencePredictor tmp(false);
                tmp.train(epochs, lr);
                base_model = tmp;
            },
            [&](){
                return sequence_mse(base_model, X, Y);
            },
            0.05,
            0.01,
            8,
            1500
        );

    cout << "  Mejor MSE = " << result_hc.first
         << " con lr=" << result_hc.second << "\n";


    cout << "\n[Grid Search]\n";

    vector<size_t> epoch_list = {800, 1500, 2500};
    vector<double> lr_list = {0.1, 0.05, 0.01};

    auto result_gs =
        TrainingPipeline::grid_search(
            [&](size_t id, size_t epochs, double lr){
                SequencePredictor tmp(false);
                tmp.train(epochs, lr);
                base_model = tmp;
            },
            [&](){
                return sequence_mse(base_model, X, Y);
            },
            epoch_list,
            lr_list
        );

    cout << "  Mejor MSE = " << result_gs.first
         << " con epochs=" << result_gs.second.first
         << "  lr=" << result_gs.second.second << "\n";

    cout << "===========================================================\n";
}


void run_pattern_pipeline()
{
    cout << "\n=============== PIPELINE: PatternClassifier ===============\n";

    PatternClassifier base_model(false);
    base_model.train(1, 0.1);

    auto X = base_model.get_X();
    auto Y = base_model.get_Y();


    cout << "\n[Random Search]\n";

    auto result_rs =
        TrainingPipeline::random_search(
            [&](size_t id, size_t epochs, double lr){
                PatternClassifier tmp(false);
                tmp.train(epochs, lr);
                base_model = tmp;
            },
            [&](){
                return 1.0 - pattern_acc(base_model, X, Y);
            },
            10,
            300, 2000,
            0.001, 0.2
        );

    cout << "  Mejor Error = " << result_rs.first
         << " con epochs=" << result_rs.second.first
         << "  lr=" << result_rs.second.second << "\n";


    cout << "\n[Grid Search]\n";

    vector<size_t> epoch_list = {300, 800, 1500};
    vector<double> lr_list = {0.1, 0.05, 0.01};

    auto result_gs =
        TrainingPipeline::grid_search(
            [&](size_t id, size_t epochs, double lr){
                PatternClassifier tmp(false);
                tmp.train(epochs, lr);
                base_model = tmp;
            },
            [&](){
                return 1.0 - pattern_acc(base_model, X, Y);
            },
            epoch_list,
            lr_list
        );

    cout << "  Mejor Error = " << result_gs.first
         << " con epochs=" << result_gs.second.first
         << "  lr=" << result_gs.second.second << "\n";

    cout << "===========================================================\n";
}