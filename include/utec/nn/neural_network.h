//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"

#include <vector>
#include <memory>
#include <fstream>
#include <string>

namespace utec::neural_network {

    template <typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> _layers;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            _layers.push_back(std::move(layer));
        }

        Tensor<T, 2> predict(const Tensor<T, 2>& X) {
            Tensor<T, 2> current_output = X;
            for (auto& layer : _layers) {
                current_output = layer->forward(current_output);
            }
            return current_output;
        }

        template <template <typename...> class LossType,
                  template <typename...> class OptimizerType = SGD>
        void train(const Tensor<T, 2>& X, const Tensor<T, 2>& Y,
                   const size_t epochs, const size_t batch_size, T learning_rate) {

            OptimizerType<T> optimizer(learning_rate);

            for (size_t epoch = 0; epoch < epochs; epoch++) {
                Tensor<T, 2> Y_pred = predict(X);

                LossType<T> loss_fn(Y_pred, Y);
                Tensor<T, 2> grad = loss_fn.loss_gradient();

                for (auto it = _layers.rbegin(); it != _layers.rend(); it++) {
                    grad = (*it)->backward(grad);
                }

                for (auto& layer : _layers) {
                    layer->update_params(optimizer);
                }

                optimizer.step();
            }
        }

        // ============================================================
        //                SERIALIZACIÓN DEL MODELO
        // ============================================================
        void save_model(const std::string& filename) {
            std::ofstream f(filename);
            if (!f.is_open()) return;

            // Guardamos número de capas
            f << _layers.size() << "\n";

            // Guardamos cada capa
            for (auto& layer : _layers) {
                f << layer->type() << "\n";
                layer->save(f);
                f << "---\n"; // separador
            }
        }

        void load_model(const std::string& filename) {
            std::ifstream f(filename);
            if (!f.is_open()) return;

            size_t num_layers = 0;
            f >> num_layers;

            _layers.clear();

            for (size_t i = 0; i < num_layers; ++i) {
                std::string type;
                if (!(f >> type)) break;

                if (type == "Dense") {
                    auto lyr = std::make_unique<Dense<T>>(1, 1, [](auto&){}, [](auto&){});
                    lyr->load(f);
                    _layers.push_back(std::move(lyr));
                }
                else if (type == "ReLU") {
                    auto lyr = std::make_unique<ReLU<T>>();
                    lyr->load(f);
                    _layers.push_back(std::move(lyr));
                }
                else if (type == "Sigmoid") {
                    auto lyr = std::make_unique<Sigmoid<T>>();
                    lyr->load(f);
                    _layers.push_back(std::move(lyr));
                }

                // consumir separador '---'
                std::string sep;
                f >> sep; // debe ser ---
            }
        }

    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
