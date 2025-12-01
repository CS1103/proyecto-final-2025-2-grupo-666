//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <vector>
#include <memory>

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
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
