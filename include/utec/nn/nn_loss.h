//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "utec/nn/nn_interfaces.h"
#include <cmath>
#include <numeric>

namespace utec::neural_network {

    template <typename T>
    class MSELoss final : public ILoss<T, 2> {
        Tensor<T, 2> _y_pred;
        Tensor<T, 2> _y_true;
        T _n_elements;

    public:
        MSELoss(const Tensor<T, 2>& y_prediction, const Tensor<T, 2>& y_true)
            : _y_pred(y_prediction), _y_true(y_true) {
            _n_elements = static_cast<T>(y_prediction.size());
        }

        T loss() const override {
            Tensor<T, 2> diff = _y_pred - _y_true;
            Tensor<T, 2> squared_diff = diff * diff;
            T sum = std::accumulate(squared_diff.begin(), squared_diff.end(), static_cast<T>(0));
            return sum / _n_elements;
        }

        Tensor<T, 2> loss_gradient() const override {
            return (_y_pred - _y_true) * (static_cast<T>(2) / _n_elements);
        }
    };


    template <typename T>
    class BCELoss final : public ILoss<T, 2> {
        Tensor<T, 2> _y_pred;
        Tensor<T, 2> _y_true;
        T _n_elements;
        T _epsilon = 1e-8;

    public:
        BCELoss(const Tensor<T, 2>& y_prediction, const Tensor<T, 2>& y_true)
            : _y_pred(y_prediction), _y_true(y_true) {
            _n_elements = static_cast<T>(y_prediction.size());
        }

        T loss() const override {
            Tensor<T, 2> term1 = _y_true * utec::algebra::log(_y_pred + _epsilon);
            Tensor<T, 2> term2 = (static_cast<T>(1) - _y_true) * utec::algebra::log(static_cast<T>(1) - _y_pred + _epsilon);
            Tensor<T, 2> sum_terms = term1 + term2;

            T total_sum = std::accumulate(sum_terms.begin(), sum_terms.end(), static_cast<T>(0));
            return -(total_sum / _n_elements);
        }

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> term1 = -(_y_true / (_y_pred + _epsilon));
            Tensor<T, 2> term2 = (static_cast<T>(1) - _y_true) / (static_cast<T>(1) - _y_pred + _epsilon);
            return (term1 + term2) * (static_cast<T>(1) / _n_elements);
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
