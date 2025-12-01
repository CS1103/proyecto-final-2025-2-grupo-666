//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "nn_interfaces.h"
#include <cmath>
#include <map>

namespace utec::neural_network {

    template <typename T>
    class SGD final : public IOptimizer<T> {
        T _learning_rate;

    public:
        explicit SGD(T learning_rate = 0.01) : _learning_rate(learning_rate) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            params = params - grads * _learning_rate;
        }
    };


    template <typename T>
    class Adam final : public IOptimizer<T> {
        T _lr;
        T _beta1;
        T _beta2;
        T _epsilon;
        size_t _t;

        struct AdamState {
            Tensor<T, 2> m;
            Tensor<T, 2> v;

            AdamState(const std::array<size_t, 2>& shape)
                : m(shape, 0.0), v(shape, 0.0) {}
        };

        std::map<void*, AdamState> _states;

    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : _lr(learning_rate), _beta1(beta1), _beta2(beta2), _epsilon(epsilon), _t(0) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            void* key = &params;

            if (_states.find(key) == _states.end()) {
                _states.emplace(key, AdamState(params.shape()));
            }

            AdamState& state = _states.at(key);
            size_t t_step = _t + 1;

            state.m = state.m * _beta1 + grads * (1 - _beta1);

            state.v = state.v * _beta2 + (grads * grads) * (1 - _beta2);

            Tensor<T, 2> m_hat = state.m / (1 - std::pow(_beta1, t_step));

            Tensor<T, 2> v_hat = state.v / (1 - std::pow(_beta2, t_step));

            params = params - (m_hat * _lr) / (algebra::sqrt(v_hat) + _epsilon);
        }

        void step() override {
            _t++;
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
