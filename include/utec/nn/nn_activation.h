//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "utec/nn/nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

    template <typename T>
    class ReLU final : public ILayer<T> {
        Tensor<T, 2> _z;
    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
            _z = z;
            return utec::algebra::max(static_cast<T>(0), z);
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& g) override {
            Tensor<T, 2> dM(g.shape());
            for (size_t i = 0; i < g.size(); ++i) {
                dM.begin()[i] = (_z.begin()[i] > 0) ? g.begin()[i] : 0;
            }
            return dM;
        }

        // ============================================
        //             SERIALIZACIÓN ReLU
        // ============================================
        std::string type() const override {
            return "ReLU";
        }

        void save(std::ostream& os) const override {
        }

        void load(std::istream& is) override {
        }
    };

    template <typename T>
    class Sigmoid final : public ILayer<T> {
        Tensor<T, 2> _s;
    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
            _s = static_cast<T>(1) / (static_cast<T>(1) + utec::algebra::exp(-z));
            return _s;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& g) override {
            return g * _s * (static_cast<T>(1) - _s);
        }

        // ============================================
        //          SERIALIZACIÓN Sigmoid
        // ============================================
        std::string type() const override {
            return "Sigmoid";
        }

        void save(std::ostream& os) const override {
        }

        void load(std::istream& is) override {
        }
    };

    template <typename T>
    class Softmax final : public ILayer<T> {
        Tensor<T, 2> _y;
    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
            auto shape = z.shape();
            size_t R = shape[0], C = shape[1];

            _y = Tensor<T,2>(shape);

            for (size_t i = 0; i < R; i++) {
                T max_val = z(i,0);
                for (size_t j = 1; j < C; j++)
                    if (z(i,j) > max_val) max_val = z(i,j);

                T sum = 0.0;
                for (size_t j = 0; j < C; j++) {
                    _y(i,j) = std::exp(z(i,j) - max_val);
                    sum += _y(i,j);
                }
                for (size_t j = 0; j < C; j++)
                    _y(i,j) /= sum;
            }
            return _y;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& g) override {
            return g * _y * (1 - _y);
        }

        std::string type() const override { return "Softmax"; }
        void save(std::ostream&) const override {}
        void load(std::istream&) override {}
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
