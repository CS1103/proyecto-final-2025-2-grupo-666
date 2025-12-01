//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"

namespace utec::neural_network {

    template <typename T>
    class Dense final : public ILayer<T> {
        Tensor<T, 2> _W;
        Tensor<T, 2> _b;

        Tensor<T, 2> _x;
        Tensor<T, 2> _dW;
        Tensor<T, 2> _db;

    public:
        template <typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
            : _W(std::array<size_t, 2>{in_f, out_f})
            , _b(std::array<size_t, 2>{1, out_f}) {

            init_w_fun(_W);
            init_b_fun(_b);

            _dW = Tensor<T, 2>(std::array<size_t, 2>{in_f, out_f}, 0.0);
            _db = Tensor<T, 2>(std::array<size_t, 2>{1, out_f}, 0.0);
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            _x = x;
            return _x.dot(_W) + _b;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
            _dW = _x.transpose().dot(dZ);
            _db = dZ.sum(0);
            Tensor<T, 2> dX = dZ.dot(_W.transpose());
            return dX;
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(_W, _dW);
            optimizer.update(_b, _db);
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
