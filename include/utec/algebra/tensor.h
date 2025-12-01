//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <vector>
#include <array>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>
#include <random>
#include <type_traits>

namespace utec::algebra {

    template <typename T, size_t DIMS>
    class Tensor {
        std::array<size_t, DIMS> _shape;
        std::vector<T> _data;

        void broadcast_add(const Tensor<T, DIMS>& other) {
            if (other._shape[0] == 1 && other._shape[1] == _shape[1]) {
                for (size_t i = 0; i < _shape[0]; ++i) {
                    for (size_t j = 0; j < _shape[1]; ++j) {
                        (*this)(i, j) += other(0, j);
                    }
                }
            } else {
                for (size_t i = 0; i < _data.size(); ++i) {
                    _data[i] += other._data[i];
                }
            }
        }

    public:
        Tensor() : _shape{}, _data{} {}

        explicit Tensor(const std::array<size_t, DIMS>& shape) : _shape(shape) {
            size_t total_size = 1;
            for (auto dim : _shape) {
                total_size *= dim;
            }
            _data.resize(total_size);
        }

        explicit Tensor(size_t rows, size_t cols) : _shape({rows, cols}) {
            static_assert(DIMS == 2, "This constructor (rows, cols) is only for 2D Tensors");
            _data.resize(rows * cols);
        }

        explicit Tensor(const std::array<size_t, DIMS>& shape, T fill_value) : Tensor(shape) {
            fill(fill_value);
        }

        T& operator()(size_t r, size_t c) {
            static_assert(DIMS == 2, "2D accessor requires DIMS == 2");
            return _data[r * _shape[1] + c];
        }

        const T& operator()(size_t r, size_t c) const {
            static_assert(DIMS == 2, "2D accessor requires DIMS == 2");
            return _data[r * _shape[1] + c];
        }

        Tensor& operator=(std::initializer_list<T> list) {
            static_assert(DIMS == 2, "Initializer list assignment requires DIMS == 2");
            if (list.size() != _data.size()) {
                throw std::invalid_argument("Initializer list size mismatch");
            }
            std::copy(list.begin(), list.end(), _data.begin());
            return *this;
        }

        auto begin() { return _data.begin(); }
        auto end() { return _data.end(); }
        auto begin() const { return _data.cbegin(); }
        auto end() const { return _data.cend(); }

        const std::array<size_t, DIMS>& shape() const { return _shape; }
        size_t size() const { return _data.size(); }
        void fill(T value) { std::fill(_data.begin(), _data.end(), value); }

        Tensor<T, 2> dot(const Tensor<T, 2>& other) const {
            static_assert(DIMS == 2, "Dot product requires DIMS == 2");
            if (_shape[1] != other._shape[0]) {
                throw std::invalid_argument("Matrix dimensions mismatch for dot product");
            }
            size_t M = _shape[0];
            size_t K = _shape[1];
            size_t N = other._shape[1];

            Tensor<T, 2> result(std::array<size_t, 2>{M, N});
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    T sum = 0;
                    for (size_t k = 0; k < K; ++k) {
                        sum += (*this)(i, k) * other(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }

        Tensor<T, 2> transpose() const {
            static_assert(DIMS == 2, "Transpose requires DIMS == 2");
            size_t R = _shape[0];
            size_t C = _shape[1];
            Tensor<T, 2> result(std::array<size_t, 2>{C, R});
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }
            return result;
        }

        Tensor<T, 2> sum(int axis) const {
            static_assert(DIMS == 2, "Sum(axis) requires DIMS == 2");
            if (axis == 0) {
                Tensor<T, 2> result(std::array<size_t, 2>{1, _shape[1]}, 0.0);
                for (size_t j = 0; j < _shape[1]; ++j) {
                    for (size_t i = 0; i < _shape[0]; ++i) {
                        result(0, j) += (*this)(i, j);
                    }
                }
                return result;
            }
            throw std::invalid_argument("Sum axis not supported");
        }

        Tensor operator+(const Tensor& other) const {
            Tensor result = *this;
            result.broadcast_add(other);
            return result;
        }

        Tensor operator-(const Tensor& other) const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) {
                result._data[i] = _data[i] - other._data[i];
            }
            return result;
        }

        Tensor operator*(const Tensor& other) const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) {
                result._data[i] = _data[i] * other._data[i];
            }
            return result;
        }

        Tensor operator/(const Tensor& other) const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) {
                result._data[i] = _data[i] / other._data[i];
            }
            return result;
        }

        Tensor operator+(T scalar) const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) result._data[i] = _data[i] + scalar;
            return result;
        }

        Tensor operator-(T scalar) const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) result._data[i] = _data[i] - scalar;
            return result;
        }

        Tensor operator*(T scalar) const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) result._data[i] = _data[i] * scalar;
            return result;
        }

        Tensor operator/(T scalar) const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) result._data[i] = _data[i] / scalar;
            return result;
        }

        Tensor operator-() const {
            Tensor result(_shape);
            for (size_t i = 0; i < _data.size(); ++i) result._data[i] = -_data[i];
            return result;
        }

        friend Tensor operator+(T scalar, const Tensor& t) {
            return t + scalar;
        }

        friend Tensor operator-(T scalar, const Tensor& t) {
            Tensor<T, DIMS> result(t.shape());
            for (size_t i = 0; i < t.size(); ++i) result.begin()[i] = scalar - t.begin()[i];
            return result;
        }

        friend Tensor operator/(T scalar, const Tensor& t) {
            Tensor<T, DIMS> result(t.shape());
            for (size_t i = 0; i < t.size(); ++i) result.begin()[i] = scalar / t.begin()[i];
            return result;
        }
    };

    template<typename T, size_t DIMS>
    std::ostream& operator<<(std::ostream& os, const Tensor<T, DIMS>& tensor) {
        static_assert(DIMS == 2, "Ostream operator requires DIMS == 2");
        auto shape = tensor.shape();
        os << "{\n";
        for (size_t i = 0; i < shape[0]; ++i) {
            os << (i == 0 ? "" : "\n");
            for (size_t j = 0; j < shape[1]; ++j) {
                os << tensor(i, j) << (j == shape[1] - 1 ? "" : " ");
            }
        }
        os << "\n}";
        return os;
    }

    template<typename T, size_t DIMS>
    Tensor<T, DIMS> log(const Tensor<T, DIMS>& t) {
        Tensor<T, DIMS> result(t.shape());
        for (size_t i = 0; i < t.size(); ++i) result.begin()[i] = std::log(t.begin()[i]);
        return result;
    }

    template<typename T, size_t DIMS>
    Tensor<T, DIMS> exp(const Tensor<T, DIMS>& t) {
        Tensor<T, DIMS> result(t.shape());
        for (size_t i = 0; i < t.size(); ++i) result.begin()[i] = std::exp(t.begin()[i]);
        return result;
    }

    template<typename T, size_t DIMS>
    Tensor<T, DIMS> sqrt(const Tensor<T, DIMS>& t) {
        Tensor<T, DIMS> result(t.shape());
        for (size_t i = 0; i < t.size(); ++i) result.begin()[i] = std::sqrt(t.begin()[i]);
        return result;
    }

    template<typename T, size_t DIMS>
    Tensor<T, DIMS> max(T val, const Tensor<T, DIMS>& t) {
        Tensor<T, DIMS> result(t.shape());
        for (size_t i = 0; i < t.size(); ++i) result.begin()[i] = std::max(val, t.begin()[i]);
        return result;
    }

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
