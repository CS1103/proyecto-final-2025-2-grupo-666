//
// Created by rudri on 10/11/2020.
//
#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace utec {
    namespace algebra {

        /**
         * @brief Representa un tensor N-dimensional.
         * @details Almacena los datos de forma contigua en memoria (row-major)
         * para optimizar el acceso y las operaciones.
         * @tparam T Tipo de dato de los elementos.
         * @tparam N Número de dimensiones (rango) del tensor.
         */

        template<typename T, std::size_t N>
        class Tensor {
        public:
            using value_type = T;
            using iterator = typename std::vector<T>::iterator;
            using const_iterator = typename std::vector<T>::const_iterator;

        private:
            std::array<std::size_t, N> dims_{}; // Dimensiones del tensor (forma o shape).
            std::array<std::size_t, N> strides_{};  // Pasos (strides) para navegar entre dimensiones en el arreglo 1D.
            std::vector<T> data_{};  // Almacenamiento contiguo de datos.

            // Calcula el número total de elementos multiplicando las dimensiones.
            static std::size_t total_size(const std::array<std::size_t, N>& d) {
                return std::accumulate(d.begin(), d.end(), std::size_t{1}, std::multiplies<>{});
            }

            // Calcula los strides en formato row-major. El stride de una dimensión
            // indica cuántos elementos saltar en `data_` para avanzar en esa dimensión.
            void compute_strides() {
                std::size_t s = 1;
                for (std::size_t i = N; i-- > 0;) {
                    strides_[i] = s;
                    s *= dims_[i];
                }
            }

            // Actualiza la forma, strides y tamaño del contenedor de datos.
            void set_shape_and_resize(const std::array<std::size_t, N>& new_dims) {
                dims_ = new_dims;
                compute_strides();
                data_.resize(total_size(dims_)); // puede crecer o reducir; rellena con T{}
            }

            // Convierte un índice N-dimensional a su equivalente lineal en el vector `data_`.
            std::size_t linear_index_from_array(const std::array<std::size_t, N>& idx) const {
                std::size_t lin = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    if (idx[i] >= dims_[i]) throw std::out_of_range("Index out of range");
                    lin += idx[i] * strides_[i];
                }
                return lin;
            }

            // Métodos de acceso interno por arreglo de índices.
            T& at_indices(const std::array<std::size_t, N>& idx) {
                return data_[linear_index_from_array(idx)];
            }

            const T& at_indices(const std::array<std::size_t, N>& idx) const {
                return data_[linear_index_from_array(idx)];
            }

            /**
             * @brief Implementación genérica para operaciones elemento a elemento (+, -, *)
             * con soporte para broadcasting.
             * @details Comprueba la compatibilidad de formas, calcula la forma del resultado
             * y aplica la operación iterando sobre el tensor resultante.
             */
            template<typename Op>
            Tensor elementwise_impl(const Tensor& other, Op op, const char* err_bc_msg) const {
                // Calcular forma resultante aplicando reglas de broadcasting.
                std::array<std::size_t, N> rdims{};
                bool compatible = true;
                for (std::size_t i = 0; i < N; ++i) {
                    const auto a = dims_[i];
                    const auto b = other.dims_[i];
                    if (a == b) {
                        rdims[i] = a;
                    } else if (a == 1) {
                        rdims[i] = b;
                    } else if (b == 1) {
                        rdims[i] = a;
                    } else {
                        compatible = false; break;
                    }
                }
                if (!compatible) {
                    throw std::invalid_argument(err_bc_msg);
                }

                Tensor r(rdims);
                // Iterar todas las posiciones del tensor resultado.
                std::array<std::size_t, N> idx{}; // Índice en r
                const auto total = total_size(rdims);
                for (std::size_t flat = 0; flat < total; ++flat) {
                    // Mapear índice del resultado a los índices de los tensores originales.
                    std::array<std::size_t, N> ia{}, ib{};
                    for (std::size_t d = 0; d < N; ++d) {
                        ia[d] = (dims_[d] == 1) ? 0 : idx[d];
                        ib[d] = (other.dims_[d] == 1) ? 0 : idx[d];
                    }
                    r.at_indices(idx) = op(at_indices(ia), other.at_indices(ib));
                    // Incrementar el índice multidimensional.
                    for (std::size_t d = N; d-- > 0;) {
                        if (++idx[d] < rdims[d]) break; else idx[d] = 0;
                    }
                }
                return r;
            }

        public:
            Tensor() = default;

            explicit Tensor(const std::array<std::size_t, N>& dims) {
                set_shape_and_resize(dims);
            }

            // Constructor variádico para crear un tensor a partir de sus dimensiones.
            // Verifica en tiempo de compilación que el número de argumentos coincida con N.
            template<typename... Dims>
            explicit Tensor(Dims... dims) {
                if constexpr (sizeof...(Dims) != N) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
                } else {
                    std::array<std::size_t, N> d{
                        static_cast<std::size_t>(dims)...
                    };
                    set_shape_and_resize(d);
                }
            }

            // Asignación de datos desde una initializer_list.
            Tensor& operator=(std::initializer_list<T> values) {
                if (values.size() != data_.size()) {
                    throw std::invalid_argument("Data size does not match tensor size");
                }
                std::copy(values.begin(), values.end(), data_.begin());
                return *this;
            }

            // Operador de acceso a elementos mediante índices variádicos `t(i, j, ...)`.
            template<typename... Idx>
            T& operator()(Idx... idxs) {
                if constexpr (sizeof...(Idx) != N) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
                } else {
                    std::array<std::size_t, N> idx{
                        static_cast<std::size_t>(idxs)...
                    };
                    return at_indices(idx);
                }
            }

            template<typename... Idx>
            const T& operator()(Idx... idxs) const {
                if constexpr (sizeof...(Idx) != N) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
                } else {
                    std::array<std::size_t, N> idx{
                        static_cast<std::size_t>(idxs)...
                    };
                    return at_indices(idx);
                }
            }

            // Acceso público por arreglo, usado por funciones libres.
            T& at_array(const std::array<std::size_t, N>& idx) {
                return at_indices(idx);
            }

            const T& at_array(const std::array<std::size_t, N>& idx) const {
                return at_indices(idx);
            }

            // Devuelve las dimensiones (shape) del tensor.
            std::array<std::size_t, N> shape() const {
                return dims_;
            }

            // Rellena el tensor con un único valor.
            void fill(const T& value) {
                std::fill(data_.begin(), data_.end(), value);
            }

            // Iteradores para acceso directo al contenedor de datos.
            iterator begin() {
                return data_.begin();
            }
            iterator end() {
                return data_.end();
            }
            const_iterator begin() const {
                return data_.begin();
            }
            const_iterator end() const {
                return data_.end();
            }
            const_iterator cbegin() const {
                return data_.cbegin();
            }
            const_iterator cend() const {
                return data_.cend();
            }

            // Modifica las dimensiones del tensor. El número total de elementos puede cambiar.
            template<typename... Dims>
            void reshape(Dims... dims) {
                if constexpr (sizeof...(Dims) != N) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
                } else {
                    std::array<std::size_t, N> nd{
                        static_cast<std::size_t>(dims)...
                    };
                    set_shape_and_resize(nd);
                }
            }

            // Operaciones elemento a elemento entre tensores.
            Tensor operator+(const Tensor& other) const {
                return elementwise_impl(other, [](const T& a, const T& b) {
                    return a + b;
                },
                    "Shapes do not match and they are not compatible for broadcasting");
            }

            Tensor operator-(const Tensor& other) const {
                return elementwise_impl(other, [](const T& a, const T& b) {
                    return a - b;
                },
                    "Shapes do not match and they are not compatible for broadcasting");
            }

            Tensor operator*(const Tensor& other) const {
                return elementwise_impl(other, [](const T& a, const T& b) {
                    return a * b;
                },
                    "Shapes do not match and they are not compatible for broadcasting");
            }

            // Operaciones entre tensor y escalar.
            Tensor operator+(const T& s) const {
                Tensor r(*this);
                for (auto& v : r.data_) v += s;
                return r;
            }

            Tensor operator-(const T& s) const {
                Tensor r(*this);
                for (auto& v : r.data_) v -= s;
                return r;
            }

            Tensor operator*(const T& s) const {
                Tensor r(*this);
                for (auto& v : r.data_) v *= s;
                return r;
            }

            Tensor operator/(const T& s) const {
                Tensor r(*this);
                for (auto& v : r.data_) v /= s;
                return r;
            }

            // Operaciones entre escalar y tensor.
            friend Tensor operator+(const T& s, const Tensor& t) {
                return t + s;
            }
            friend Tensor operator*(const T& s, const Tensor& t) {
                return t * s;
            }
            friend Tensor operator-(const T& s, const Tensor& t) {
                Tensor r(t);
                for (auto& v : r.data_) v = s - v;
                return r;
            }
            friend Tensor operator/(const T& s, const Tensor& t) {
                Tensor r(t);
                for (auto& v : r.data_) v = s / v;
                return r;
            }

            // Sobrecarga del operador de inserción para imprimir el tensor con formato anidado.
            // Utiliza una función lambda recursiva para manejar las diferentes dimensiones.
            friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
                if (t.data_.empty() && N > 0) {
                    os << "{}";
                    return os;
                }

                std::array<std::size_t, N> idx{};

                std::function<void(std::size_t)> print_recursive =
                    [&](std::size_t dim) {

                        if (dim == N - 1) {
                            for (std::size_t i = 0; i < t.dims_[dim]; ++i) {
                                idx[dim] = i;
                                os << t.at_indices(idx) << (i + 1 < t.dims_[dim] ? " " : "");
                            }
                            return;
                        }

                        os << "{\n";
                        for (std::size_t i = 0; i < t.dims_[dim]; ++i) {
                            idx[dim] = i;
                            print_recursive(dim + 1);
                            if (i < t.dims_[dim] - 1) {
                                os << "\n";
                            }
                        }
                        os << "\n}";
                };

                if constexpr (N == 1) {
                    for (std::size_t i = 0; i < t.dims_[0]; ++i) {
                        os << t.data_[i] << (i + 1 < t.dims_[0] ? " " : "");
                    }
                }
                else if constexpr (N > 1) {
                    print_recursive(0);
                }
                else {
                    if (!t.data_.empty()) os << t.data_[0];
                }

                return os;
            }
        };

        // Función libre para transponer las dos últimas dimensiones de un tensor.
        template<std::size_t N, typename T>
        Tensor<T, N> transpose_2d(const Tensor<T, N>& t) {
            if (N < 2) {
                throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
            }

            auto shp = t.shape();
            std::array<std::size_t, N> new_shp = shp;
            std::swap(new_shp[N-2], new_shp[N-1]);

            Tensor<T, N> r(new_shp);

            std::array<std::size_t, N> idx{};
            const auto total = std::accumulate(new_shp.begin(), new_shp.end(), std::size_t{1}, std::multiplies<>{});

            for (std::size_t flat = 0; flat < total; ++flat) {
                auto src = idx;
                std::swap(src[N-2], src[N-1]);

                r.at_array(idx) = t.at_array(src);

                // Incrementar idx
                for (std::size_t d = N; d-- > 0;) {
                    if (++idx[d] < new_shp[d]) break;
                    else idx[d] = 0;
                }
            }

            return r;
        }

        /**
         * @brief Función libre para la multiplicación de matrices.
         * @details Soporta matrices 2D (MxK * KxN) y lotes de matrices 3D (BxMxK * BxKxN).
         */
        template<std::size_t N, typename T>
        Tensor<T, N> matrix_product(const Tensor<T, N>& a, const Tensor<T, N>& b) {
            static_assert(N == 2 || N == 3, "Matrix product only implemented for 2D and 3D tensors");

            const auto sa = a.shape();
            const auto sb = b.shape();

            // Validar compatibilidad de dimensiones internas y de lote (batch).
            if (sa[N-1] != sb[N-2]) {
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
            }

            if constexpr (N == 3) {
                if (sa[0] != sb[0]) {
                    throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
                }
            }

            // Calcular forma del resultado.
            std::array<std::size_t, N> rs{};
            if constexpr (N == 2) {
                rs[0] = sa[0]; rs[1] = sb[1];
            }
            else {
                rs[0] = sa[0]; rs[1] = sa[1]; rs[2] = sb[2];
            }

            Tensor<T, N> r(rs);

            if constexpr (N == 2) {
                for (std::size_t i = 0; i < sa[0]; ++i) {
                    for (std::size_t j = 0; j < sb[1]; ++j) {
                        T acc{};
                        for (std::size_t k = 0; k < sa[1]; ++k) acc += a(i, k) * b(k, j);
                        r(i, j) = acc;
                    }
                }
            } else {
                for (std::size_t batch = 0; batch < sa[0]; ++batch) {
                    for (std::size_t i = 0; i < sa[1]; ++i) {
                        for (std::size_t j = 0; j < sb[2]; ++j) {
                            T acc{};
                            for (std::size_t k = 0; k < sa[2]; ++k) acc += a(batch, i, k) * b(batch, k, j);
                            r(batch, i, j) = acc;
                        }
                    }
                }
            }

            return r;
        }

    } // namespace algebra
} // namespace utec

#endif // PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H