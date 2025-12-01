//
// Created by Usuario on 1/12/2025.
//

#include <cassert>
#include <iostream>
#include <cmath>
#include "utec/algebra/tensor.h"

using namespace utec::algebra;

int main() {

    // ============================================================
    // 1) CREACIÓN BÁSICA
    // ============================================================
    Tensor<double,2> A(std::array<size_t,2>{2,2});
    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;

    assert(A(0,0)==1);
    assert(A(1,1)==4);

    // ============================================================
    // 2) TRANSPOSE
    // ============================================================
    auto T = A.transpose();
    assert(T(0,1)==3);
    assert(T(1,0)==2);

    // ============================================================
    // 3) DOT PRODUCT
    // ============================================================
    auto D = A.dot(T);
    assert(D(0,0)==5);
    assert(D(1,1)==25);

    // ============================================================
    // 4) OPERACIONES ELEMENT-WISE
    // ============================================================
    Tensor<double,2> B(std::array<size_t,2>{2,2});
    B(0,0)=5; B(0,1)=6;
    B(1,0)=7; B(1,1)=8;

    auto C = A + B;
    assert(C(1,1) == 12);

    // ============================================================
    // 5) FUNCIONES MATEMÁTICAS
    // ============================================================
    auto E = exp(A);
    assert(std::abs(E(0,0) - std::exp(1)) < 1e-9);

    auto L = log(exp(A));
    assert(std::abs(L(1,1) - 4) < 1e-9);

    // ============================================================
    // 6) CASOS BORDE — Tensores 1xN, Nx1
    // ============================================================
    Tensor<double,2> R(std::array<size_t,2>{1,3});
    R(0,0)=1; R(0,1)=2; R(0,2)=3;

    Tensor<double,2> C2(std::array<size_t,2>{3,1});
    C2(0,0)=1; C2(1,0)=2; C2(2,0)=3;

    auto dot1 = R.dot(C2);
    assert(dot1(0,0)==14);

    auto dot2 = C2.dot(R);
    assert(dot2(2,1)==6);

    // ============================================================
    // 7) STRESS TEST — Dot product 200×200
    // ============================================================
    Tensor<double,2> S1(std::array<size_t,2>{200,200});
    Tensor<double,2> S2(std::array<size_t,2>{200,200});
    for(size_t i=0;i<200;i++)
        for(size_t j=0;j<200;j++){
            S1(i,j)=1;
            S2(i,j)=1;
        }

    auto S3 = S1.dot(S2);
    assert(S3(0,0)==200);

    // ============================================================
    // 8) TEST DE ESTABILIDAD
    // ============================================================
    Tensor<double,2> Z(std::array<size_t,2>{1,1});
    Z(0,0)=1e-12;
    auto Ltiny = log(Z);
    assert(std::isfinite(Ltiny(0,0)));

    std::cout << "test_tensor.cpp OK\n";
    return 0;
}
