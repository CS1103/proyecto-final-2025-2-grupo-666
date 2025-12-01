//
// Created by Usuario on 1/12/2025.
//

#ifndef PROYECTO_FINAL_2025_2_GRUPO_666_ENVGYM_H
#define PROYECTO_FINAL_2025_2_GRUPO_666_ENVGYM_H

#pragma once

#include <utility>

class EnvGym {
public:
    EnvGym();

    std::pair<double,double> reset();

    std::pair<double,double> step(int action);

private:
    double position;
    double velocity;

    double force_left  = -0.1;
    double force_right =  0.1;
    double friction    =  0.01;
};

#endif //PROYECTO_FINAL_2025_2_GRUPO_666_ENVGYM_H