//
// Created by Usuario on 1/12/2025.
//

#include "utec/apps/EnvGym.h"
#include <cmath>

EnvGym::EnvGym() : position(0.0), velocity(0.0) {}

std::pair<double,double> EnvGym::reset() {
    position = ( (rand() % 2000) / 1000.0 ) - 1.0; // [-1,1]
    velocity = 0.0;
    return {position, velocity};
}

std::pair<double,double> EnvGym::step(int action) {
    double force = 0.0;

    if (action == 0) force = force_left;
    if (action == 2) force = force_right;

    velocity += force;
    velocity *= (1.0 - friction); // fricción

    position += velocity;

    return {position, velocity};
}
