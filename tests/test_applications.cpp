//
// Created by Usuario on 1/12/2025.
//

#include <cassert>
#include <iostream>
#include "utec/apps/PatternClassifier.h"
#include "utec/apps/SequencePredictor.h"
#include "utec/apps/ControllerDemo.h"

int main() {

    // ============================================================
    // 1) PATTERN CLASSIFIER
    // ============================================================
    PatternClassifier pc;
    pc.train(2000, 0.1);

    assert(pc.predict({0.9,0.1,0.1}) == 0);
    assert(pc.predict({0.1,0.9,0.1}) == 1);
    assert(pc.predict({0.1,0.1,0.9}) == 2);

    assert(pc.predict({0.92,0.12,0.05}) == 0);

    pc.save_model("pc.model");
    PatternClassifier pc2;
    pc2.load_model("pc.model");
    assert(pc2.predict({0.9,0.1,0.1}) == 0);


    // ============================================================
    // 2) SEQUENCE PREDICTOR
    // ============================================================
    SequencePredictor sp;
    sp.train(2500, 0.01);

    double p1 = sp.predict({1,2,3});
    assert(std::abs(p1 - 4) < 1.0);

    double p2 = sp.predict({1.1,2.1,2.9});
    assert(std::isfinite(p2));

    sp.save_model("sp.model");
    SequencePredictor sp2;
    sp2.load_model("sp.model");

    double p3 = sp2.predict({1,2,3});
    assert(std::abs(p1 - p3) < 0.5);


    // ============================================================
    // 3) CONTROLLER DEMO
    // ============================================================
    ControllerDemo cd;
    cd.train(2000, 0.05);

    int a1 = cd.predict(0.5,0.0);
    assert(a1>=0 && a1<=2);

    int a2 = cd.predict(-0.9,0.2);
    assert(a2>=0 && a2<=2);

    cd.save_model("cd.model");
    ControllerDemo cd2;
    cd2.load_model("cd.model");

    int a3 = cd2.predict(0.5,0.0);
    assert(a3 == a1);

    std::cout << "test_applications.cpp OK\n";
    return 0;
}
