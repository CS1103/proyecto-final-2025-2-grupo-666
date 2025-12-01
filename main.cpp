//
// Proyecto Final - EPIC 3
// main.cpp
//

#include <iostream>
#include "utec/apps/PatternClassifier.h"
#include "utec/apps/SequencePredictor.h"
#include "utec/apps/ControllerDemo.h"
#include "utec/apps/TrainingPipeline.h"

int main() {

    // ===========================================================
    // 1) PATTERN CLASSIFIER — Clasificacion de patrones
    // ===========================================================
    std::cout << "===========================================\n";
    std::cout << "   1) PatternClassifier (Entrenamiento)\n";
    std::cout << "===========================================\n";

    PatternClassifier pc(true);
    pc.train(3000, 0.1);

    std::cout << "Predicciones:\n";
    std::cout << "  {0.9,0.1,0.1} -> " << pc.predict({0.9, 0.1, 0.1}) << " (esperado ~0)\n";
    std::cout << "  {0.1,0.9,0.1} -> " << pc.predict({0.1, 0.9, 0.1}) << " (esperado ~1)\n";
    std::cout << "  {0.1,0.1,0.9} -> " << pc.predict({0.1, 0.1, 0.9}) << " (esperado ~2)\n";

    // --- Serializamos el modelo ---
    pc.save_model("pattern.model");
    std::cout << "Modelo PatternClassifier guardado en pattern.model\n";

    // --- Lo cargamos en un nuevo objeto ---
    PatternClassifier pc_loaded;
    pc_loaded.load_model("pattern.model");

    std::cout << "Prediccion desde modelo cargado: "
              << pc_loaded.predict({0.9, 0.1, 0.1})
              << " (esperado ~0)\n";


    // ===========================================================
    // 2) SEQUENCE PREDICTOR — Prediccion de series numéricas
    // ===========================================================
    std::cout << "\n===========================================\n";
    std::cout << "   2) SequencePredictor (Entrenamiento)\n";
    std::cout << "===========================================\n";

    SequencePredictor sp(true);
    sp.train(4000, 0.01);

    std::cout << "Predicciones:\n";
    std::cout << "  [1,2,3]   -> " << sp.predict({1,2,3})    << " (esperado ~4)\n";
    std::cout << "  [5,6,7]   -> " << sp.predict({5,6,7})    << " (esperado ~8)\n";
    std::cout << "  [10,11,12]-> " << sp.predict({10,11,12}) << " (esperado ~13)\n";

    // --- Serializar modelo ---
    sp.save_model("sequence.model");
    std::cout << "Modelo SequencePredictor guardado en sequence.model\n";

    // --- Cargar modelo ---
    SequencePredictor sp_loaded;
    sp_loaded.load_model("sequence.model");

    std::cout << "Prediccion desde modelo cargado [1,2,3]: "
              << sp_loaded.predict({1,2,3})
              << " (esperado ~4)\n";


    // ===========================================================
    // 3) CONTROLLER DEMO — Control estilo Gym
    // ===========================================================
    std::cout << "\n===========================================\n";
    std::cout << "   3) ControllerDemo (Entrenamiento)\n";
    std::cout << "===========================================\n";

    ControllerDemo cd;
    cd.train(3000, 0.05);

    // --- Serializar controlador ---
    cd.save_model("controller.model");
    std::cout << "Modelo ControllerDemo guardado en controller.model\n";

    // --- Cargar modelo ---
    ControllerDemo cd_loaded;
    cd_loaded.load_model("controller.model");

    std::cout << "\nSimulacion usando el modelo cargado:\n";
    cd_loaded.run_simulation(25);


    // ===========================================================
    // 4) GENERALIZACIÓN Y ROBUSTEZ — Requisito EPIC 3
    // ===========================================================
    std::cout << "\n===========================================\n";
    std::cout << "   4) Generalizacion y Robustez\n";
    std::cout << "===========================================\n";

    // Entrenamos y guardamos
    SequencePredictor spA;
    spA.train(2000, 0.01);
    spA.save_model("robust.model");

    // Cargamos modelo
    SequencePredictor spB;
    spB.load_model("robust.model");

    std::cout << "\n-- Prueba con ruido --\n";
    std::cout << "Entrada ruidosa [1.05, 2.1, 3.2] -> "
              << spB.predict({1.05, 2.1, 3.2})
              << " (esperado cercano a 4)\n";

    std::cout << "\n-- Prueba fuera de rango --\n";
    std::cout << "Entrada [20, 21, 22] -> "
              << spB.predict({20,21,22})
              << " (esperado cercano a 23)\n";

    std::cout << "\n-- Prueba con valores negativos --\n";
    std::cout << "Entrada [-5, -4, -3] -> "
              << spB.predict({-5, -4, -3})
              << " (generalizacion no vista)\n";

    // =======================================
    //    PIPELINE EXPERIMENTAL
    // =======================================
    run_sequence_pipeline();
    run_pattern_pipeline();

    // ===========================================================
    // FIN
    // ===========================================================
    std::cout << "\nFin de ejecucion.\n";
    return 0;
}
