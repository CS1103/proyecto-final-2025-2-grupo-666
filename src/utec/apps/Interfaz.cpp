//
// Created by Usuario on 1/12/2025.
//

#include <iostream>
#include <limits>
#include <filesystem>
#include <thread>
#include <chrono>

#include "utec/apps/PatternClassifier.h"
#include "utec/apps/SequencePredictor.h"
#include "utec/apps/ControllerDemo.h"
#include "utec/apps/TrainingPipeline.h"

namespace fs = std::filesystem;
// COLORES ANSI
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

void clear_input() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

bool file_exists(const std::string& name) {
    return fs::exists(name);
}

void pause() {
    std::cout << YELLOW << "\nPresiona ENTER para continuar..." << RESET;
    std::cin.get();
}

// Loader animado
void loader(const std::string& msg, int steps = 3, int delay_ms = 250) {
    std::cout << CYAN << msg << RESET;
    for (int i = 0; i < steps; i++) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    std::cout << "\n";
}

// Input seguro
int read_int(const std::string& prompt, int min, int max) {
    while (true) {
        std::cout << prompt;

        int v;
        std::cin >> v;

        if (std::cin.good() && v >= min && v <= max) {
            return v;
        }

        std::cout << RED << "Entrada invalida. Intenta nuevamente.\n" << RESET;
        clear_input();
    }
}

void run_console_interface() {

    PatternClassifier pc;
    SequencePredictor sp;
    ControllerDemo cd;

    bool classifier_trained = false;
    bool sequence_trained = false;
    bool controller_trained = false;

    while (true) {

        std::cout << BLUE << BOLD;
        std::cout << "\n=========================================\n";
        std::cout << "   EPIC 3 - Neural Systems Interface\n";
        std::cout << "=========================================\n" << RESET;

        std::cout << CYAN;
        std::cout << "1) Entrenar PatternClassifier\n";
        std::cout << "2) Entrenar SequencePredictor\n";
        std::cout << "3) Entrenar ControllerDemo\n";
        std::cout << "4) Probar modelos entrenados\n";
        std::cout << "5) Guardar modelos\n";
        std::cout << "6) Cargar modelos\n";
        std::cout << "7) Ejecutar Pipeline Experimental\n";
        std::cout << "8) Salir\n" << RESET;

        int op = read_int("\nSelecciona una opcion: ", 1, 8);

        //   1) PATTERN CLASSIFIER
        if (op == 1) {
            loader("Entrenando PatternClassifier");
            pc = PatternClassifier(true);
            pc.train(3000, 0.1);
            classifier_trained = true;

            std::cout << GREEN << "Entrenado: " << RESET;
            std::cout << "{0.9,0.1,0.1} -> " << pc.predict({0.9,0.1,0.1}) << "\n";
            pause();
        }

        //   2) SEQUENCE PREDICTOR
        else if (op == 2) {
            loader("Entrenando SequencePredictor");
            sp = SequencePredictor(true);
            sp.train(3000, 0.01);
            sequence_trained = true;

            std::cout << GREEN << "Entrenado: " << RESET;
            std::cout << "[1,2,3] -> " << sp.predict({1,2,3}) << "\n";
            pause();
        }

        //   3) CONTROLLER DEMO
        else if (op == 3) {
            loader("Entrenando ControllerDemo");
            cd = ControllerDemo();
            cd.train(3000, 0.05);
            controller_trained = true;

            std::cout << GREEN << "Entrenado: " << RESET;
            std::cout << "accion(0.5,0.0) -> " << cd.predict(0.5,0.0) << "\n";
            pause();
        }

        //   4) PRUEBAS
        else if (op == 4) {

            if (!classifier_trained && !sequence_trained && !controller_trained) {
                std::cout << RED << "No hay modelos entrenados.\n" << RESET;
                pause();
                continue;
            }

            std::cout << "\n" << CYAN << "--- PRUEBAS ---\n" << RESET;

            if (classifier_trained)
                std::cout << "PatternClassifier: {0.1,0.9,0.1} -> "
                          << pc.predict({0.1,0.9,0.1}) << "\n";

            if (sequence_trained)
                std::cout << "SequencePredictor: [5,6,7] -> "
                          << sp.predict({5,6,7}) << "\n";

            if (controller_trained)
                std::cout << "Controller: accion(0.2,-0.1) -> "
                          << cd.predict(0.2,-0.1) << "\n";

            pause();
        }

        //   5) GUARDAR
        else if (op == 5) {
            if (classifier_trained) pc.save_model("pattern.model");
            if (sequence_trained)  sp.save_model("sequence.model");
            if (controller_trained) cd.save_model("controller.model");

            std::cout << GREEN << "Modelos guardados.\n" << RESET;
            pause();
        }

        //   6) CARGAR
        else if (op == 6) {

            if (file_exists("pattern.model")) {
                pc.load_model("pattern.model");
                classifier_trained = true;
            }

            if (file_exists("sequence.model")) {
                sp.load_model("sequence.model");
                sequence_trained = true;
            }

            if (file_exists("controller.model")) {
                cd.load_model("controller.model");
                controller_trained = true;
            }

            std::cout << GREEN << "Modelos cargados.\n" << RESET;
            pause();
        }

        //   7) PIPELINE
        else if (op == 7) {
            loader("Ejecutando Pipeline Experimental", 5, 150);

            std::cout << BLUE << "\n=== PIPELINE SECUENCIA ===\n" << RESET;
            run_sequence_pipeline();

            std::cout << BLUE << "\n=== PIPELINE CLASIFICACION ===\n" << RESET;
            run_pattern_pipeline();

            pause();
        }

        //  8) SALIR
        else if (op == 8) {
            std::cout << GREEN << "\nSaliendo...\n" << RESET;
            break;
        }
    }
}
