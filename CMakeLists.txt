cmake_minimum_required(VERSION 3.15)
project(proyecto_final_2025_2_grupo_666)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================
#           INCLUDES
# ============================
include_directories(include)

# ============================
#    FUENTES DEL PROYECTO
# ============================
file(GLOB_RECURSE PROJECT_SOURCES
        src/*.cpp
)

# ============================
#    EJECUTABLE PRINCIPAL
# ============================
add_executable(pong_ai
        main.cpp
        ${PROJECT_SOURCES}
        include/utec/apps/TrainingPipeline.h
        src/utec/apps/TrainingPipeline.cpp
        src/utec/apps/Interfaz.cpp
        include/utec/apps/Interfaz.h
)

# ============================
#      TEST 1 — Tensor
# ============================
add_executable(test_tensor
        tests/test_tensor.cpp
)
target_include_directories(test_tensor PRIVATE include)

# ============================
#   TEST 2 — Neural Network
# ============================
add_executable(test_nn
        tests/test_neural_network.cpp
        ${PROJECT_SOURCES}
)
target_include_directories(test_nn PRIVATE include)

# ============================
#    TEST 3 — Applications
# ============================
add_executable(test_apps
        tests/test_applications.cpp
        ${PROJECT_SOURCES}
)
target_include_directories(test_apps PRIVATE include)
