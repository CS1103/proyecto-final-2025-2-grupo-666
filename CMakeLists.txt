cmake_minimum_required(VERSION 3.15)
project(proyecto_final_2025_2_grupo_666)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluye las cabeceras
include_directories(include)

# Busca todos los archivos .cpp en src/
file(GLOB_RECURSE SOURCES
        src/*.cpp
)

add_executable(pong_ai ${SOURCES}
        include/utec/apps/EnvGym.h
        src/utec/apps/EnvGym.cpp)
