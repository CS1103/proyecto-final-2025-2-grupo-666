[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**
Este proyecto consiste en la implementación desde cero de un framework de Deep Learning en C++20 moderno. El sistema se construye sobre una librería de tensores genérica, permitiendo la creación, entrenamiento y serialización de redes neuronales multicapa.

Como demostración de la capacidad de generalización del framework, se implementan tres casos de uso prácticos: clasificación de patrones geométricos, predicción de series numéricas y un controlador para un entorno simulado de física (EnvGym), optimizados mediante algoritmos de búsqueda de hiperparámetros.

### Contenidos
1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Estructura del Proyecto (Epics)](#estructura-del-proyecto-epics)
4. [Investigación teórica](#1-investigación-teórica)
5. [Diseño e implementación](#2-diseño-e-implementación)
6. [Manual de Uso](#3-manual-de-uso)
7. [Análisis del rendimiento y Pipelines](#4-análisis-del-rendimiento-y-pipelines)
8. [Trabajo en equipo](#5-trabajo-en-equipo)
9. [Conclusiones](#6-conclusiones)
10. [Bibliografía](#7-bibliografía)

---

### Datos generales
* **Curso**: Programación III
* **Tema**: Redes Neuronales y Algoritmos de Optimización en C++
* **Grupo**: `[NOMBRE_DE_TU_GRUPO]`
* **Integrantes**:

| Código | Alumno | Rol Principal |
|:------:|:-------|:--------------|
| 209900X | Alumno A | Desarrollo del Core (Tensores y NN) |
| 209900Y | Alumno B | Implementación de Optimizadores y Loss |
| 209900Z | Alumno C | Desarrollo de Aplicaciones (EnvGym, Patterns) |
| 209900W | Alumno D | Pipelines de Entrenamiento y Pruebas |
| 209900Q | Alumno E | Documentación, UI y Video Demo |

> *Nota: Se utilizó GitHub para el control de versiones y GitLab para la gestión del proyecto (Kanban).*

---

### Requisitos e instalación

El proyecto hace uso intensivo de características modernas de C++20 (concepts, templates variádicos, smart pointers).

1. **Compilador**: GCC 10+ o Clang 12+ (Soporte C++20 requerido).
2. **Sistema de Construcción**: CMake 3.18 o superior.
3. **Dependencias**:
   * Biblioteca Estándar (STL).
   * **No se utilizan librerías externas** para álgebra lineal; toda la matemática se maneja con la clase propia `utec::algebra::Tensor`.

4. **Instalación y Compilación**:

   ```bash
   # Clonar el repositorio
   git clone [https://github.com/TU_USUARIO/proyecto-nn-cpp.git](https://github.com/TU_USUARIO/proyecto-nn-cpp.git)
   cd proyecto-nn-cpp

   # Crear directorio de construcción
   mkdir build && cd build

   # Configurar CMake
   cmake ..

   # Compilar
   make
   ```
---

### 1. Investigación teórica

Para el desarrollo de este motor de inteligencia artificial se investigaron los siguientes conceptos fundamentales:

1.  **Tensores y Broadcasting**: Estructuras de datos esenciales para manejar pesos y sesgos en dimensiones arbitrarias. Se estudió cómo las librerías como PyTorch o NumPy manejan la expansión de dimensiones automática.
2.  **Forward & Backward Propagation**:
    * *Forward*: El flujo de información desde la entrada hasta la salida.
    * *Backward*: La aplicación de la regla de la cadena para calcular gradientes parciales de la función de pérdida con respecto a los pesos.
3.  **Funciones de Activación**:
    * **ReLU**: Crucial para redes profundas, evitando el problema del desvanecimiento del gradiente.
    * **Softmax**: Utilizada en la capa de salida para convertir logits en probabilidades sumables a 1.
4.  **Optimización Estocástica**:
    * **Adam**: Se investigó su algoritmo de momentos (media y varianza de los gradientes) para lograr una convergencia más rápida y estable que el descenso de gradiente estándar.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de Clases
El diseño sigue principios SOLID, utilizando polimorfismo y templates de C++20.

* **`utec::algebra::Tensor<T, DIMS>`**: El núcleo matemático. Gestiona la memoria y las operaciones algebraicas.
* **`utec::neural_network::ILayer`**: Interfaz base para todas las capas. Define `forward`, `backward`, `update_params` y métodos de serialización.
* **`utec::neural_network::IOptimizer`**: Interfaz para estrategias de actualización de pesos (`SGD`, `Adam`).
* **`utec::neural_network::ILoss`**: Abstracción para calcular el error y su gradiente.

* **Estructura de carpetas**:

  ```
  proyecto-final/
  ├── include/
  │   └── utec/
  │       ├── algebra/
  │       │   └── tensor.h          # Core matemático
  │       ├── nn/
  │       │   ├── neural_network.h  # Clase contenedora principal
  │       │   ├── nn_dense.h        # Capa densa
  │       │   ├── nn_activation.h   # ReLU, Sigmoid, Softmax
  │       │   ├── nn_optimizer.h    # Adam, SGD
  │       │   └── nn_loss.h         # MSE, CrossEntropy
  │       └── apps/
  │           ├── ControllerDemo.h  # Lógica del controlador (Epic 3)
  │           ├── PatternClassifier.h
  │           ├── SequencePredictor.h
  │           ├── TrainingPipeline.h # Búsqueda de hiperparámetros
  │           ├── EnvGym.h          # Entorno de simulación
  |           └── Interfaz.h                 
  ├── src/
  |     └── utec/
  |          └── apps/
  │               ├── ControllerDemo.cpp  # Lógica del controlador (Epic 3)
  │               ├── PatternClassifier.cpp
  │               ├── SequencePredictor.cpp
  │               ├── TrainingPipeline.cpp # Búsqueda de hiperparámetros
  │               ├── EnvGym.cpp          # Entorno de simulación
  |               └── Interfaz.cpp  
  |         
  ├── tests/
  |    ├── test_applications.cpp
  |    ├── test_neural_network.cpp
  |    └── test_tensor.cpp
  ├── docs/
  |    ├── BIBLIOGRAFIA.md
  |    └── README.md
  └──main.cpp                        # Punto de entrada
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
