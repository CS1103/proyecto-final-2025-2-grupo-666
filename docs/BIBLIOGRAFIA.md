# Bibliografía

[1] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.  
Obra fundamental en el campo de las redes neuronales modernas. Aporta el marco teórico utilizado para comprender los fundamentos, arquitecturas, funciones de activación, métodos de optimización y principios de generalización que guiaron el diseño de la red neuronal implementada en este proyecto.

[2] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2015.  
Referencia clave para el estudio de métodos de optimización estocástica. Aunque el proyecto implementa SGD, hill-climbing y búsqueda aleatoria, este trabajo sirve como marco conceptual para comprender optimizadores modernos y situar el proyecto dentro del contexto actual del aprendizaje profundo.

[3] B. Stroustrup, *The C++ Programming Language*, 4th ed. Boston, MA, USA: Addison-Wesley, 2013.  
Texto oficial del creador de C++. Su relevancia para este proyecto radica en el uso intensivo de características modernas del lenguaje, diseño orientado a objetos, manejo de memoria, plantillas, abstracciones y estructuras eficientes utilizadas en la implementación del motor tensorial, la red neuronal y los módulos del sistema.

[4] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” *Nature*, vol. 323, no. 6088, pp. 533–536, 1986.  
Artículo histórico que introduce el algoritmo de retropropagación del error. Constituye la base matemática principal para el entrenamiento de redes multicapa, fundamento esencial para el desarrollo de la arquitectura y funciones de entrenamiento implementadas en este proyecto.

---

## Referencias adicionales relacionadas con la implementación

[5] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, 2010.  
Obra clásica de inteligencia artificial que describe métodos de optimización basados en búsqueda local como hill-climbing, simulated annealing y random search, técnicas empleadas en el módulo de *pipeline* experimental para el ajuste de parámetros del modelo.

[6] D. H. Wolpert and W. G. Macready, “No free lunch theorems for optimization,” *IEEE Trans. Evol. Comput.*, vol. 1, no. 1, pp. 67–82, 1997.  
Trabajo fundamental que contextualiza el uso de algoritmos sencillos de optimización (como búsqueda aleatoria y hill-climbing) mostrando que no existe un método universalmente superior, lo que justifica su aplicación práctica en problemas específicos como los abordados en este proyecto.

[7] G. Brockman *et al.*, “OpenAI Gym,” *arXiv preprint arXiv:1606.01540*, 2016.  
Referencia oficial de Gym, utilizada como inspiración para el diseño del entorno `EnvGym`, donde se modela un sistema dinámico simplificado con estados (posición, velocidad) y acciones discretas para evaluar el desempeño del controlador basado en la red neuronal.

[8] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.  
Proporciona el marco teórico para entender entornos de control, dinámica de estados y política de acciones, conceptos relacionados con la aplicación de “control simplificado” desarrollada sobre la red y el entorno `EnvGym`.

[9] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed. Baltimore, MD, USA: Johns Hopkins University Press, 2013.  
Referencia clásica en álgebra lineal numérica, directamente relacionada con las operaciones implementadas en la clase `Tensor` (productos punto, combinaciones lineales, operaciones elemento a elemento) que constituyen la base numérica del modelo.

[10] J. R. Gilbert, C. Moler, and R. Schreiber, “Sparse matrices in MATLAB: Design and implementation,” *SIAM J. Matrix Anal. Appl.*, vol. 13, no. 1, pp. 333–356, 1992.  
Trabajo relevante para el diseño y manejo eficiente de estructuras matriciales, que sirve como inspiración conceptual para la implementación de tensores y operaciones optimizadas en el proyecto.

[11] E. Gamma, R. Helm, R. Johnson, and J. Vlissides, *Design Patterns: Elements of Reusable Object-Oriented Software*. Reading, MA, USA: Addison-Wesley, 1994.  
Fundamento teórico del diseño orientado a objetos aplicado en la separación de responsabilidades del proyecto: capas de red, optimizadores, aplicaciones, entorno y utilidades, siguiendo principios de reutilización y extensibilidad.

[12] R. C. Martin, *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Upper Saddle River, NJ, USA: Prentice Hall, 2017.  
Apoya las decisiones de arquitectura limpia: separación de módulos en `include/`, `src/`, `apps/`, `tests/`, y la clara división entre lógica de negocio (red neuronal, tensor), presentación (interfaz) y experimentación (pipeline).

[13] G. J. Myers, C. Sandler, and T. Badgett, *The Art of Software Testing*, 3rd ed. Hoboken, NJ, USA: Wiley, 2011.  
Referencia clave sobre técnicas de pruebas de software. Justifica el diseño de los tests unitarios incluidos para tensores, red neuronal y aplicaciones, así como la importancia de cubrir múltiples casos y escenarios de uso.

[14] A. Williams, *C++ Concurrency in Action*. Shelter Island, NY, USA: Manning, 2019.  
Obra que respalda el uso de características de concurrencia y temporización en C++ (por ejemplo, `std::thread`, `std::this_thread::sleep_for`), utilizadas en la interfaz del programa para mejorar la experiencia de usuario y la interacción con las diferentes demos.

---

> **Nota:** Las referencias seleccionadas combinan fundamentos teóricos de redes neuronales, aprendizaje profundo, optimización, diseño de software y programación en C++, alineándose con las decisiones de diseño, implementación y experimentación realizadas en este proyecto. Proveen un sustento académico sólido y actual.
