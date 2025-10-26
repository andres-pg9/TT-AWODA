import numpy as np
from typing import List, Dict, Tuple
from funciones import calcular_utilidad

# ============================================================================
# CLASE PARTICLE SWARM OPTIMIZER
# ============================================================================

class ParticleSwarmOptimizer:
    """
    Optimizador basado en Particle Swarm Optimization (PSO).

    PSO es un algoritmo de optimización inspirado en el comportamiento social
    de bandadas de aves o cardúmenes de peces. Un conjunto de "partículas"
    explora el espacio de búsqueda, compartiendo información sobre las mejores
    soluciones encontradas.

    En este caso, buscamos los mejores pesos (α, β, γ, δ) que maximicen
    la función de utilidad multiobjetivo.

    Parámetros:
        n_particles: Número de partículas (soluciones candidatas)
        n_iterations: Número de iteraciones de optimización
        w: Peso de inercia (controla velocidad actual)
        c1: Coeficiente cognitivo (atracción a mejor personal)
        c2: Coeficiente social (atracción a mejor global)
        seed: Semilla aleatoria para reproducibilidad

    Restricción: α + β + γ + δ = 1 (normalización de pesos)
    """

    def __init__(self, n_particles: int = 30, n_iterations: int = 150,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 seed: int = None):
        """
        Inicializa el optimizador PSO.

        Args:
            n_particles: Tamaño del enjambre
            n_iterations: Iteraciones de optimización
            w: Peso de inercia (típicamente 0.4-0.9)
            c1: Factor cognitivo (típicamente 1.5-2.0)
            c2: Factor social (típicamente 1.5-2.0)
            seed: Semilla para reproducibilidad
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inercia
        self.c1 = c1  # Componente cognitivo
        self.c2 = c2  # Componente social
        self.seed = seed
        self.history = []

        if seed is not None:
            np.random.seed(seed)

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, Dict, List]:
        """
        Ejecuta el algoritmo PSO para encontrar pesos óptimos.

        Proceso:
        1. Inicializar enjambre con pesos aleatorios (suma = 1)
        2. Evaluar fitness de cada partícula
        3. Para cada iteración:
           - Actualizar velocidad basada en mejor personal y global
           - Actualizar posición
           - Evaluar nuevas posiciones
           - Actualizar mejores soluciones
        4. Retornar mejor solución encontrada

        Args:
            verbose: Si True, muestra progreso

        Returns:
            Tupla con:
            - Mejor posición encontrada (pesos α,β,γ,δ)
            - Resultado de utilidad de esa posición
            - Historial de optimización por iteración
        """

        if verbose:
            print("\nIniciando optimización PSO...")
            print(f"Partículas: {self.n_particles} | Iteraciones: {self.n_iterations}")
            print(f"Parámetros: w={self.w}, c1={self.c1}, c2={self.c2}")
            print("="*70)

        # PASO 1: Inicialización del enjambre
        # Usar distribución de Dirichlet para asegurar que suma = 1
        # Cada posición es un vector de 4 pesos que suman 1
        positions = np.random.dirichlet(np.ones(4), self.n_particles)
        velocities = np.random.randn(self.n_particles, 4) * 0.1

        # PASO 2: Evaluar fitness inicial
        fitness = np.array([
            calcular_utilidad(*pos)['utilidad_total'] for pos in positions
        ])

        # PASO 3: Inicializar mejores posiciones
        # Mejor personal (pbest): mejor posición que ha visitado cada partícula
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()

        # Mejor global (gbest): mejor posición encontrada por todo el enjambre
        global_best_idx = np.argmax(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        global_best_result = calcular_utilidad(*global_best_position)

        return None