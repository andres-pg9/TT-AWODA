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

