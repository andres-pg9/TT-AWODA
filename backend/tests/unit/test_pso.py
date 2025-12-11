"""
Pruebas unitarias para el algoritmo PSO (ia/pso.py).

Clase testeada:
    - ParticleSwarmOptimizer: Optimizador de enjambre de partículas

Tests incluidos:
    1. test_pso_inicializacion: Parámetros correctos al inicializar
    2. test_pso_reproducibilidad: Mismo seed produce mismos resultados
    3. test_pso_restriccion_suma_pesos: α + β + γ + δ = 1.0
    4. test_pso_pesos_positivos: Todos los pesos son >= 0
    5. test_pso_convergencia: El algoritmo converge (mejora fitness)
    6. test_pso_historial: El historial registra todas las iteraciones
    7. test_pso_resultado_valido: La solución final es válida
    8. test_pso_mejora_fitness: El fitness mejora o se mantiene
"""

import pytest
import numpy as np
from ia.pso import ParticleSwarmOptimizer
from ia.funciones import calcular_utilidad
from ia.normalizacion import normalizar_valores


# ============================================================================
# TESTS - Inicialización y Configuración
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
def test_pso_inicializacion():
    """
    Verifica que el PSO se inicialice correctamente con los parámetros dados.
    
    Comportamiento esperado:
        - Los parámetros se asignan correctamente
        - El historial inicia vacío
        - No se lanza ninguna excepción
    """
    pso = ParticleSwarmOptimizer(
        n_particles=20,
        n_iterations=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=42
    )
    
    assert pso.n_particles == 20, "n_particles debe ser 20"
    assert pso.n_iterations == 100, "n_iterations debe ser 100"
    assert pso.w == 0.7, "w debe ser 0.7"
    assert pso.c1 == 1.5, "c1 debe ser 1.5"
    assert pso.c2 == 1.5, "c2 debe ser 1.5"
    assert pso.seed == 42, "seed debe ser 42"
    assert pso.history == [], "El historial debe iniciar vacío"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
def test_pso_reproducibilidad(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que usar el mismo seed produzca resultados reproducibles.
    
    Comportamiento esperado:
        - Dos ejecuciones con el mismo seed deben dar el mismo resultado
        - Los pesos finales deben ser idénticos
    """
    seed = 42
    
    # Primera ejecución
    pso1 = ParticleSwarmOptimizer(n_particles=10, n_iterations=30, seed=seed)
    pesos1, resultado1, _, _ = pso1.optimize(datos_consumo_prueba, datos_reportes_prueba)
    
    # Segunda ejecución con el mismo seed
    pso2 = ParticleSwarmOptimizer(n_particles=10, n_iterations=30, seed=seed)
    pesos2, resultado2, _, _ = pso2.optimize(datos_consumo_prueba, datos_reportes_prueba)
    
    # Verificar que los pesos son idénticos
    assert np.allclose(pesos1, pesos2, atol=1e-6), \
        f"Los pesos deben ser idénticos con el mismo seed.\nPesos1: {pesos1}\nPesos2: {pesos2}"
    
    # Verificar que las utilidades son idénticas
    assert np.isclose(resultado1['utilidad_total'], resultado2['utilidad_total'], atol=1e-6), \
        f"Las utilidades deben ser idénticas con el mismo seed"


# ============================================================================
# TESTS - Restricciones del Algoritmo
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
def test_pso_restriccion_suma_pesos(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que los pesos óptimos siempre sumen 1.0.
    
    Comportamiento esperado:
        - α + β + γ + δ = 1.0 (restricción fundamental)
        - Esto debe cumplirse para cualquier configuración del PSO
    """
    pso = ParticleSwarmOptimizer(n_particles=15, n_iterations=50, seed=123)
    pesos_optimos, _, _, _ = pso.optimize(datos_consumo_prueba, datos_reportes_prueba)
    
    suma_pesos = np.sum(pesos_optimos)
    
    assert np.isclose(suma_pesos, 1.0, atol=1e-6), \
        f"La suma de los pesos debe ser 1.0, pero es {suma_pesos}"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
def test_pso_pesos_positivos(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que todos los pesos sean no negativos.
    
    Comportamiento esperado:
        - α, β, γ, δ >= 0 (no tiene sentido tener pesos negativos)
        - El PSO usa np.abs() para garantizar esto
    """
    pso = ParticleSwarmOptimizer(n_particles=15, n_iterations=50, seed=456)
    pesos_optimos, _, _, _ = pso.optimize(datos_consumo_prueba, datos_reportes_prueba)
    
    assert all(peso >= 0 for peso in pesos_optimos), \
        f"Todos los pesos deben ser no negativos, pero son: {pesos_optimos}"


# ============================================================================
# TESTS - Convergencia y Optimización
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
@pytest.mark.slow
def test_pso_convergencia(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que el algoritmo PSO converja a una solución.
    
    Comportamiento esperado:
        - El fitness de la mejor solución debe mejorar o mantenerse
        - Al final, el fitness debe ser razonable (> 0)
    """
    pso = ParticleSwarmOptimizer(n_particles=20, n_iterations=100, seed=789)
    pesos_optimos, resultado, historial, _ = pso.optimize(datos_consumo_prueba, datos_reportes_prueba)
    
    # Verificar que hay un historial de optimización
    assert len(historial) == 100, f"Debe haber 100 registros en el historial"
    
    # Verificar que el fitness final es positivo
    fitness_final = resultado['utilidad_total']
    assert fitness_final > 0, \
        f"El fitness final debe ser positivo, pero es {fitness_final}"
    
    # Verificar que el fitness mejora o se mantiene a lo largo de las iteraciones
    fitness_inicial = historial[0]['best_fitness']
    fitness_medio = historial[len(historial)//2]['best_fitness']
    
    assert fitness_medio >= fitness_inicial, \
        f"El fitness debe mejorar: inicial={fitness_inicial:.2f}, medio={fitness_medio:.2f}"
    assert fitness_final >= fitness_medio, \
        f"El fitness debe mejorar: medio={fitness_medio:.2f}, final={fitness_final:.2f}"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
def test_pso_mejora_fitness(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que el fitness mejore comparado con una solución aleatoria inicial.
    
    Comportamiento esperado:
        - La solución del PSO debe ser mejor que pesos aleatorios
        - El PSO debe explorar efectivamente el espacio de búsqueda
    """
    # Solución aleatoria inicial
    np.random.seed(999)
    pesos_aleatorios = np.random.dirichlet(np.ones(4))
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    fitness_aleatorio = calcular_utilidad(*pesos_aleatorios, 
                                         consumo_norm=consumo_norm,
                                         reportes_norm=reportes_norm)['utilidad_total']
    
    # Solución optimizada con PSO
    pso = ParticleSwarmOptimizer(n_particles=20, n_iterations=80, seed=999)
    pesos_optimos, resultado_pso, _, _ = pso.optimize(datos_consumo_prueba, datos_reportes_prueba)
    fitness_pso = resultado_pso['utilidad_total']
    
    # El PSO debe mejorar o igualar la solución aleatoria
    assert fitness_pso >= fitness_aleatorio * 0.95, \
        f"El PSO debe mejorar la solución aleatoria.\n" \
        f"Fitness aleatorio: {fitness_aleatorio:.2f}\n" \
        f"Fitness PSO: {fitness_pso:.2f}"


# ============================================================================
# TESTS - Historial y Resultados
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
def test_pso_historial(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que el historial registre correctamente todas las iteraciones.
    
    Comportamiento esperado:
        - Debe haber un registro por cada iteración
        - Cada registro debe tener las claves esperadas
        - Los valores deben ser coherentes
    """
    pso = ParticleSwarmOptimizer(n_particles=10, n_iterations=25, seed=111)
    _, _, historial, _ = pso.optimize(datos_consumo_prueba, datos_reportes_prueba)
    
    # Verificar longitud del historial
    assert len(historial) == 25, \
        f"El historial debe tener 25 registros, pero tiene {len(historial)}"
    
    # Verificar estructura de cada registro
    claves_esperadas = {'iteration', 'best_fitness', 'mean_fitness', 'std_fitness',
                       'alpha', 'beta', 'gamma', 'delta'}
    
    for i, registro in enumerate(historial):
        assert set(registro.keys()) == claves_esperadas, \
            f"El registro {i} no tiene las claves esperadas"
        
        # Verificar que los pesos suman 1
        suma = registro['alpha'] + registro['beta'] + registro['gamma'] + registro['delta']
        assert np.isclose(suma, 1.0, atol=1e-6), \
            f"En iteración {i}, la suma de pesos es {suma}, debe ser 1.0"
        
        # Verificar que el número de iteración es correcto
        assert registro['iteration'] == i, \
            f"El número de iteración debe ser {i}, pero es {registro['iteration']}"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.pso
def test_pso_resultado_valido(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que la solución final sea válida y coherente.
    
    Comportamiento esperado:
        - Los pesos deben estar en [0, 1] y sumar 1
        - El resultado debe tener todas las componentes de utilidad
        - El resultado debe ser reproducible con los pesos finales
    """
    pso = ParticleSwarmOptimizer(n_particles=15, n_iterations=40, seed=222)
    pesos_optimos, resultado_pso, _, _ = pso.optimize(datos_consumo_prueba, datos_reportes_prueba)
    
    # Verificar pesos en rango [0, 1]
    assert all(0 <= p <= 1 for p in pesos_optimos), \
        f"Los pesos deben estar en [0, 1]: {pesos_optimos}"
    
    # Verificar que el resultado tiene las claves esperadas
    claves_esperadas = {'utilidad_total', 'equidad', 'satisfaccion_social',
                       'cumplimiento_legal', 'atencion_consumo', 'atencion_reportes',
                       'coeficiente_gini'}
    assert set(resultado_pso.keys()) == claves_esperadas, \
        "El resultado no tiene las claves esperadas"
    
    # Reproducir el cálculo manualmente
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    resultado_manual = calcular_utilidad(*pesos_optimos,
                                        consumo_norm=consumo_norm,
                                        reportes_norm=reportes_norm)
    
    # Verificar que coinciden
    assert np.isclose(resultado_pso['utilidad_total'], 
                     resultado_manual['utilidad_total'], atol=1e-3), \
        f"El resultado del PSO debe coincidir con el cálculo manual"
