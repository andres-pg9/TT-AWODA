import pytest
import numpy as np
from ia.pso import ParticleSwarmOptimizer
from ia.normalizacion import CONSUMO, REPORTES


class TestMetricasTiempoPSO:
    """
    Pruebas unitarias para verificar que las métricas de tiempo
    se calculan correctamente en el algoritmo PSO.
    """
    
    def test_pso_retorna_metricas_tiempo(self):
        """Verifica que PSO retorna métricas de tiempo en el resultado."""
        pso = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=5,
            seed=42
        )
        
        resultado = pso.optimize(CONSUMO, REPORTES, verbose=False)
        
        # Verificar que retorna 4 elementos (incluyendo métricas)
        assert len(resultado) == 4, "PSO debe retornar 4 elementos"
        
        pesos, utilidad, historial, metricas = resultado
        
        # Verificar estructura de métricas
        assert isinstance(metricas, dict), "Métricas debe ser un diccionario"
        assert "tiempo_total" in metricas, "Debe incluir tiempo_total"
        assert "tiempo_normalizacion" in metricas, "Debe incluir tiempo_normalizacion"
        assert "tiempo_inicializacion" in metricas, "Debe incluir tiempo_inicializacion"
        assert "tiempo_iteraciones" in metricas, "Debe incluir tiempo_iteraciones"
        assert "tiempo_promedio_por_iteracion" in metricas, "Debe incluir tiempo_promedio_por_iteracion"
    
    def test_metricas_tiempo_son_numericas(self):
        """Verifica que todas las métricas de tiempo son valores numéricos positivos."""
        pso = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=5,
            seed=42
        )
        
        _, _, _, metricas = pso.optimize(CONSUMO, REPORTES, verbose=False)
        
        # Verificar que todos los tiempos son números positivos
        assert metricas["tiempo_total"] > 0, "Tiempo total debe ser positivo"
        assert metricas["tiempo_normalizacion"] >= 0, "Tiempo normalización debe ser no negativo"
        assert metricas["tiempo_inicializacion"] > 0, "Tiempo inicialización debe ser positivo"
        assert metricas["tiempo_iteraciones"] > 0, "Tiempo iteraciones debe ser positivo"
        assert metricas["tiempo_promedio_por_iteracion"] > 0, "Tiempo promedio debe ser positivo"
    
    def test_tiempo_total_es_suma_de_componentes(self):
        """Verifica que el tiempo total es aproximadamente la suma de sus componentes."""
        pso = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=5,
            seed=42
        )
        
        _, _, _, metricas = pso.optimize(CONSUMO, REPORTES, verbose=False)
        
        suma_componentes = (
            metricas["tiempo_normalizacion"] +
            metricas["tiempo_inicializacion"] +
            metricas["tiempo_iteraciones"]
        )
        
        # Permitir pequeña diferencia por overhead de medición
        diferencia = abs(metricas["tiempo_total"] - suma_componentes)
        assert diferencia < 0.1, "Tiempo total debe ser aproximadamente suma de componentes"
    
    def test_tiempo_promedio_por_iteracion_es_correcto(self):
        """Verifica que el tiempo promedio por iteración se calcula correctamente."""
        pso = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=20,
            seed=42
        )
        
        _, _, _, metricas = pso.optimize(CONSUMO, REPORTES, verbose=False)
        
        promedio_calculado = metricas["tiempo_iteraciones"] / metricas["iteraciones_totales"]
        
        # Verificar cálculo del promedio
        assert abs(promedio_calculado - metricas["tiempo_promedio_por_iteracion"]) < 1e-6, \
            "Promedio por iteración debe calcularse correctamente"
    
    def test_metricas_incluyen_metadatos(self):
        """Verifica que las métricas incluyen metadatos de configuración."""
        n_particles = 15
        n_iterations = 25
        
        pso = ParticleSwarmOptimizer(
            n_particles=n_particles,
            n_iterations=n_iterations,
            seed=42
        )
        
        _, _, _, metricas = pso.optimize(CONSUMO, REPORTES, verbose=False)
        
        assert "iteraciones_totales" in metricas, "Debe incluir iteraciones_totales"
        assert "particulas_totales" in metricas, "Debe incluir particulas_totales"
        assert metricas["iteraciones_totales"] == n_iterations, "Iteraciones debe coincidir"
        assert metricas["particulas_totales"] == n_particles, "Partículas debe coincidir"
    
    def test_mas_iteraciones_toma_mas_tiempo(self):
        """Verifica que aumentar iteraciones incrementa el tiempo de ejecución."""
        pso_rapido = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=5,
            seed=42
        )
        
        pso_lento = ParticleSwarmOptimizer(
            n_particles=10,
            n_iterations=50,
            seed=42
        )
        
        _, _, _, metricas_rapido = pso_rapido.optimize(CONSUMO, REPORTES, verbose=False)
        _, _, _, metricas_lento = pso_lento.optimize(CONSUMO, REPORTES, verbose=False)
        
        assert metricas_lento["tiempo_iteraciones"] > metricas_rapido["tiempo_iteraciones"], \
            "Más iteraciones debe tomar más tiempo"
    
    def test_mas_particulas_toma_mas_tiempo(self):
        """Verifica que aumentar partículas incrementa el tiempo de ejecución."""
        pso_pocas = ParticleSwarmOptimizer(
            n_particles=5,
            n_iterations=10,
            seed=42
        )
        
        pso_muchas = ParticleSwarmOptimizer(
            n_particles=50,
            n_iterations=10,
            seed=42
        )
        
        _, _, _, metricas_pocas = pso_pocas.optimize(CONSUMO, REPORTES, verbose=False)
        _, _, _, metricas_muchas = pso_muchas.optimize(CONSUMO, REPORTES, verbose=False)
        
        assert metricas_muchas["tiempo_total"] > metricas_pocas["tiempo_total"], \
            "Más partículas debe tomar más tiempo"
    
    def test_tiempo_normalizacion_es_constante(self):
        """Verifica que el tiempo de normalización no depende de parámetros PSO."""
        pso1 = ParticleSwarmOptimizer(n_particles=10, n_iterations=5, seed=42)
        pso2 = ParticleSwarmOptimizer(n_particles=50, n_iterations=50, seed=42)
        
        _, _, _, metricas1 = pso1.optimize(CONSUMO, REPORTES, verbose=False)
        _, _, _, metricas2 = pso2.optimize(CONSUMO, REPORTES, verbose=False)
        
        # Tiempo de normalización debe ser similar (datos iguales)
        diferencia = abs(metricas1["tiempo_normalizacion"] - metricas2["tiempo_normalizacion"])
        assert diferencia < 0.01, "Tiempo de normalización debe ser independiente de PSO"


class TestMetricasTiempoIntegracion:
    """
    Pruebas de integración para verificar que las métricas se propagan correctamente.
    """
    
    def test_consistencia_de_resultados_con_metricas(self):
        """Verifica que agregar métricas no afecta los resultados de optimización."""
        pso = ParticleSwarmOptimizer(
            n_particles=30,
            n_iterations=20,
            seed=42
        )
        
        pesos, utilidad, historial, metricas = pso.optimize(CONSUMO, REPORTES, verbose=False)
        
        # Verificar que los resultados principales siguen siendo válidos
        assert len(pesos) == 4, "Debe retornar 4 pesos"
        assert sum(pesos) == pytest.approx(1.0, abs=1e-6), "Pesos deben sumar 1"
        assert all(p >= 0 for p in pesos), "Todos los pesos deben ser no negativos"
        assert isinstance(utilidad, dict), "Utilidad debe ser un diccionario"
        assert len(historial) == 20, "Historial debe tener 20 iteraciones"
        
        # Métricas no deben afectar la optimización
        assert metricas["tiempo_total"] > 0, "Debe tener tiempo de ejecución"
    
    def test_reproducibilidad_con_seed(self):
        """Verifica que las métricas no afectan la reproducibilidad con seed."""
        # Configuración idéntica
        config = {"n_particles": 20, "n_iterations": 10, "seed": 42}
        
        # Primera ejecución
        np.random.seed(42)
        pso1 = ParticleSwarmOptimizer(**config)
        pesos1, _, _, _ = pso1.optimize(CONSUMO, REPORTES, verbose=False)
        
        # Segunda ejecución con reset de seed
        np.random.seed(42)
        pso2 = ParticleSwarmOptimizer(**config)
        pesos2, _, _, _ = pso2.optimize(CONSUMO, REPORTES, verbose=False)
        
        # Los pesos deben ser idénticos con el mismo seed
        np.testing.assert_array_almost_equal(pesos1, pesos2, decimal=10)
