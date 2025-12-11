"""
Script de demostración de métricas de tiempo de ejecución.

Ejecuta el algoritmo PSO con diferentes configuraciones y muestra
las métricas de tiempo para evaluar el rendimiento.
"""

import sys
import time
from ia.pso import ParticleSwarmOptimizer
from ia.normalizacion import CONSUMO, REPORTES


def formatear_tiempo(segundos):
    """Formatea tiempo en segundos a formato legible."""
    if segundos < 0.001:
        return f"{segundos * 1000000:.2f}µs"
    elif segundos < 1:
        return f"{segundos * 1000:.2f}ms"
    else:
        return f"{segundos:.4f}s"


def ejecutar_prueba(nombre, n_particles, n_iterations, seed=None):
    """Ejecuta una prueba de PSO y muestra métricas."""
    print(f"\n{'='*70}")
    print(f"PRUEBA: {nombre}")
    print(f"{'='*70}")
    print(f"Configuración:")
    print(f"  - Partículas: {n_particles}")
    print(f"  - Iteraciones: {n_iterations}")
    print(f"  - Seed: {seed if seed else 'Aleatorio'}")
    print(f"-"*70)
    
    # Crear optimizador
    pso = ParticleSwarmOptimizer(
        n_particles=n_particles,
        n_iterations=n_iterations,
        seed=seed
    )
    
    # Ejecutar optimización
    inicio = time.perf_counter()
    pesos, resultado, historial, metricas = pso.optimize(
        CONSUMO, 
        REPORTES, 
        verbose=False
    )
    tiempo_total_medido = time.perf_counter() - inicio
    
    # Mostrar resultados
    print(f"\nRESULTADOS:")
    print(f"  Utilidad total: {resultado['utilidad_total']:.2f}")
    print(f"  Pesos óptimos:")
    print(f"    α (legal):    {pesos[0]:.6f}")
    print(f"    β (social):   {pesos[1]:.6f}")
    print(f"    γ (consumo):  {pesos[2]:.6f}")
    print(f"    δ (reportes): {pesos[3]:.6f}")
    
    print(f"\nMÉTRICAS DE TIEMPO:")
    print(f"  Normalización:     {formatear_tiempo(metricas['tiempo_normalizacion'])}")
    print(f"  Inicialización:    {formatear_tiempo(metricas['tiempo_inicializacion'])}")
    print(f"  Iteraciones:       {formatear_tiempo(metricas['tiempo_iteraciones'])}")
    print(f"  Por iteración:     {formatear_tiempo(metricas['tiempo_promedio_por_iteracion'])}")
    print(f"  Total PSO:         {formatear_tiempo(metricas['tiempo_total'])}")
    print(f"  Total medido:      {formatear_tiempo(tiempo_total_medido)}")
    
    # Calcular porcentajes
    tiempo_base = metricas['tiempo_total']
    print(f"\nDISTRIBUCIÓN DEL TIEMPO:")
    print(f"  Normalización:   {(metricas['tiempo_normalizacion']/tiempo_base)*100:5.2f}%")
    print(f"  Inicialización:  {(metricas['tiempo_inicializacion']/tiempo_base)*100:5.2f}%")
    print(f"  Iteraciones:     {(metricas['tiempo_iteraciones']/tiempo_base)*100:5.2f}%")
    
    # Calcular throughput
    evaluaciones_totales = n_particles * n_iterations
    evaluaciones_por_segundo = evaluaciones_totales / metricas['tiempo_iteraciones']
    
    print(f"\nRENDIMIENTO:")
    print(f"  Evaluaciones totales:      {evaluaciones_totales:,}")
    print(f"  Evaluaciones por segundo:  {evaluaciones_por_segundo:,.0f}")
    
    return metricas


def comparar_configuraciones():
    """Compara diferentes configuraciones de PSO."""
    configuraciones = [
        ("Rápida (10 partículas, 50 iter)", 10, 50, 42),
        ("Estándar (30 partículas, 150 iter)", 30, 150, 42),
        ("Intensiva (50 partículas, 200 iter)", 50, 200, 42),
    ]
    
    print("\n" + "="*70)
    print("COMPARACIÓN DE CONFIGURACIONES")
    print("="*70)
    
    resultados = []
    for nombre, particulas, iteraciones, seed in configuraciones:
        metricas = ejecutar_prueba(nombre, particulas, iteraciones, seed)
        resultados.append((nombre, metricas))
    
    # Tabla comparativa
    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA")
    print(f"{'='*70}")
    print(f"{'Configuración':<40} {'Tiempo Total':>15} {'Calidad':>10}")
    print(f"{'-'*70}")
    
    for nombre, metricas in resultados:
        tiempo = formatear_tiempo(metricas['tiempo_total'])
        print(f"{nombre:<40} {tiempo:>15}")
    
    print(f"{'='*70}")


def analizar_escalabilidad():
    """Analiza cómo escala el tiempo con diferentes parámetros."""
    print(f"\n{'='*70}")
    print("ANÁLISIS DE ESCALABILIDAD")
    print(f"{'='*70}")
    
    print("\n1. Escalabilidad con ITERACIONES (30 partículas):")
    print(f"{'-'*70}")
    print(f"{'Iteraciones':>12} {'Tiempo Total':>15} {'Tiempo/Iter':>15}")
    print(f"{'-'*70}")
    
    for n_iter in [10, 50, 100, 150]:
        pso = ParticleSwarmOptimizer(n_particles=30, n_iterations=n_iter, seed=42)
        _, _, _, metricas = pso.optimize(CONSUMO, REPORTES, verbose=False)
        print(f"{n_iter:>12} {formatear_tiempo(metricas['tiempo_total']):>15} "
              f"{formatear_tiempo(metricas['tiempo_promedio_por_iteracion']):>15}")
    
    print(f"\n2. Escalabilidad con PARTÍCULAS (100 iteraciones):")
    print(f"{'-'*70}")
    print(f"{'Partículas':>12} {'Tiempo Total':>15} {'Tiempo/Iter':>15}")
    print(f"{'-'*70}")
    
    for n_part in [10, 20, 30, 50]:
        pso = ParticleSwarmOptimizer(n_particles=n_part, n_iterations=100, seed=42)
        _, _, _, metricas = pso.optimize(CONSUMO, REPORTES, verbose=False)
        print(f"{n_part:>12} {formatear_tiempo(metricas['tiempo_total']):>15} "
              f"{formatear_tiempo(metricas['tiempo_promedio_por_iteracion']):>15}")


def main():
    """Función principal."""
    print("\n" + "="*70)
    print(" DEMOSTRACIÓN DE MÉTRICAS DE TIEMPO - PSO ")
    print("="*70)
    
    if len(sys.argv) > 1:
        modo = sys.argv[1]
        
        if modo == "rapido":
            ejecutar_prueba("Prueba Rápida", 10, 20, 42)
        elif modo == "estandar":
            ejecutar_prueba("Configuración Estándar", 30, 150, 42)
        elif modo == "comparar":
            comparar_configuraciones()
        elif modo == "escalabilidad":
            analizar_escalabilidad()
        else:
            print(f"\nModo '{modo}' no reconocido.")
            print("\nModos disponibles:")
            print("  rapido        - Prueba rápida (10 partículas, 20 iteraciones)")
            print("  estandar      - Configuración estándar (30 partículas, 150 iteraciones)")
            print("  comparar      - Comparar múltiples configuraciones")
            print("  escalabilidad - Análisis de escalabilidad")
            return 1
    else:
        # Por defecto, ejecutar comparación
        comparar_configuraciones()
        analizar_escalabilidad()
    
    print(f"\n{'='*70}")
    print("Demostración completada exitosamente")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
