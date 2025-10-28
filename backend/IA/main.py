from pso import ParticleSwarmOptimizer
from resultados import imprimir_resultados_detallados

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*80)
    print("SISTEMA DE OPTIMIZACIÓN DE DISTRIBUCIÓN DE AGUA - CDMX")
    print("Alcaldía Gustavo A. Madero")
    print("Algoritmo: Particle Swarm Optimization (PSO)")
    print("Función de Utilidad: 5 Componentes")
    print("  • Equidad (25%) • Social (30%) • Legal (25%)")
    print("  • Alto Consumo (10%) • Reportes (10%)")
    print("="*80 + "\n")

    # Ejecutar PSO
    pso = ParticleSwarmOptimizer(
        n_particles=30,
        n_iterations=150,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=None
    )

    pesos_optimos, resultado, historial = pso.optimize(verbose=True)

    # Mostrar resultados
    imprimir_resultados_detallados(pesos_optimos, resultado)

    print("\n" + "="*80)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*80 + "\n")
