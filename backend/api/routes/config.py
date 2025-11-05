from fastapi import APIRouter
from ia.pso import ParticleSwarmOptimizer
from ia.normalizacion import CONSUMO, REPORTES, normalizar_valores

router = APIRouter()

@router.get("/")
def obtener_configuracion():
    """
    GET /api/config
    
    Retorna la configuración óptima del sistema:
    - utilidad_total: Valor de la función objetivo optimizada
    - pesos_optimos: Pesos α, β, γ, δ encontrados por el algoritmo PSO
    
    Usa los datos por defecto definidos en normalizacion.py
    """
    try:
        # Crear el optimizador PSO con los parámetros base
        pso = ParticleSwarmOptimizer(
            n_particles=30,
            n_iterations=150,
            w=0.7,
            c1=1.5,
            c2=1.5,
            seed=42  # Semilla fija para resultados reproducibles en GET
        )

        # Ejecutar optimización con datos por defecto
        pesos_optimos, resultado, _ = pso.optimize(
            consumo=CONSUMO,
            reportes=REPORTES,
            verbose=False
        )

        # Retornar solo utilidad_total y pesos_optimos
        return {
            "utilidad_total": resultado.get("utilidad_total", 0),
            "pesos_optimos": {
                "α": round(pesos_optimos[0], 4),
                "β": round(pesos_optimos[1], 4),
                "γ": round(pesos_optimos[2], 4),
                "δ": round(pesos_optimos[3], 4)
            }
        }

    except Exception as e:
        return {
            "error": f"Error al calcular configuración: {str(e)}",
            "utilidad_total": 0,
            "pesos_optimos": {"α": 0, "β": 0, "γ": 0, "δ": 0}
        }
