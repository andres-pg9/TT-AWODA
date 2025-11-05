from fastapi import APIRouter, HTTPException
from models.schemas import DatosEntrada, ResultadoSalida
from ia.pso import ParticleSwarmOptimizer
from ia.resultados import imprimir_resultados_detallados
from ia.normalizacion import normalizar_valores, CONSUMO, REPORTES
import traceback

router = APIRouter()

@router.get("/")
def obtener_rankings():
    """
    GET /api/optimize
    
    Retorna los rankings de colonias y edificaciones calculados con los datos por defecto.
    - colonias: Lista ordenada por prioridad (mayor a menor)
    - edificaciones: Lista ordenada por prioridad (mayor a menor)
    
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

        # Normalizar los valores por defecto
        consumo_norm = normalizar_valores(CONSUMO, piso=0.3)
        reportes_norm = normalizar_valores(REPORTES, piso=0.3)

        # Procesar resultados completos
        salida_completa = imprimir_resultados_detallados(
            pesos_optimos, 
            resultado, 
            modo_json=True,
            consumo_norm=consumo_norm,
            reportes_norm=reportes_norm
        )
        
        # Retornar solo colonias y edificaciones
        return {
            "colonias": salida_completa.get("colonias", []),
            "edificaciones": salida_completa.get("edificaciones", [])
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al obtener rankings: {str(e)}")


@router.post("/", response_model=ResultadoSalida)
def ejecutar_optimizacion(datos: DatosEntrada):
    """
    POST /api/optimize
    
    Ejecuta el algoritmo PSO con los datos enviados desde el frontend.
    Retorna la respuesta completa: utilidad_total, pesos_optimos, colonias y edificaciones.
    """
    try:
        # 1️⃣ Crear el optimizador PSO con los parámetros base
        pso = ParticleSwarmOptimizer(
            n_particles=30,
            n_iterations=150,
            w=0.7,
            c1=1.5,
            c2=1.5,
            seed=None
        )

        # 2️⃣ Ejecutar optimización pasando consumo y reportes dinámicos
        # El método optimize() internamente normaliza estos valores
        pesos_optimos, resultado, _ = pso.optimize(
            consumo=datos.consumo,
            reportes=datos.reportes,
            verbose=False
        )

        # 3️⃣ Normalizar los mismos valores para pasarlos a resultados
        consumo_norm = normalizar_valores(datos.consumo, piso=0.3)
        reportes_norm = normalizar_valores(datos.reportes, piso=0.3)

        # 4️⃣ Procesar resultados en formato JSON, pasando los valores normalizados
        salida = imprimir_resultados_detallados(
            pesos_optimos, 
            resultado, 
            modo_json=True,
            consumo_norm=consumo_norm,
            reportes_norm=reportes_norm
        )
        
        return salida

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al ejecutar el algoritmo: {str(e)}")
