from fastapi import APIRouter, HTTPException
from models.schemas import DatosEntrada, ResultadoSalida
from ia.pso import ParticleSwarmOptimizer
from ia.resultados import imprimir_resultados_detallados
from ia.normalizacion import normalizar_valores, CONSUMO, REPORTES
import traceback

router = APIRouter()

def validar_y_obtener_datos(consumo_input: dict, reportes_input: dict) -> tuple:
    """
    Valida los datos de entrada y retorna los datos a usar (input o default).
    
    Si todos los valores de consumo o reportes son 0, usa los valores por defecto.
    
    Args:
        consumo_input: Diccionario de consumo desde el frontend
        reportes_input: Diccionario de reportes desde el frontend
        
    Returns:
        tuple: (consumo_a_usar, reportes_a_usar, usando_defaults)
    """
    # Verificar si todos los valores de consumo son 0
    consumo_todos_cero = all(v == 0 for v in consumo_input.values())
    
    # Verificar si todos los valores de reportes son 0
    reportes_todos_cero = all(v == 0 for v in reportes_input.values())
    
    # Decidir qué datos usar
    consumo_a_usar = CONSUMO if consumo_todos_cero else consumo_input
    reportes_a_usar = REPORTES if reportes_todos_cero else reportes_input
    
    # Determinar si se están usando defaults
    usando_defaults = consumo_todos_cero or reportes_todos_cero
    
    return consumo_a_usar, reportes_a_usar, usando_defaults


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
    Si todos los valores de consumo o reportes son 0, usa los valores por defecto.
    Retorna la respuesta completa: utilidad_total, pesos_optimos, colonias y edificaciones.
    """
    try:
        # 1️⃣ Validar datos de entrada y decidir si usar defaults
        consumo_a_usar, reportes_a_usar, usando_defaults = validar_y_obtener_datos(
            datos.consumo,
            datos.reportes
        )
        
        # Log para debug (opcional)
        if usando_defaults:
            print("⚠️ Usando valores por defecto porque se detectaron datos en 0")
            print(f"   - Consumo: {'DEFAULT' if all(v == 0 for v in datos.consumo.values()) else 'CUSTOM'}")
            print(f"   - Reportes: {'DEFAULT' if all(v == 0 for v in datos.reportes.values()) else 'CUSTOM'}")
        
        # 2️⃣ Crear el optimizador PSO con los parámetros base
        pso = ParticleSwarmOptimizer(
            n_particles=30,
            n_iterations=150,
            w=0.7,
            c1=1.5,
            c2=1.5,
            seed=42 if usando_defaults else None  # Semilla fija si usa defaults
        )

        # 3️⃣ Ejecutar optimización con los datos validados
        pesos_optimos, resultado, _ = pso.optimize(
            consumo=consumo_a_usar,
            reportes=reportes_a_usar,
            verbose=False
        )

        # 4️⃣ Normalizar los valores usados para pasarlos a resultados
        consumo_norm = normalizar_valores(consumo_a_usar, piso=0.3)
        reportes_norm = normalizar_valores(reportes_a_usar, piso=0.3)

        # 5️⃣ Procesar resultados en formato JSON
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