from fastapi import APIRouter, HTTPException
from models.schemas import DatosEntrada, ResultadoSalida
from ia.pso import ParticleSwarmOptimizer
from ia.resultados import imprimir_resultados_detallados
from ia.normalizacion import normalizar_valores, CONSUMO, REPORTES, COLONIAS
from database import DatosColoniaRepository, ResultadoOptimizacionRepository
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends
from api.routes.auth import get_current_user
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
async def obtener_rankings():
    """
    GET /api/optimize
    
    Retorna los rankings de colonias y edificaciones del último resultado guardado en MongoDB.
    Si no hay resultados previos, ejecuta una optimización con datos por defecto.
    
    Returns:
        - colonias: Lista ordenada por prioridad (mayor a menor)
        - edificaciones: Lista ordenada por prioridad (mayor a menor)
    """
    try:
        # Intentar obtener el último resultado de MongoDB
        ultimo_resultado = await ResultadoOptimizacionRepository.get_ultimo_resultado()
        
        if ultimo_resultado:
            # Si existe un resultado previo, usarlo
            print("📊 Usando último resultado de MongoDB")
            return {
                "colonias": ultimo_resultado.get("ranking_colonias", []),
                "edificaciones": ultimo_resultado.get("ranking_edificaciones", [])
            }
        
        # Si no hay resultados previos, calcular con datos por defecto
        print("🔄 No hay resultados previos, calculando con datos por defecto...")
        
        # Crear el optimizador PSO con los parámetros base
        pso = ParticleSwarmOptimizer(
            n_particles=30,
            n_iterations=150,
            w=0.7,
            c1=1.5,
            c2=1.5,
            seed=42  # Semilla fija para resultados reproducibles
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
async def ejecutar_optimizacion(
    datos: DatosEntrada,
    current_user: Optional[dict] = Depends(get_current_user)  # Usuario autenticado (opcional)
):
    """
    POST /api/optimize
    
    Ejecuta el algoritmo PSO con los datos enviados desde el frontend.
    
    Flujo:
    1. Valida y procesa los datos de entrada
    2. Guarda los datos de entrada en MongoDB (datos_colonias)
    3. Ejecuta el algoritmo PSO
    4. Guarda los resultados en MongoDB (resultados_optimizacion)
    5. Retorna la respuesta completa
    
    Si todos los valores de consumo o reportes son 0, usa los valores por defecto.
    Retorna: utilidad_total, pesos_optimos, colonias y edificaciones.
    """
    try:
        usuario_id = current_user["_id"] if current_user else None
        
        if usuario_id:
            print(f"Usuario autenticado: {current_user['nombre_empleado']}")
        else:
            print("Usuario anónimo (sin autenticación)")
        
        # 1️⃣ Validar datos de entrada y decidir si usar defaults
        consumo_a_usar, reportes_a_usar, usando_defaults = validar_y_obtener_datos(
            datos.consumo,
            datos.reportes
        )
        
        # Log para debug
        if usando_defaults:
            print("Usando valores por defecto porque se detectaron datos en 0")
            print(f"   - Consumo: {'DEFAULT' if all(v == 0 for v in datos.consumo.values()) else 'CUSTOM'}")
            print(f"   - Reportes: {'DEFAULT' if all(v == 0 for v in datos.reportes.values()) else 'CUSTOM'}")
        
        # 2️⃣ Guardar datos de entrada en MongoDB (si no son defaults)
        if not usando_defaults:
            print("Guardando datos de entrada en MongoDB...")
            fecha_actual = datetime.utcnow()
            
            # Guardar datos de cada colonia
            for colonia in COLONIAS:
                datos_colonia = {
                    "colonia": colonia,
                    "fecha_consulta": fecha_actual,
                    "numero_reportes": reportes_a_usar.get(colonia, 0),
                    "consumo_promedio_agua": consumo_a_usar.get(colonia, 0.0)
                }
                await DatosColoniaRepository.create_datos_colonia(datos_colonia)
            
            print("   ✅ Datos de entrada guardados")
        
        # 3️⃣ Crear el optimizador PSO con los parámetros base
        pso = ParticleSwarmOptimizer(
            n_particles=30,
            n_iterations=150,
            w=0.7,
            c1=1.5,
            c2=1.5,
            seed=42 if usando_defaults else None  # Semilla fija si usa defaults
        )

        # 4️⃣ Ejecutar optimización con los datos validados
        print("🔄 Ejecutando optimización PSO...")
        pesos_optimos, resultado, _ = pso.optimize(
            consumo=consumo_a_usar,
            reportes=reportes_a_usar,
            verbose=False
        )

        # 5️⃣ Normalizar los valores usados para pasarlos a resultados
        consumo_norm = normalizar_valores(consumo_a_usar, piso=0.3)
        reportes_norm = normalizar_valores(reportes_a_usar, piso=0.3)

        # 6️⃣ Procesar resultados en formato JSON
        salida = imprimir_resultados_detallados(
            pesos_optimos, 
            resultado, 
            modo_json=True,
            consumo_norm=consumo_norm,
            reportes_norm=reportes_norm
        )
        
        # 7️⃣ Guardar resultado en MongoDB
        print("Guardando resultado en MongoDB...")
        resultado_data = {
            "fecha_calculo": datetime.utcnow(),
            "usuario_id": usuario_id,  # Por ahora sin usuario (lo agregaremos en Paso 4)
            "pesos_heuristica": {
                "alfa_legal": float(pesos_optimos[0]),
                "beta_social": float(pesos_optimos[1]),
                "gamma_consumo": float(pesos_optimos[2]),
                "delta_reportes": float(pesos_optimos[3])
            },
            "utilidad_total": float(resultado.get("utilidad_total", 0)),
            "componentes_utilidad": {
                "equidad": float(resultado.get("equidad", 0)),
                "social": float(resultado.get("satisfaccion_social", 0)),
                "legal": float(resultado.get("cumplimiento_legal", 0)),
                "atencion_consumo": float(resultado.get("atencion_consumo", 0)),
                "atencion_reportes": float(resultado.get("atencion_reportes", 0)),
                "coef_gini": float(resultado.get("coeficiente_gini", 0))
            },
            "ranking_colonias": [
                {
                    "colonia": col["nombre"],
                    "prioridad": float(col["prioridad"]),
                    "ranking": int(col["ranking"])
                }
                for col in salida.get("colonias", [])
            ],
            "ranking_edificaciones": [
                {
                    "tipo": edif["nombre"],
                    "prioridad": float(edif["prioridad"]),
                    "ranking": int(edif["ranking"])
                }
                for edif in salida.get("edificaciones", [])
            ],
            "version_algoritmo": "PSO_v1.0"
        }
        
        resultado_id = await ResultadoOptimizacionRepository.create_resultado(resultado_data)
        
        if usuario_id:
            print(f"Resultado guardado con ID: {resultado_id} (asociado a usuario {current_user['nombre_empleado']})")
        else:
            print(f"Resultado guardado con ID: {resultado_id} (anónimo)")
        
        return salida

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al ejecutar el algoritmo: {str(e)}")


@router.get("/historial")
async def obtener_historial(limit: int = 10):
    """
    GET /api/optimize/historial
    
    Obtiene el historial de los últimos N resultados de optimización.
    
    Args:
        limit: Número de resultados a retornar (default: 10, max: 50)
        
    Returns:
        Lista de resultados ordenados por fecha (más reciente primero)
    """
    try:
        # Limitar el máximo a 50 resultados
        limit = min(limit, 50)
        
        # Obtener últimos resultados de MongoDB
        resultados = await ResultadoOptimizacionRepository.get_ultimos_resultados(limit=limit)
        
        # Formatear respuesta
        historial = []
        for resultado in resultados:
            historial.append({
                "id": resultado["_id"],
                "fecha_calculo": resultado["fecha_calculo"],
                "utilidad_total": resultado["utilidad_total"],
                "pesos_heuristica": resultado["pesos_heuristica"],
                "total_colonias": len(resultado.get("ranking_colonias", [])),
                "total_edificaciones": len(resultado.get("ranking_edificaciones", []))
            })
        
        return {
            "total": len(historial),
            "resultados": historial
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")


@router.get("/{resultado_id}")
async def obtener_resultado_especifico(resultado_id: str):
    """
    GET /api/optimize/{resultado_id}
    
    Obtiene un resultado específico de optimización por su ID.
    
    Args:
        resultado_id: ID del resultado en MongoDB
        
    Returns:
        Resultado completo con todos los detalles
    """
    try:
        # Buscar resultado por ID
        resultado = await ResultadoOptimizacionRepository.get_resultado_by_id(resultado_id)
        
        if not resultado:
            raise HTTPException(status_code=404, detail="Resultado no encontrado")
        
        # Retornar resultado completo
        return {
            "id": resultado["_id"],
            "fecha_calculo": resultado["fecha_calculo"],
            "utilidad_total": resultado["utilidad_total"],
            "pesos_heuristica": resultado["pesos_heuristica"],
            "componentes_utilidad": resultado["componentes_utilidad"],
            "ranking_colonias": resultado["ranking_colonias"],
            "ranking_edificaciones": resultado["ranking_edificaciones"],
            "version_algoritmo": resultado.get("version_algoritmo", "PSO_v1.0")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al obtener resultado: {str(e)}")