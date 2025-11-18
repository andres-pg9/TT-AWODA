from fastapi import APIRouter, HTTPException, Query
from database.repositories import DatosColoniaRepository
from typing import Optional
import traceback

router = APIRouter()


@router.get("/{nombre_colonia}/historial")
async def obtener_historial_colonia(
    nombre_colonia: str,
    limit: int = Query(default=10, ge=1, le=50, description="Número de registros a retornar (entre 1 y 50)")
):
    """
    GET /api/colonias/{nombre_colonia}/historial?limit=10
    
    Obtiene el historial de consumo y reportes de una colonia específica.
    """
    try:
        if not nombre_colonia or nombre_colonia.strip() == "":
            raise HTTPException(status_code=400, detail="El nombre de la colonia no puede estar vacío")
        
        historial = await DatosColoniaRepository.get_historial_by_colonia(
            nombre_colonia=nombre_colonia,
            limit=limit
        )
        
        if not historial:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron datos históricos para la colonia '{nombre_colonia}'"
            )
        
        # Formatear datos para la respuesta estándar
        datos_formateados = []
        for registro in historial:
            datos_formateados.append({
                "id": registro["_id"],
                "fecha": registro["fecha_consulta"],
                "consumo": float(registro["consumo_promedio_agua"]),
                "reportes": int(registro["numero_reportes"])
            })
        
        # Formatear datos específicamente para Nivo Line Chart
        datos_nivo_consumo = []
        datos_nivo_reportes = []
        
        # Invertir el orden para que las gráficas muestren los datos de más antiguo a más reciente
        for registro in reversed(historial):
            # 🔧 FIX: Formatear fecha con hora y minuto para que cada punto sea único
            fecha_str = registro["fecha_consulta"].strftime("%Y-%m-%d %H:%M")
            
            datos_nivo_consumo.append({
                "x": fecha_str,
                "y": float(registro["consumo_promedio_agua"])
            })
            
            datos_nivo_reportes.append({
                "x": fecha_str,
                "y": int(registro["numero_reportes"])
            })
        
        # Construir respuesta completa
        return {
            "colonia": nombre_colonia,
            "total_registros": len(historial),
            "limite_aplicado": limit,
            "datos": datos_formateados,
            "formato_nivo": {
                "consumo": datos_nivo_consumo,
                "reportes": datos_nivo_reportes
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error al obtener historial de la colonia: {str(e)}"
        )


@router.get("/")
async def listar_colonias():
    """
    GET /api/colonias
    
    Obtiene la lista de todas las colonias disponibles en el sistema.
    """
    try:
        colonias = await DatosColoniaRepository.get_lista_colonias()
        
        return {
            "total": len(colonias),
            "colonias": sorted(colonias)
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener lista de colonias: {str(e)}"
        )