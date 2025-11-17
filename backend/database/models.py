from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId


# ============================================================================
# HELPER para ObjectId (Compatible con Pydantic v2)
# ============================================================================

class PyObjectId(str):
    """
    Clase helper para trabajar con ObjectId de MongoDB en Pydantic v2.
    """
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ])
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: str(x)
        ))
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("ObjectId inválido")
        return ObjectId(v)


# ============================================================================
# MODELO: Usuario
# ============================================================================
class UsuarioModel(BaseModel):
    """Modelo para la colección 'usuarios'"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    numero_empleado: int
    password_hash: str
    nombre_empleado: str
    rol_usuario: str  # "administrador" o "trabajador"

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )


# ============================================================================
# MODELO: Datos de Colonias
# ============================================================================
class DatosColoniaModel(BaseModel):
    """Modelo para la colección 'datos_colonias'"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    colonia: str
    fecha_consulta: datetime
    numero_reportes: int
    consumo_promedio_agua: float

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
    )


# ============================================================================
# MODELO: Resultados de Optimización
# ============================================================================
class RankingItem(BaseModel):
    """Item individual de ranking (colonia o edificación)"""
    colonia: Optional[str] = None
    tipo: Optional[str] = None  # Para edificaciones
    prioridad: float
    ranking: int


class ResultadoOptimizacionModel(BaseModel):
    """Modelo para la colección 'resultados_optimizacion'"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    fecha_calculo: datetime
    usuario_id: Optional[PyObjectId] = None
    pesos_heuristica: Dict[str, float]  # alfa_legal, beta_social, gamma_consumo, delta_reportes
    utilidad_total: float
    componentes_utilidad: Dict[str, float]  # equidad, social, legal, atencion_consumo, atencion_reportes, coef_gini
    ranking_colonias: List[Dict]
    ranking_edificaciones: List[Dict]
    version_algoritmo: str = "PSO_v1.0"

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
    )