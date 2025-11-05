from pydantic import BaseModel
from typing import Dict, List

class DatosEntrada(BaseModel):
    consumo: Dict[str, float]
    reportes: Dict[str, float]

class ResultadoColonia(BaseModel):
    nombre: str
    prioridad: float
    ranking: int

class ResultadoEdificacion(BaseModel):
    nombre: str
    prioridad: float
    ranking: int

class ResultadoSalida(BaseModel):
    utilidad_total: float
    pesos_optimos: Dict[str, float]
    colonias: List[ResultadoColonia]
    edificaciones: List[ResultadoEdificacion]
