from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional

# ============================================================================
# SCHEMAS DE OPTIMIZACION
# ============================================================================

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


# ============================================================================
# SCHEMAS DE USUARIOS
# ============================================================================

class UsuarioCrear(BaseModel):
    """Schema para crear un nuevo usuario"""
    numero_empleado: int = Field(..., description="Numero de empleado unico", gt=0)
    password: str = Field(..., min_length=6, description="Contraseña minimo 6 caracteres")
    nombre_empleado: str = Field(..., min_length=3, description="Nombre completo del empleado")
    rol_usuario: str = Field(..., description="Rol: administrador o trabajador")
    
    @validator('rol_usuario')
    def validar_rol(cls, valor):
        """Valida que el rol sea valido"""
        roles_validos = ['administrador', 'trabajador']
        if valor not in roles_validos:
            raise ValueError(f'El rol debe ser uno de: {", ".join(roles_validos)}')
        return valor
    
    class Config:
        schema_extra = {
            "example": {
                "numero_empleado": 215648,
                "password": "password123",
                "nombre_empleado": "Maria Lopez",
                "rol_usuario": "trabajador"
            }
        }


class UsuarioActualizar(BaseModel):
    """Schema para actualizar un usuario existente"""
    nombre_empleado: Optional[str] = Field(None, min_length=3, description="Nombre completo del empleado")
    rol_usuario: Optional[str] = Field(None, description="Rol: administrador o trabajador")
    password: Optional[str] = Field(None, min_length=6, description="Nueva contraseña")
    
    @validator('rol_usuario')
    def validar_rol(cls, valor):
        """Valida que el rol sea valido"""
        if valor is not None:
            roles_validos = ['administrador', 'trabajador']
            if valor not in roles_validos:
                raise ValueError(f'El rol debe ser uno de: {", ".join(roles_validos)}')
        return valor
    
    class Config:
        schema_extra = {
            "example": {
                "nombre_empleado": "Maria Lopez Garcia",
                "rol_usuario": "administrador"
            }
        }


class UsuarioRespuesta(BaseModel):
    """Schema para respuesta de informacion de usuario"""
    id: str = Field(..., description="ID del usuario en MongoDB")
    numero_empleado: int = Field(..., description="Numero de empleado unico")
    nombre_empleado: str = Field(..., description="Nombre completo del empleado")
    rol_usuario: str = Field(..., description="Rol del usuario")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "numero_empleado": 215646,
                "nombre_empleado": "Luisa Martinez",
                "rol_usuario": "administrador"
            }
        }
