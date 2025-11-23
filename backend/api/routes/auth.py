"""
Sistema de Autenticación para AWODA Backend

Endpoints:
- POST /api/auth/login - Login de usuarios
- GET /api/auth/me - Información del usuario actual
- POST /api/auth/logout - Cerrar sesión

Usa JWT (JSON Web Tokens) para manejar sesiones.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from database import UsuarioRepository
import os

router = APIRouter()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Clave secreta para JWT (en producción debe estar en .env)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "tu_clave_secreta_super_segura_cambiar_en_produccion")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 horas

# Contexto para hashear/verificar passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme para JWT
security = HTTPBearer()


# ============================================================================
# SCHEMAS DE PYDANTIC
# ============================================================================

class LoginRequest(BaseModel):
    """Schema para request de login"""
    numero_empleado: int
    password: str


class LoginResponse(BaseModel):
    """Schema para response de login exitoso"""
    access_token: str
    token_type: str
    usuario: dict


class UsuarioActual(BaseModel):
    """Schema para información del usuario autenticado"""
    id: str
    numero_empleado: int
    nombre_empleado: str
    rol_usuario: str


# ============================================================================
# UTILIDADES
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifica si un password coincide con su hash.
    
    Args:
        plain_password: Password en texto plano
        hashed_password: Password hasheado
        
    Returns:
        True si coincide, False si no
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Crea un token JWT.
    
    Args:
        data: Datos a incluir en el token (usuario_id, numero_empleado, etc.)
        expires_delta: Tiempo de expiración opcional
        
    Returns:
        Token JWT como string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Obtiene el usuario actual desde el token JWT.
    Se usa como dependencia en endpoints protegidos.
    
    Args:
        credentials: Credenciales HTTP Bearer con el token
        
    Returns:
        Diccionario con información del usuario
        
    Raises:
        HTTPException: Si el token es inválido o el usuario no existe
    """
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decodificar token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        usuario_id: str = payload.get("sub")
        
        if usuario_id is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Buscar usuario en la base de datos
    usuario = await UsuarioRepository.get_usuario_by_id(usuario_id)
    
    if usuario is None:
        raise credentials_exception
    
    return usuario


async def obtener_usuario_admin(usuario_actual: dict = Depends(get_current_user)) -> dict:
    """
    Verifica que el usuario actual sea administrador.
    Se usa como dependencia en endpoints que requieren privilegios de admin.
    
    Args:
        usuario_actual: Usuario autenticado obtenido por get_current_user
        
    Returns:
        Diccionario con informacion del usuario administrador
        
    Raises:
        HTTPException 403: Si el usuario no es administrador
    """
    if usuario_actual.get("rol_usuario") != "administrador":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permisos de administrador para realizar esta accion"
        )
    
    return usuario_actual


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """
    POST /api/auth/login
    
    Autentica a un usuario con número de empleado y password.
    
    Request body:
    {
        "numero_empleado": 215646,
        "password": "admin123"
    }
    
    Response:
    {
        "access_token": "eyJhbGciOiJIUzI1NiIs...",
        "token_type": "bearer",
        "usuario": {
            "id": "...",
            "numero_empleado": 215646,
            "nombre_empleado": "Luisa Martínez",
            "rol_usuario": "administrador"
        }
    }
    
    Errores:
    - 401: Credenciales inválidas
    - 500: Error del servidor
    """
    try:
        # 1. Buscar usuario por número de empleado
        usuario = await UsuarioRepository.get_usuario_by_numero_empleado(
            login_data.numero_empleado
        )
        
        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Número de empleado o contraseña incorrectos"
            )
        
        # 2. Verificar password
        if not verify_password(login_data.password, usuario["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Número de empleado o contraseña incorrectos"
            )
        
        # 3. Crear token JWT
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": usuario["_id"],  # Subject del token (ID del usuario)
                "numero_empleado": usuario["numero_empleado"],
                "rol": usuario["rol_usuario"]
            },
            expires_delta=access_token_expires
        )
        
        # 4. Preparar respuesta (sin el password_hash)
        usuario_response = {
            "id": usuario["_id"],
            "numero_empleado": usuario["numero_empleado"],
            "nombre_empleado": usuario["nombre_empleado"],
            "rol_usuario": usuario["rol_usuario"]
        }
        
        print(f"✅ Login exitoso: {usuario['nombre_empleado']} ({usuario['numero_empleado']})")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "usuario": usuario_response
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar login: {str(e)}"
        )


@router.get("/me", response_model=UsuarioActual)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    GET /api/auth/me
    
    Obtiene información del usuario autenticado actual.
    Requiere token JWT en el header Authorization.
    
    Headers:
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    
    Response:
    {
        "id": "...",
        "numero_empleado": 215646,
        "nombre_empleado": "Luisa Martínez",
        "rol_usuario": "administrador"
    }
    
    Errores:
    - 401: Token inválido o expirado
    - 500: Error del servidor
    """
    return {
        "id": current_user["_id"],
        "numero_empleado": current_user["numero_empleado"],
        "nombre_empleado": current_user["nombre_empleado"],
        "rol_usuario": current_user["rol_usuario"]
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    POST /api/auth/logout
    
    Cierra la sesión del usuario.
    Nota: Con JWT, el logout es principalmente del lado del cliente
    (eliminar el token del localStorage/sessionStorage).
    
    Este endpoint es más que nada para logging y futuras extensiones.
    
    Headers:
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    
    Response:
    {
        "message": "Sesión cerrada exitosamente"
    }
    """
    print(f"👋 Logout: {current_user['nombre_empleado']} ({current_user['numero_empleado']})")
    
    return {
        "message": "Sesión cerrada exitosamente",
        "usuario": current_user["nombre_empleado"]
    }


@router.get("/validate-token")
async def validate_token(current_user: dict = Depends(get_current_user)):
    """
    GET /api/auth/validate-token
    
    Valida si un token JWT es válido.
    Útil para el frontend para verificar si la sesión sigue activa.
    
    Headers:
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    
    Response:
    {
        "valid": true,
        "usuario": {
            "id": "...",
            "numero_empleado": 215646,
            "nombre_empleado": "Luisa Martínez",
            "rol_usuario": "administrador"
        }
    }
    
    Errores:
    - 401: Token inválido o expirado
    """
    return {
        "valid": True,
        "usuario": {
            "id": current_user["_id"],
            "numero_empleado": current_user["numero_empleado"],
            "nombre_empleado": current_user["nombre_empleado"],
            "rol_usuario": current_user["rol_usuario"]
        }
    }