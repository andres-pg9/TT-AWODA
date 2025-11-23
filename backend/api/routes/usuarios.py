"""
Sistema de Gestion de Usuarios para AWODA Backend

Endpoints exclusivos para administradores:
- GET /api/usuarios - Listar todos los usuarios
- POST /api/usuarios - Crear nuevo usuario
- PUT /api/usuarios/{numero_empleado} - Actualizar usuario
- DELETE /api/usuarios/{numero_empleado} - Eliminar usuario

Todos los endpoints requieren autenticacion y rol de administrador.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from passlib.context import CryptContext
from database import UsuarioRepository
from models.schemas import UsuarioCrear, UsuarioActualizar, UsuarioRespuesta
from api.routes.auth import obtener_usuario_admin

router = APIRouter()

# Contexto para hashear passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================================
# ENDPOINTS DE GESTION DE USUARIOS (SOLO ADMINISTRADORES)
# ============================================================================

@router.get("/", response_model=List[UsuarioRespuesta])
async def listar_usuarios(admin_actual: dict = Depends(obtener_usuario_admin)):
    """
    GET /api/usuarios
    
    Lista todos los usuarios del sistema.
    Requiere permisos de administrador.
    
    Headers:
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    
    Response:
    [
        {
            "id": "...",
            "numero_empleado": 215646,
            "nombre_empleado": "Luisa Martinez",
            "rol_usuario": "administrador"
        },
        ...
    ]
    
    Errores:
    - 401: Token invalido o no proporcionado
    - 403: Usuario no es administrador
    """
    try:
        usuarios = await UsuarioRepository.get_all_usuarios()
        
        # Formatear respuesta sin password_hash
        usuarios_respuesta = [
            {
                "id": u["_id"],
                "numero_empleado": u["numero_empleado"],
                "nombre_empleado": u["nombre_empleado"],
                "rol_usuario": u["rol_usuario"]
            }
            for u in usuarios
        ]
        
        print(f"📋 Listando {len(usuarios_respuesta)} usuarios - Admin: {admin_actual['nombre_empleado']}")
        
        return usuarios_respuesta
        
    except Exception as e:
        print(f"❌ Error al listar usuarios: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener usuarios: {str(e)}"
        )


@router.post("/", response_model=UsuarioRespuesta, status_code=status.HTTP_201_CREATED)
async def crear_usuario(
    usuario_datos: UsuarioCrear,
    admin_actual: dict = Depends(obtener_usuario_admin)
):
    """
    POST /api/usuarios
    
    Crea un nuevo usuario en el sistema.
    Requiere permisos de administrador.
    
    Headers:
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    
    Request body:
    {
        "numero_empleado": 215648,
        "password": "password123",
        "nombre_empleado": "Maria Lopez",
        "rol_usuario": "trabajador"
    }
    
    Response:
    {
        "id": "...",
        "numero_empleado": 215648,
        "nombre_empleado": "Maria Lopez",
        "rol_usuario": "trabajador"
    }
    
    Errores:
    - 400: Numero de empleado ya existe
    - 401: Token invalido
    - 403: Usuario no es administrador
    - 422: Datos invalidos
    """
    try:
        # Verificar que el numero de empleado no exista
        usuario_existente = await UsuarioRepository.get_usuario_by_numero_empleado(
            usuario_datos.numero_empleado
        )
        
        if usuario_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe un usuario con el numero de empleado {usuario_datos.numero_empleado}"
            )
        
        # Hashear password
        password_hash = pwd_context.hash(usuario_datos.password)
        
        # Preparar datos para insertar
        nuevo_usuario = {
            "numero_empleado": usuario_datos.numero_empleado,
            "password_hash": password_hash,
            "nombre_empleado": usuario_datos.nombre_empleado,
            "rol_usuario": usuario_datos.rol_usuario
        }
        
        # Crear usuario
        usuario_id = await UsuarioRepository.create_usuario(nuevo_usuario)
        
        print(f"✅ Usuario creado: {usuario_datos.nombre_empleado} ({usuario_datos.numero_empleado}) - Admin: {admin_actual['nombre_empleado']}")
        
        return {
            "id": usuario_id,
            "numero_empleado": usuario_datos.numero_empleado,
            "nombre_empleado": usuario_datos.nombre_empleado,
            "rol_usuario": usuario_datos.rol_usuario
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error al crear usuario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear usuario: {str(e)}"
        )


@router.put("/{numero_empleado}", response_model=UsuarioRespuesta)
async def actualizar_usuario(
    numero_empleado: int,
    usuario_datos: UsuarioActualizar,
    admin_actual: dict = Depends(obtener_usuario_admin)
):
    """
    PUT /api/usuarios/{numero_empleado}
    
    Actualiza un usuario existente.
    Requiere permisos de administrador.
    
    Headers:
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    
    Request body (todos los campos son opcionales):
    {
        "nombre_empleado": "Maria Lopez Garcia",
        "rol_usuario": "administrador",
        "password": "nueva_password123"
    }
    
    Response:
    {
        "id": "...",
        "numero_empleado": 215648,
        "nombre_empleado": "Maria Lopez Garcia",
        "rol_usuario": "administrador"
    }
    
    Errores:
    - 400: No se puede cambiar rol del ultimo administrador
    - 404: Usuario no encontrado
    - 401: Token invalido
    - 403: Usuario no es administrador
    """
    try:
        # Verificar que el usuario exista
        usuario_existente = await UsuarioRepository.get_usuario_by_numero_empleado(numero_empleado)
        
        if not usuario_existente:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontro usuario con numero de empleado {numero_empleado}"
            )
        
        # Si se esta cambiando el rol de administrador a trabajador, validar
        if (usuario_datos.rol_usuario and 
            usuario_datos.rol_usuario != "administrador" and 
            usuario_existente["rol_usuario"] == "administrador"):
            
            # Contar administradores
            count_admins = await UsuarioRepository.contar_administradores()
            
            if count_admins <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No se puede cambiar el rol del ultimo administrador del sistema"
                )
        
        # Preparar datos para actualizar
        datos_actualizacion = {}
        
        if usuario_datos.nombre_empleado:
            datos_actualizacion["nombre_empleado"] = usuario_datos.nombre_empleado
        
        if usuario_datos.rol_usuario:
            datos_actualizacion["rol_usuario"] = usuario_datos.rol_usuario
        
        if usuario_datos.password:
            datos_actualizacion["password_hash"] = pwd_context.hash(usuario_datos.password)
        
        # Actualizar usuario
        if datos_actualizacion:
            await UsuarioRepository.update_usuario(numero_empleado, datos_actualizacion)
        
        # Obtener usuario actualizado
        usuario_actualizado = await UsuarioRepository.get_usuario_by_numero_empleado(numero_empleado)
        
        print(f"✏️ Usuario actualizado: {usuario_actualizado['nombre_empleado']} ({numero_empleado}) - Admin: {admin_actual['nombre_empleado']}")
        
        return {
            "id": usuario_actualizado["_id"],
            "numero_empleado": usuario_actualizado["numero_empleado"],
            "nombre_empleado": usuario_actualizado["nombre_empleado"],
            "rol_usuario": usuario_actualizado["rol_usuario"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error al actualizar usuario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar usuario: {str(e)}"
        )


@router.delete("/{numero_empleado}")
async def eliminar_usuario(
    numero_empleado: int,
    admin_actual: dict = Depends(obtener_usuario_admin)
):
    """
    DELETE /api/usuarios/{numero_empleado}
    
    Elimina un usuario del sistema.
    No se puede eliminar el ultimo administrador.
    Requiere permisos de administrador.
    
    Headers:
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    
    Response:
    {
        "message": "Usuario eliminado exitosamente",
        "numero_empleado": 215648,
        "nombre_empleado": "Maria Lopez"
    }
    
    Errores:
    - 400: No se puede eliminar el ultimo administrador
    - 404: Usuario no encontrado
    - 401: Token invalido
    - 403: Usuario no es administrador
    """
    try:
        # Verificar que el usuario exista
        usuario = await UsuarioRepository.get_usuario_by_numero_empleado(numero_empleado)
        
        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontro usuario con numero de empleado {numero_empleado}"
            )
        
        # Si es administrador, validar que no sea el ultimo
        if usuario["rol_usuario"] == "administrador":
            count_admins = await UsuarioRepository.contar_administradores()
            
            if count_admins <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No se puede eliminar el ultimo administrador del sistema. Debe existir al menos un administrador."
                )
        
        # Eliminar usuario
        eliminado = await UsuarioRepository.delete_usuario(numero_empleado)
        
        if not eliminado:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al eliminar el usuario"
            )
        
        print(f"🗑️ Usuario eliminado: {usuario['nombre_empleado']} ({numero_empleado}) - Admin: {admin_actual['nombre_empleado']}")
        
        return {
            "message": "Usuario eliminado exitosamente",
            "numero_empleado": numero_empleado,
            "nombre_empleado": usuario["nombre_empleado"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error al eliminar usuario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar usuario: {str(e)}"
        )
