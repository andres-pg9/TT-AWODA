"""
Repositories (Repositorios) - Capa de acceso a datos

Este módulo contiene las funciones CRUD (Create, Read, Update, Delete)
para interactuar con las colecciones de MongoDB.

Arquitectura:
    - Cada colección tiene su propia clase Repository
    - Métodos async para compatibilidad con FastAPI
    - Validación con modelos Pydantic
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from database.connection import Database
from database.models import UsuarioModel, DatosColoniaModel, ResultadoOptimizacionModel
from core.settings import settings


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_objectid_to_str(document: Dict) -> Dict:
    """
    Convierte ObjectId a string en un documento de MongoDB.
    Útil para serializar documentos en JSON.
    
    Args:
        document: Documento de MongoDB
        
    Returns:
        Documento con _id como string
    """
    if document and "_id" in document:
        document["_id"] = str(document["_id"])
    return document


# ============================================================================
# REPOSITORY: Usuarios
# ============================================================================

class UsuarioRepository:
    """
    Repository para la colección 'usuarios'.
    Maneja operaciones CRUD para usuarios del sistema.
    """
    
    @staticmethod
    def get_collection() -> AsyncIOMotorCollection:
        """Obtiene la colección de usuarios"""
        return Database.get_collection(settings.COLLECTION_USUARIOS)
    
    @staticmethod
    async def create_usuario(usuario_data: Dict[str, Any]) -> str:
        """
        Crea un nuevo usuario.
        
        Args:
            usuario_data: Diccionario con datos del usuario
            
        Returns:
            ID del usuario creado (como string)
        """
        collection = UsuarioRepository.get_collection()
        result = await collection.insert_one(usuario_data)
        return str(result.inserted_id)
    
    @staticmethod
    async def get_usuario_by_numero_empleado(numero_empleado: int) -> Optional[Dict]:
        """
        Busca un usuario por su número de empleado.
        
        Args:
            numero_empleado: Número de empleado único
            
        Returns:
            Documento del usuario o None si no existe
        """
        collection = UsuarioRepository.get_collection()
        usuario = await collection.find_one({"numero_empleado": numero_empleado})
        return convert_objectid_to_str(usuario) if usuario else None
    
    @staticmethod
    async def get_usuario_by_id(usuario_id: str) -> Optional[Dict]:
        """
        Busca un usuario por su ID de MongoDB.
        
        Args:
            usuario_id: ID del usuario (string)
            
        Returns:
            Documento del usuario o None si no existe
        """
        collection = UsuarioRepository.get_collection()
        usuario = await collection.find_one({"_id": ObjectId(usuario_id)})
        return convert_objectid_to_str(usuario) if usuario else None
    
    @staticmethod
    async def get_all_usuarios() -> List[Dict]:
        """
        Obtiene todos los usuarios.
        
        Returns:
            Lista de documentos de usuarios
        """
        collection = UsuarioRepository.get_collection()
        usuarios = await collection.find().to_list(length=None)
        return [convert_objectid_to_str(u) for u in usuarios]
    
    @staticmethod
    async def update_usuario(numero_empleado: int, update_data: Dict[str, Any]) -> bool:
        """
        Actualiza un usuario existente.
        
        Args:
            numero_empleado: Número de empleado del usuario
            update_data: Diccionario con campos a actualizar
            
        Returns:
            True si se actualizó, False si no se encontró
        """
        collection = UsuarioRepository.get_collection()
        result = await collection.update_one(
            {"numero_empleado": numero_empleado},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    @staticmethod
    async def delete_usuario(numero_empleado: int) -> bool:
        """
        Elimina un usuario.
        
        Args:
            numero_empleado: Número de empleado del usuario
            
        Returns:
            True si se eliminó, False si no se encontró
        """
        collection = UsuarioRepository.get_collection()
        result = await collection.delete_one({"numero_empleado": numero_empleado})
        return result.deleted_count > 0
    
    @staticmethod
    async def contar_administradores() -> int:
        """
        Cuenta cuantos usuarios administradores existen.
        Esto para validar que siempre exista al menos uno.
        
        Returns:
            Numero de administradores en el sistema
        """
        collection = UsuarioRepository.get_collection()
        count = await collection.count_documents({"rol_usuario": "administrador"})
        return count


# ============================================================================
# REPOSITORY: Datos de Colonias
# ============================================================================

class DatosColoniaRepository:
    """
    Repository para la colección 'datos_colonias'.
    Maneja datos históricos de consumo y reportes por colonia.
    """
    
    @staticmethod
    def get_collection() -> AsyncIOMotorCollection:
        """Obtiene la colección de datos de colonias"""
        return Database.get_collection(settings.COLLECTION_DATOS_COLONIAS)
    
    @staticmethod
    async def create_datos_colonia(datos: Dict[str, Any]) -> str:
        """
        Crea un nuevo registro de datos de colonia.
        
        Args:
            datos: Diccionario con datos de la colonia
            
        Returns:
            ID del registro creado (como string)
        """
        collection = DatosColoniaRepository.get_collection()
        
        # Asegurar que fecha_consulta sea datetime
        if "fecha_consulta" in datos and not isinstance(datos["fecha_consulta"], datetime):
            datos["fecha_consulta"] = datetime.fromisoformat(datos["fecha_consulta"])
        
        result = await collection.insert_one(datos)
        return str(result.inserted_id)
    
    @staticmethod
    async def get_datos_by_colonia(colonia: str, limit: int = 10) -> List[Dict]:
        """
        Obtiene los datos más recientes de una colonia específica.
        
        Args:
            colonia: Nombre de la colonia
            limit: Número máximo de registros a retornar
            
        Returns:
            Lista de documentos ordenados por fecha (más reciente primero)
        """
        collection = DatosColoniaRepository.get_collection()
        datos = await collection.find(
            {"colonia": colonia}
        ).sort("fecha_consulta", -1).limit(limit).to_list(length=limit)
        
        return [convert_objectid_to_str(d) for d in datos]
    
    @staticmethod
    async def get_ultimos_datos_todas_colonias() -> Dict[str, Dict]:
        """
        Obtiene el registro más reciente de cada colonia.
        Útil para obtener el estado actual de todas las colonias.
        
        Returns:
            Diccionario con nombre_colonia: datos_más_recientes
        """
        collection = DatosColoniaRepository.get_collection()
        
        # Pipeline de agregación para obtener el último registro por colonia
        pipeline = [
            {"$sort": {"fecha_consulta": -1}},
            {"$group": {
                "_id": "$colonia",
                "ultimo_dato": {"$first": "$$ROOT"}
            }}
        ]
        
        resultados = await collection.aggregate(pipeline).to_list(length=None)
        
        # Convertir a diccionario colonia -> datos
        datos_por_colonia = {}
        for resultado in resultados:
            colonia = resultado["_id"]
            datos = convert_objectid_to_str(resultado["ultimo_dato"])
            datos_por_colonia[colonia] = datos
        
        return datos_por_colonia
    
    @staticmethod
    async def get_datos_by_fecha_range(
        fecha_inicio: datetime,
        fecha_fin: datetime
    ) -> List[Dict]:
        """
        Obtiene datos dentro de un rango de fechas.
        
        Args:
            fecha_inicio: Fecha inicial del rango
            fecha_fin: Fecha final del rango
            
        Returns:
            Lista de documentos en el rango de fechas
        """
        collection = DatosColoniaRepository.get_collection()
        datos = await collection.find({
            "fecha_consulta": {
                "$gte": fecha_inicio,
                "$lte": fecha_fin
            }
        }).sort("fecha_consulta", -1).to_list(length=None)
        
        return [convert_objectid_to_str(d) for d in datos]
    
    @staticmethod
    async def delete_datos_colonia(dato_id: str) -> bool:
        """
        Elimina un registro de datos de colonia.
        
        Args:
            dato_id: ID del registro (string)
            
        Returns:
            True si se eliminó, False si no se encontró
        """
        collection = DatosColoniaRepository.get_collection()
        result = await collection.delete_one({"_id": ObjectId(dato_id)})
        return result.deleted_count > 0

    @staticmethod
    async def get_historial_by_colonia(nombre_colonia: str, limit: int = 10) -> List[Dict]:
        """
        Obtiene el historial de datos de una colonia específica.
        
        Args:
            nombre_colonia: Nombre de la colonia a consultar
            limit: Número máximo de registros a retornar
            
        Returns:
            Lista de documentos ordenados por fecha_consulta descendente (más reciente primero)
        """
        collection = DatosColoniaRepository.get_collection()
        
        # Buscar todos los registros de la colonia, ordenados por fecha descendente
        datos = await collection.find(
            {"colonia": nombre_colonia}
        ).sort(
            "fecha_consulta", -1  # -1 = descendente (más reciente primero)
        ).limit(limit).to_list(length=limit)
        
        return [convert_objectid_to_str(d) for d in datos]
    
    @staticmethod
    async def get_lista_colonias() -> List[str]:
        """
        Obtiene la lista única de todas las colonias en la base de datos.
        
        Returns:
            Lista de nombres de colonias (strings únicos)
        """
        collection = DatosColoniaRepository.get_collection()
        
        # Usar distinct para obtener valores únicos del campo "colonia"
        colonias = await collection.distinct("colonia")
        
        return colonias

# ============================================================================
# REPOSITORY: Resultados de Optimización
# ============================================================================

class ResultadoOptimizacionRepository:
    """
    Repository para la colección 'resultados_optimizacion'.
    Maneja los resultados del algoritmo PSO.
    """
    
    @staticmethod
    def get_collection() -> AsyncIOMotorCollection:
        """Obtiene la colección de resultados de optimización"""
        return Database.get_collection(settings.COLLECTION_RESULTADOS_OPTIMIZACION)
    
    @staticmethod
    async def create_resultado(resultado_data: Dict[str, Any]) -> str:
        """
        Guarda un nuevo resultado de optimización.
        
        Args:
            resultado_data: Diccionario con el resultado completo del PSO
            
        Returns:
            ID del resultado creado (como string)
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        
        # Asegurar que fecha_calculo sea datetime
        if "fecha_calculo" not in resultado_data:
            resultado_data["fecha_calculo"] = datetime.utcnow()
        elif not isinstance(resultado_data["fecha_calculo"], datetime):
            resultado_data["fecha_calculo"] = datetime.fromisoformat(resultado_data["fecha_calculo"])
        
        result = await collection.insert_one(resultado_data)
        return str(result.inserted_id)
    
    @staticmethod
    async def get_ultimo_resultado() -> Optional[Dict]:
        """
        Obtiene el resultado más reciente.
        
        Returns:
            Documento del último resultado o None si no hay resultados
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        resultado = await collection.find_one(
            sort=[("fecha_calculo", -1)]
        )
        return convert_objectid_to_str(resultado) if resultado else None
    
    @staticmethod
    async def get_resultado_by_id(resultado_id: str) -> Optional[Dict]:
        """
        Busca un resultado específico por ID.
        
        Args:
            resultado_id: ID del resultado (string)
            
        Returns:
            Documento del resultado o None si no existe
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        resultado = await collection.find_one({"_id": ObjectId(resultado_id)})
        return convert_objectid_to_str(resultado) if resultado else None
    
    @staticmethod
    async def get_resultados_by_usuario(usuario_id: str, limit: int = 10) -> List[Dict]:
        """
        Obtiene los resultados generados por un usuario específico.
        
        Args:
            usuario_id: ID del usuario (string)
            limit: Número máximo de resultados a retornar
            
        Returns:
            Lista de documentos de resultados
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        resultados = await collection.find(
            {"usuario_id": ObjectId(usuario_id)}
        ).sort("fecha_calculo", -1).limit(limit).to_list(length=limit)
        
        return [convert_objectid_to_str(r) for r in resultados]
    
    @staticmethod
    async def get_ultimos_resultados(limit: int = 10) -> List[Dict]:
        """
        Obtiene los últimos N resultados.
        
        Args:
            limit: Número de resultados a retornar
            
        Returns:
            Lista de documentos ordenados por fecha (más reciente primero)
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        resultados = await collection.find().sort(
            "fecha_calculo", -1
        ).limit(limit).to_list(length=limit)
        
        return [convert_objectid_to_str(r) for r in resultados]
    
    @staticmethod
    async def get_resultados_by_fecha_range(
        fecha_inicio: datetime,
        fecha_fin: datetime
    ) -> List[Dict]:
        """
        Obtiene resultados dentro de un rango de fechas.
        
        Args:
            fecha_inicio: Fecha inicial del rango
            fecha_fin: Fecha final del rango
            
        Returns:
            Lista de documentos en el rango de fechas
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        resultados = await collection.find({
            "fecha_calculo": {
                "$gte": fecha_inicio,
                "$lte": fecha_fin
            }
        }).sort("fecha_calculo", -1).to_list(length=None)
        
        return [convert_objectid_to_str(r) for r in resultados]
    
    @staticmethod
    async def delete_resultado(resultado_id: str) -> bool:
        """
        Elimina un resultado de optimización.
        
        Args:
            resultado_id: ID del resultado (string)
            
        Returns:
            True si se eliminó, False si no se encontró
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        result = await collection.delete_one({"_id": ObjectId(resultado_id)})
        return result.deleted_count > 0
    
    @staticmethod
    async def count_resultados() -> int:
        """
        Cuenta el total de resultados almacenados.
        
        Returns:
            Número total de resultados
        """
        collection = ResultadoOptimizacionRepository.get_collection()
        return await collection.count_documents({})