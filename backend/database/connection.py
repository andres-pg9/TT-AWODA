from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from core.settings import settings
import logging

logger = logging.getLogger(__name__)

class Database:
    """
    Clase para manejar la conexión a MongoDB usando Motor (async driver).
    Sigue el patrón Singleton para mantener una única instancia de conexión.
    """
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect_db(cls):
        """
        Establece la conexión con MongoDB.
        Se llama al iniciar la aplicación (startup event).
        """
        try:
            cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
            cls.db = cls.client[settings.MONGODB_DB_NAME]
            
            # Verificar conexión
            await cls.client.admin.command('ping')
            logger.info(f"✅ Conexión exitosa a MongoDB: {settings.MONGODB_DB_NAME}")
            
        except ConnectionFailure as e:
            logger.error(f"❌ Error al conectar a MongoDB: {e}")
            raise

    @classmethod
    async def close_db(cls):
        """
        Cierra la conexión con MongoDB.
        Se llama al cerrar la aplicación (shutdown event).
        """
        if cls.client:
            cls.client.close()
            logger.info("🔌 Conexión a MongoDB cerrada")

    @classmethod
    def get_db(cls):
        """
        Retorna la instancia de la base de datos.
        
        Returns:
            AsyncIOMotorDatabase: Instancia de la base de datos MongoDB
        """
        if cls.db is None:
            raise Exception("Database no está conectada. Llama a connect_db() primero.")
        return cls.db

    @classmethod
    def get_collection(cls, collection_name: str):
        """
        Retorna una colección específica de la base de datos.
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            AsyncIOMotorCollection: Colección de MongoDB
        """
        db = cls.get_db()
        return db[collection_name]


# Alias para facilitar el acceso
database = Database()