import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "AWODA Backend"
    VERSION = "1.0"
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
    
    # Configuración de MongoDB
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "awoda_db")
    
    # Colecciones de MongoDB
    COLLECTION_USUARIOS = "usuarios"
    COLLECTION_DATOS_COLONIAS = "datos_colonias"
    COLLECTION_RESULTADOS_OPTIMIZACION = "resultados_optimizacion"

settings = Settings()