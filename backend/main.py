from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database.connection import Database
from api.routes import optimize, config, auth, colonias
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EVENTOS DE CICLO DE VIDA (Startup/Shutdown)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja eventos de inicio y cierre de la aplicación.
    - Startup: Conecta a MongoDB
    - Shutdown: Cierra la conexión a MongoDB
    """
    # Startup
    logger.info("🚀 Iniciando aplicación AWODA Backend...")
    try:
        await Database.connect_db()
        logger.info("✅ Aplicación lista")
    except Exception as e:
        logger.error(f"❌ Error al iniciar: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Cerrando aplicación...")
    await Database.close_db()
    logger.info("👋 Aplicación cerrada correctamente")


# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================
app = FastAPI(
    title="AWODA Backend",
    description="Sistema de Optimización de Distribución de Agua - CDMX",
    version="1.0",
    lifespan=lifespan
)

# ============================================================================
# MIDDLEWARE - CORS
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ROUTERS
# ============================================================================
app.include_router(optimize.router, prefix="/api/optimize", tags=["Optimización"])
app.include_router(config.router, prefix="/api/config", tags=["Configuración"])
app.include_router(auth.router, prefix="/api/auth", tags=["Autenticación"])
app.include_router(colonias.router, prefix="/api/colonias", tags=["Colonias"])

# ============================================================================
# ENDPOINT RAÍZ
# ============================================================================
@app.get("/")
def home():
    """Endpoint de prueba para verificar que el backend está activo"""
    return {
        "status": "Backend de AWODA activo",
        "version": "1.0",
        "database": "MongoDB conectado" if Database.db is not None else "MongoDB no conectado"
    }


@app.get("/health")
async def health_check():
    """
    Endpoint de health check para verificar el estado de la aplicación y MongoDB
    """
    try:
        # Verificar conexión a MongoDB
        db = Database.get_db()
        await db.command('ping')
        
        return {
            "status": "healthy",
            "database": "connected",
            "database_name": db.name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }