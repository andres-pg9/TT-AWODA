"""
Script para inicializar la base de datos MongoDB con datos de ejemplo.
Ejecutar DESPUÉS de tener MongoDB corriendo.

Uso:
    python init_database.py
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from passlib.context import CryptContext
import os

# Configuración
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "awoda_db")

# Para hashear passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def init_database():
    """Inicializa la base de datos con datos de ejemplo"""
    
    print("🔌 Conectando a MongoDB...")
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB_NAME]
    
    try:
        # Verificar conexión
        await client.admin.command('ping')
        print(f"✅ Conectado a: {MONGODB_DB_NAME}\n")
        
        # ============================================================
        # 1. CREAR COLECCIÓN: usuarios
        # ============================================================
        print("👤 Creando usuarios de ejemplo...")
        usuarios_collection = db["usuarios"]
        
        usuarios_ejemplo = [
            {
                "numero_empleado": 215646,
                "password_hash": pwd_context.hash("admin123"),  # Password: admin123
                "nombre_empleado": "Luisa Martínez",
                "rol_usuario": "administrador"
            },
            {
                "numero_empleado": 215647,
                "password_hash": pwd_context.hash("trabajador123"),  # Password: trabajador123
                "nombre_empleado": "Juan Pérez",
                "rol_usuario": "trabajador"
            }
        ]
        
        # Limpiar colección si existe
        await usuarios_collection.delete_many({})
        
        # Insertar usuarios
        result = await usuarios_collection.insert_many(usuarios_ejemplo)
        print(f"   ✔️ {len(result.inserted_ids)} usuarios creados")
        print(f"      • Admin: 215646 / admin123")
        print(f"      • Trabajador: 215647 / trabajador123\n")
        
        # Crear índice único en numero_empleado
        await usuarios_collection.create_index("numero_empleado", unique=True)
        
        # ============================================================
        # 2. CREAR COLECCIÓN: datos_colonias
        # ============================================================
        print("🏘️  Creando datos de colonias de ejemplo...")
        datos_colonias_collection = db["datos_colonias"]
        
        colonias_ejemplo = [
            {
                "colonia": "Lindavista I",
                "fecha_consulta": datetime(2025, 11, 10, 12, 0, 0),
                "numero_reportes": 405,
                "consumo_promedio_agua": 369692.29
            },
            {
                "colonia": "Lindavista II",
                "fecha_consulta": datetime(2025, 11, 10, 12, 0, 0),
                "numero_reportes": 265,
                "consumo_promedio_agua": 523630.27
            },
            {
                "colonia": "Tepeyac Insurgentes",
                "fecha_consulta": datetime(2025, 11, 10, 12, 0, 0),
                "numero_reportes": 319,
                "consumo_promedio_agua": 213349.26
            },
            {
                "colonia": "Magdalena de las Salinas",
                "fecha_consulta": datetime(2025, 11, 10, 12, 0, 0),
                "numero_reportes": 128,
                "consumo_promedio_agua": 440823.13
            },
            {
                "colonia": "Residencial Zacatenco",
                "fecha_consulta": datetime(2025, 11, 10, 12, 0, 0),
                "numero_reportes": 195,
                "consumo_promedio_agua": 126827.67
            },
            {
                "colonia": "Villa GAM",
                "fecha_consulta": datetime(2025, 11, 10, 12, 0, 0),
                "numero_reportes": 79,
                "consumo_promedio_agua": 98406.67
            },
            {
                "colonia": "Capultitlán",
                "fecha_consulta": datetime(2025, 11, 10, 12, 0, 0),
                "numero_reportes": 33,
                "consumo_promedio_agua": 87980.69
            }
        ]
        
        # Limpiar colección si existe
        await datos_colonias_collection.delete_many({})
        
        # Insertar datos
        result = await datos_colonias_collection.insert_many(colonias_ejemplo)
        print(f"   ✔️ {len(result.inserted_ids)} registros de colonias creados\n")
        
        # Crear índices
        await datos_colonias_collection.create_index([("colonia", 1), ("fecha_consulta", -1)])
        
        # ============================================================
        # 3. CREAR COLECCIÓN: resultados_optimizacion (vacía)
        # ============================================================
        print("📊 Creando colección de resultados...")
        resultados_collection = db["resultados_optimizacion"]
        
        # Solo limpiar, se llenará cuando se ejecute optimize
        await resultados_collection.delete_many({})
        print(f"   ✔️ Colección 'resultados_optimizacion' lista (vacía)\n")
        
        # Crear índices
        await resultados_collection.create_index([("fecha_calculo", -1)])
        await resultados_collection.create_index("usuario_id")
        
        # ============================================================
        # RESUMEN
        # ============================================================
        print("="*60)
        print("✅ BASE DE DATOS INICIALIZADA CORRECTAMENTE")
        print("="*60)
        print(f"\n📊 Estadísticas:")
        print(f"   • Usuarios: {await usuarios_collection.count_documents({})}")
        print(f"   • Datos colonias: {await datos_colonias_collection.count_documents({})}")
        print(f"   • Resultados: {await resultados_collection.count_documents({})}")
        print("\n🔐 Credenciales de prueba:")
        print("   • Admin: 215646 / admin123")
        print("   • Trabajador: 215647 / trabajador123")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        client.close()
        print("🔌 Conexión cerrada")


if __name__ == "__main__":
    asyncio.run(init_database())