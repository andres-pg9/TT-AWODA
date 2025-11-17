"""
Script de prueba para verificar que los repositories funcionan correctamente.

Ejecutar DESPUÉS de tener MongoDB corriendo y haber ejecutado init_database.py

Uso:
    python test_repositories.py
"""

import asyncio
from datetime import datetime
from database import Database, UsuarioRepository, DatosColoniaRepository, ResultadoOptimizacionRepository


async def test_repositories():
    """Prueba las operaciones básicas de cada repository"""
    
    print("\n" + "="*70)
    print("🧪 PROBANDO REPOSITORIES")
    print("="*70 + "\n")
    
    # Conectar a MongoDB
    await Database.connect_db()
    
    try:
        # ============================================================
        # TEST 1: UsuarioRepository
        # ============================================================
        print("1️⃣  Probando UsuarioRepository...")
        
        # Obtener usuario por número de empleado
        usuario = await UsuarioRepository.get_usuario_by_numero_empleado(215646)
        if usuario:
            print(f"   ✅ Usuario encontrado: {usuario['nombre_empleado']}")
        else:
            print("   ❌ Usuario no encontrado")
        
        # Listar todos los usuarios
        todos_usuarios = await UsuarioRepository.get_all_usuarios()
        print(f"   ✅ Total de usuarios en DB: {len(todos_usuarios)}\n")
        
        # ============================================================
        # TEST 2: DatosColoniaRepository
        # ============================================================
        print("2️⃣  Probando DatosColoniaRepository...")
        
        # Obtener últimos datos de todas las colonias
        datos_colonias = await DatosColoniaRepository.get_ultimos_datos_todas_colonias()
        print(f"   ✅ Colonias con datos: {len(datos_colonias)}")
        
        # Obtener datos de una colonia específica
        datos_lindavista = await DatosColoniaRepository.get_datos_by_colonia("Lindavista I", limit=5)
        if datos_lindavista:
            print(f"   ✅ Datos de Lindavista I: {len(datos_lindavista)} registros")
            ultimo = datos_lindavista[0]
            print(f"      - Último reporte: {ultimo['numero_reportes']} reportes")
            print(f"      - Consumo promedio: {ultimo['consumo_promedio_agua']:.2f} litros\n")
        
        # ============================================================
        # TEST 3: ResultadoOptimizacionRepository
        # ============================================================
        print("3️⃣  Probando ResultadoOptimizacionRepository...")
        
        # Contar resultados
        total_resultados = await ResultadoOptimizacionRepository.count_resultados()
        print(f"   ✅ Total de resultados en DB: {total_resultados}")
        
        # Obtener último resultado
        ultimo_resultado = await ResultadoOptimizacionRepository.get_ultimo_resultado()
        if ultimo_resultado:
            print(f"   ✅ Último resultado encontrado")
            print(f"      - Utilidad total: {ultimo_resultado['utilidad_total']:.2f}")
            print(f"      - Fecha: {ultimo_resultado['fecha_calculo']}")
        else:
            print(f"   ℹ️  No hay resultados aún (normal si no has ejecutado /optimize)\n")
        
        # ============================================================
        # TEST 4: Crear un resultado de prueba
        # ============================================================
        print("\n4️⃣  Creando resultado de prueba...")
        
        resultado_prueba = {
            "fecha_calculo": datetime.utcnow(),
            "usuario_id": None,  # Sin usuario específico
            "pesos_heuristica": {
                "alfa_legal": 0.25,
                "beta_social": 0.25,
                "gamma_consumo": 0.25,
                "delta_reportes": 0.25
            },
            "utilidad_total": 75.5,
            "componentes_utilidad": {
                "equidad": 80.0,
                "social": 70.0,
                "legal": 75.0,
                "atencion_consumo": 72.0,
                "atencion_reportes": 78.0,
                "coef_gini": 0.15
            },
            "ranking_colonias": [
                {"colonia": "Lindavista I", "prioridad": 0.85, "ranking": 1},
                {"colonia": "Lindavista II", "prioridad": 0.80, "ranking": 2}
            ],
            "ranking_edificaciones": [
                {"tipo": "Hospital", "prioridad": 0.90, "ranking": 1},
                {"tipo": "Escuelas", "prioridad": 0.85, "ranking": 2}
            ],
            "version_algoritmo": "PSO_v1.0_TEST"
        }
        
        resultado_id = await ResultadoOptimizacionRepository.create_resultado(resultado_prueba)
        print(f"   ✅ Resultado de prueba creado con ID: {resultado_id}")
        
        # Verificar que se creó
        nuevo_total = await ResultadoOptimizacionRepository.count_resultados()
        print(f"   ✅ Nuevo total de resultados: {nuevo_total}\n")
        
        # ============================================================
        # RESUMEN
        # ============================================================
        print("="*70)
        print("✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
        print("="*70)
        print("\n📊 Resumen:")
        print(f"   • Usuarios en DB: {len(todos_usuarios)}")
        print(f"   • Colonias con datos: {len(datos_colonias)}")
        print(f"   • Resultados almacenados: {nuevo_total}")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cerrar conexión
        await Database.close_db()


if __name__ == "__main__":
    asyncio.run(test_repositories())
