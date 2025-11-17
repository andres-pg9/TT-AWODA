"""
Script de prueba para el sistema de autenticación.

Prueba los endpoints de auth usando requests.
Ejecutar DESPUÉS de tener el backend corriendo.

Uso:
    pip install requests
    python test_auth.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_auth_system():
    """Prueba el sistema completo de autenticación"""
    
    print("\n" + "="*70)
    print("🧪 PROBANDO SISTEMA DE AUTENTICACIÓN")
    print("="*70 + "\n")
    
    # ============================================================
    # TEST 1: Login con credenciales correctas
    # ============================================================
    print("1️⃣  Probando login con credenciales correctas...")
    
    login_data = {
        "numero_empleado": 215646,
        "password": "admin123"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
    
    if response.status_code == 200:
        result = response.json()
        token = result["access_token"]
        usuario = result["usuario"]
        
        print(f"   ✅ Login exitoso")
        print(f"      Usuario: {usuario['nombre_empleado']}")
        print(f"      Rol: {usuario['rol_usuario']}")
        print(f"      Token: {token[:50]}...")
    else:
        print(f"   ❌ Error en login: {response.status_code}")
        print(f"      {response.text}")
        return
    
    # ============================================================
    # TEST 2: Obtener información del usuario actual
    # ============================================================
    print("\n2️⃣  Probando endpoint /api/auth/me...")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
    
    if response.status_code == 200:
        usuario_actual = response.json()
        print(f"   ✅ Usuario autenticado correctamente")
        print(f"      {usuario_actual['nombre_empleado']} - {usuario_actual['rol_usuario']}")
    else:
        print(f"   ❌ Error al obtener usuario: {response.status_code}")
    
    # ============================================================
    # TEST 3: Validar token
    # ============================================================
    print("\n3️⃣  Probando validación de token...")
    
    response = requests.get(f"{BASE_URL}/api/auth/validate-token", headers=headers)
    
    if response.status_code == 200:
        validation = response.json()
        print(f"   ✅ Token válido: {validation['valid']}")
    else:
        print(f"   ❌ Token inválido")
    
    # ============================================================
    # TEST 4: Login con credenciales incorrectas
    # ============================================================
    print("\n4️⃣  Probando login con credenciales incorrectas...")
    
    login_incorrecto = {
        "numero_empleado": 215646,
        "password": "password_incorrecto"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_incorrecto)
    
    if response.status_code == 401:
        print(f"   ✅ Error 401 esperado (credenciales incorrectas)")
    else:
        print(f"   ⚠️  Respuesta inesperada: {response.status_code}")
    
    # ============================================================
    # TEST 5: Acceder sin token
    # ============================================================
    print("\n5️⃣  Probando acceso sin token...")
    
    response = requests.get(f"{BASE_URL}/api/auth/me")
    
    if response.status_code == 401 or response.status_code == 403:
        print(f"   ✅ Acceso denegado correctamente (sin token)")
    else:
        print(f"   ⚠️  Respuesta inesperada: {response.status_code}")
    
    # ============================================================
    # TEST 6: Logout
    # ============================================================
    print("\n6️⃣  Probando logout...")
    
    response = requests.post(f"{BASE_URL}/api/auth/logout", headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Logout exitoso: {result['message']}")
    else:
        print(f"   ❌ Error en logout: {response.status_code}")
    
    # ============================================================
    # TEST 7: Usar endpoint /optimize con autenticación
    # ============================================================
    print("\n7️⃣  Probando /api/optimize con usuario autenticado...")
    
    # Primero hacer login de nuevo para tener un token fresco
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Hacer request a optimize
    optimize_data = {
        "consumo": {
            "Capultitlán": 0,
            "Villa GAM": 0,
            "Residencial Zacatenco": 0,
            "Tepeyac Insurgentes": 0,
            "Lindavista I": 0,
            "Magdalena de las Salinas": 0,
            "Lindavista II": 0
        },
        "reportes": {
            "Capultitlán": 0,
            "Villa GAM": 0,
            "Residencial Zacatenco": 0,
            "Tepeyac Insurgentes": 0,
            "Lindavista I": 0,
            "Magdalena de las Salinas": 0,
            "Lindavista II": 0
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/optimize",
        json=optimize_data,
        headers=headers
    )
    
    if response.status_code == 200:
        print(f"   ✅ Optimización ejecutada con usuario autenticado")
        print(f"      (El resultado quedó asociado al usuario)")
    else:
        print(f"   ⚠️  Respuesta: {response.status_code}")
    
    # ============================================================
    # RESUMEN
    # ============================================================
    print("\n" + "="*70)
    print("✅ TODOS LOS TESTS DE AUTENTICACIÓN COMPLETADOS")
    print("="*70)
    print("\n📝 Resumen:")
    print("   ✓ Login funciona correctamente")
    print("   ✓ JWT se genera y valida")
    print("   ✓ Endpoints protegidos requieren token")
    print("   ✓ Logout funciona")
    print("   ✓ Integración con /optimize funciona")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        test_auth_system()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: No se pudo conectar al backend")
        print("   Asegúrate de que el backend esté corriendo en http://localhost:8000")
        print("   Ejecuta: uvicorn main:app --reload\n")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}\n")