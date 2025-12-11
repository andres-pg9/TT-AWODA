"""
Pruebas unitarias para el módulo de normalización (ia/normalizacion.py).

Funciones testeadas:
    - normalizar_valores(): Normalización con piso mínimo
    - normalizar_prioridades(): Normalización simple dividiendo por máximo

Tests incluidos:
    1. test_normalizar_valores_normales: Valores en rango normal
    2. test_normalizar_valores_division_por_cero: Todos los valores iguales
    3. test_normalizar_valores_piso_minimo: Verificar piso del 30%
    4. test_normalizar_valores_rango_valido: Resultado en [0.3, 1.0]
    5. test_normalizar_prioridades_correcta: Normalización simple al máximo
"""

import pytest
import numpy as np
from ia.normalizacion import normalizar_valores, normalizar_prioridades


# ============================================================================
# TESTS - normalizar_valores()
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.normalizacion
def test_normalizar_valores_normales(datos_consumo_prueba):
    """
    Verifica que normalizar_valores funcione con datos normales.
    
    Comportamiento esperado:
        - Los valores se normalizan entre 0.3 y 1.0
        - El valor mínimo original se mapea a 0.3
        - El valor máximo original se mapea a 1.0
    """
    resultado = normalizar_valores(datos_consumo_prueba, piso=0.3)
    
    # Verificar que todos los valores están en el rango correcto
    assert all(0.3 <= v <= 1.0 for v in resultado.values()), \
        "Los valores normalizados deben estar entre 0.3 y 1.0"
    
    # Verificar que el mínimo está cerca de 0.3
    min_val = min(resultado.values())
    assert np.isclose(min_val, 0.3, atol=1e-6), \
        f"El valor mínimo normalizado debe ser 0.3, pero es {min_val}"
    
    # Verificar que el máximo es 1.0
    max_val = max(resultado.values())
    assert np.isclose(max_val, 1.0, atol=1e-6), \
        f"El valor máximo normalizado debe ser 1.0, pero es {max_val}"
    
    # Verificar que se mantiene el orden relativo
    assert resultado['Capultitlán'] < resultado['Lindavista II'], \
        "El orden relativo debe mantenerse después de normalizar"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.normalizacion
def test_normalizar_valores_division_por_cero(datos_consumo_iguales):
    """
    Verifica el manejo de división por cero cuando todos los valores son iguales.
    
    Comportamiento esperado:
        - Cuando max == min, todos los valores se normalizan a 1.0
        - No debe lanzar excepciones
    """
    resultado = normalizar_valores(datos_consumo_iguales, piso=0.3)
    
    # Todos los valores deben ser 1.0
    assert all(np.isclose(v, 1.0, atol=1e-6) for v in resultado.values()), \
        "Cuando todos los valores son iguales, deben normalizarse a 1.0"
    
    # Verificar que tiene las mismas claves
    assert set(resultado.keys()) == set(datos_consumo_iguales.keys()), \
        "Las claves del diccionario deben mantenerse"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.normalizacion
def test_normalizar_valores_piso_minimo():
    """
    Verifica que el parámetro 'piso' funcione correctamente.
    
    Comportamiento esperado:
        - Con piso=0.5, los valores deben estar en [0.5, 1.0]
        - El valor mínimo se mapea exactamente al piso
    """
    datos = {'A': 10, 'B': 50, 'C': 100}
    resultado = normalizar_valores(datos, piso=0.5)
    
    # Verificar rango [0.5, 1.0]
    assert all(0.5 <= v <= 1.0 for v in resultado.values()), \
        "Con piso=0.5, los valores deben estar entre 0.5 y 1.0"
    
    # Verificar que el mínimo es exactamente 0.5
    assert np.isclose(min(resultado.values()), 0.5, atol=1e-6), \
        "El valor mínimo debe ser igual al piso especificado"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.normalizacion
def test_normalizar_valores_rango_valido(datos_consumo_extremos):
    """
    Verifica que los valores normalizados estén siempre en el rango válido.
    
    Comportamiento esperado:
        - Con valores extremos (0, 50, 100), la normalización debe funcionar
        - Los valores resultantes deben estar en [piso, 1.0]
    """
    piso = 0.3
    resultado = normalizar_valores(datos_consumo_extremos, piso=piso)
    
    # Verificar que todos están en el rango
    for clave, valor in resultado.items():
        assert piso <= valor <= 1.0, \
            f"Valor para {clave} ({valor}) está fuera del rango [{piso}, 1.0]"
    
    # Verificar interpolación correcta
    # Min (0) -> 0.3, Medio (50) -> 0.65, Max (100) -> 1.0
    assert np.isclose(resultado['Min'], 0.3, atol=1e-6)
    assert np.isclose(resultado['Max'], 1.0, atol=1e-6)
    assert 0.3 < resultado['Medio'] < 1.0


# ============================================================================
# TESTS - normalizar_prioridades()
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.normalizacion
def test_normalizar_prioridades_correcta():
    """
    Verifica que normalizar_prioridades divida correctamente por el máximo.
    
    Comportamiento esperado:
        - Todos los valores se dividen por el valor máximo
        - El valor máximo se normaliza a 1.0
        - Los demás valores mantienen su proporción relativa
    """
    prioridades = {
        'Hospital': 7,
        'Clínicas': 6,
        'Escuelas': 5,
        'Casas': 4
    }
    
    resultado = normalizar_prioridades(prioridades)
    
    # Verificar que el máximo es 1.0
    assert np.isclose(max(resultado.values()), 1.0, atol=1e-6), \
        "El valor máximo debe normalizarse a 1.0"
    
    # Verificar proporciones relativas
    assert np.isclose(resultado['Hospital'], 7/7, atol=1e-6)
    assert np.isclose(resultado['Clínicas'], 6/7, atol=1e-6)
    assert np.isclose(resultado['Escuelas'], 5/7, atol=1e-6)
    assert np.isclose(resultado['Casas'], 4/7, atol=1e-6)
    
    # Verificar que todos están en [0, 1]
    assert all(0 <= v <= 1.0 for v in resultado.values()), \
        "Todos los valores normalizados deben estar entre 0 y 1"
