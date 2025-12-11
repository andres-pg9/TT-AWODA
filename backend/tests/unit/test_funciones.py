"""
Pruebas unitarias para funciones heurísticas y de utilidad (ia/funciones.py).

Funciones testeadas:
    - calcular_heuristica(): Combina 4 pesos con valores normalizados
    - calcular_coeficiente_gini(): Mide desigualdad en distribución
    - calcular_utilidad(): Función multiobjetivo con 5 componentes

Tests incluidos:
    1. test_heuristica_pesos_validos: Heurística con pesos que suman 1
    2. test_heuristica_combinacion_correcta: Fórmula correcta α·x + β·y + γ·z + δ·w
    3. test_gini_lista_vacia: Gini retorna 0 con lista vacía
    4. test_gini_valores_iguales: Gini = 0 cuando no hay desigualdad
    5. test_gini_desigualdad_maxima: Gini cercano a 1 con desigualdad extrema
    6. test_utilidad_componentes_correctos: Utilidad tiene 5 componentes en rango 0-100
"""

import pytest
import numpy as np
from ia.funciones import (
    calcular_heuristica,
    calcular_coeficiente_gini,
    calcular_utilidad
)
from ia.normalizacion import normalizar_valores


# ============================================================================
# TESTS - calcular_heuristica()
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.heuristica
def test_heuristica_pesos_validos(pesos_validos, datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que calcular_heuristica funcione con pesos válidos.
    
    Comportamiento esperado:
        - La función debe ejecutarse sin errores
        - El resultado debe estar en el rango [0, 1]
        - Con pesos que suman 1, el resultado es una combinación convexa
    """
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    resultado = calcular_heuristica(
        alpha=pesos_validos['alpha'],
        beta=pesos_validos['beta'],
        gamma=pesos_validos['gamma'],
        delta=pesos_validos['delta'],
        edificacion='Hospital',
        colonia='Capultitlán',
        consumo_norm=consumo_norm,
        reportes_norm=reportes_norm
    )
    
    # Verificar que el resultado está en [0, 1]
    assert 0 <= resultado <= 1.0, \
        f"La heurística debe estar en [0, 1], pero es {resultado}"
    
    # Verificar que es un número válido
    assert not np.isnan(resultado), "El resultado no debe ser NaN"
    assert not np.isinf(resultado), "El resultado no debe ser infinito"


@pytest.mark.unit
@pytest.mark.ia
@pytest.mark.heuristica
def test_heuristica_combinacion_correcta():
    """
    Verifica que la fórmula H = α·x + β·y + γ·z + δ·w se aplique correctamente.
    
    Comportamiento esperado:
        - La heurística es una suma ponderada de 4 componentes
        - Con pesos extremos, el resultado refleja el componente dominante
    """
    consumo_norm = {'Colonia A': 0.5}
    reportes_norm = {'Colonia A': 0.5}
    
    # Test 1: Solo peso legal (alpha=1)
    h1 = calcular_heuristica(1.0, 0.0, 0.0, 0.0, 'Hospital', 'Colonia A',
                             consumo_norm, reportes_norm)
    
    # Test 2: Solo peso social (beta=1)  
    h2 = calcular_heuristica(0.0, 1.0, 0.0, 0.0, 'Hospital', 'Colonia A',
                             consumo_norm, reportes_norm)
    
    # Test 3: Solo peso consumo (gamma=1)
    h3 = calcular_heuristica(0.0, 0.0, 1.0, 0.0, 'Hospital', 'Colonia A',
                             consumo_norm, reportes_norm)
    
    # Test 4: Solo peso reportes (delta=1)
    h4 = calcular_heuristica(0.0, 0.0, 0.0, 1.0, 'Hospital', 'Colonia A',
                             consumo_norm, reportes_norm)
    
    # Verificar que son valores válidos en [0, 1]
    assert 0 <= h1 <= 1.0, f"h1 debe estar en [0, 1], pero es {h1}"
    assert 0 <= h2 <= 1.0, f"h2 debe estar en [0, 1], pero es {h2}"
    assert np.isclose(h3, 0.5, atol=1e-6), "Gamma puro debe ser 0.5"
    assert np.isclose(h4, 0.5, atol=1e-6), "Delta puro debe ser 0.5"
    
    # Test 5: Combinación balanceada
    h_balance = calcular_heuristica(0.25, 0.25, 0.25, 0.25, 'Hospital', 'Colonia A',
                                    consumo_norm, reportes_norm)
    assert 0 <= h_balance <= 1.0, "Combinación balanceada debe estar en [0, 1]"
    
    # Test 6: Verificar que diferentes edificaciones dan diferentes resultados
    h_hospital = calcular_heuristica(0.5, 0.5, 0.0, 0.0, 'Hospital', 'Colonia A',
                                     consumo_norm, reportes_norm)
    h_comercio = calcular_heuristica(0.5, 0.5, 0.0, 0.0, 'Comercios', 'Colonia A',
                                     consumo_norm, reportes_norm)
    assert h_hospital != h_comercio, "Hospital y Comercios deben tener heurísticas diferentes"


# ============================================================================
# TESTS - calcular_coeficiente_gini()
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
def test_gini_lista_vacia():
    """
    Verifica que Gini retorne 0 con lista vacía.
    
    Comportamiento esperado:
        - No debe lanzar excepciones
        - Debe retornar 0 (sin desigualdad porque no hay datos)
    """
    resultado = calcular_coeficiente_gini([])
    assert resultado == 0, "Gini con lista vacía debe ser 0"


@pytest.mark.unit
@pytest.mark.ia
def test_gini_valores_iguales():
    """
    Verifica que Gini sea 0 cuando todos los valores son iguales.
    
    Comportamiento esperado:
        - Gini = 0 indica distribución perfectamente equitativa
        - Cuando todos tienen lo mismo, no hay desigualdad
    """
    valores = [10, 10, 10, 10, 10]
    resultado = calcular_coeficiente_gini(valores)
    
    assert np.isclose(resultado, 0, atol=1e-6), \
        f"Gini con valores iguales debe ser 0, pero es {resultado}"


@pytest.mark.unit
@pytest.mark.ia
def test_gini_desigualdad_maxima():
    """
    Verifica que Gini sea cercano a 1 con desigualdad extrema.
    
    Comportamiento esperado:
        - Gini → 1 cuando una persona tiene todo y los demás nada
        - Con [0, 0, 0, 100], Gini debe ser alto
    """
    valores = [0, 0, 0, 100]
    resultado = calcular_coeficiente_gini(valores)
    
    # Gini debe ser alto (cercano a 1, pero no exactamente 1)
    assert resultado > 0.5, \
        f"Gini con desigualdad extrema debe ser > 0.5, pero es {resultado}"
    assert resultado <= 1.0, \
        f"Gini no puede ser mayor a 1, pero es {resultado}"


# ============================================================================
# TESTS - calcular_utilidad()
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
def test_utilidad_componentes_correctos(pesos_validos, datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que calcular_utilidad retorne todos los componentes esperados.
    
    Comportamiento esperado:
        - Debe retornar diccionario con 7 claves
        - Todos los valores deben estar en rango 0-100
        - utilidad_total es combinación ponderada de los 5 componentes
    """
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    resultado = calcular_utilidad(
        alpha=pesos_validos['alpha'],
        beta=pesos_validos['beta'],
        gamma=pesos_validos['gamma'],
        delta=pesos_validos['delta'],
        consumo_norm=consumo_norm,
        reportes_norm=reportes_norm
    )
    
    # Verificar que tiene las 7 claves esperadas
    claves_esperadas = {
        'utilidad_total',
        'equidad',
        'satisfaccion_social',
        'cumplimiento_legal',
        'atencion_consumo',
        'atencion_reportes',
        'coeficiente_gini'
    }
    assert set(resultado.keys()) == claves_esperadas, \
        f"Faltan claves en el resultado. Esperadas: {claves_esperadas}, Recibidas: {set(resultado.keys())}"
    
    # Verificar rangos de los componentes (0-100)
    for clave in ['equidad', 'satisfaccion_social', 'cumplimiento_legal', 
                  'atencion_consumo', 'atencion_reportes', 'utilidad_total']:
        valor = resultado[clave]
        assert 0 <= valor <= 100, \
            f"{clave} debe estar en [0, 100], pero es {valor}"
    
    # Verificar que Gini está en [0, 1]
    assert 0 <= resultado['coeficiente_gini'] <= 1.0, \
        f"coeficiente_gini debe estar en [0, 1], pero es {resultado['coeficiente_gini']}"


@pytest.mark.unit
@pytest.mark.ia
def test_utilidad_suma_ponderada(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que utilidad_total sea la suma ponderada correcta de componentes.
    
    Comportamiento esperado:
        - utilidad_total = 0.30·equidad + 0.25·social + 0.25·legal + 0.10·consumo + 0.10·reportes
        - Los pesos deben sumar 1.0
    """
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    resultado = calcular_utilidad(0.25, 0.25, 0.25, 0.25,
                                  consumo_norm=consumo_norm,
                                  reportes_norm=reportes_norm)
    
    # Calcular manualmente la utilidad total
    w_equidad, w_social, w_legal, w_consumo, w_reportes = 0.30, 0.25, 0.25, 0.10, 0.10
    utilidad_calculada = (
        w_equidad * resultado['equidad'] +
        w_social * resultado['satisfaccion_social'] +
        w_legal * resultado['cumplimiento_legal'] +
        w_consumo * resultado['atencion_consumo'] +
        w_reportes * resultado['atencion_reportes']
    )
    
    # Verificar que coincide con el valor retornado
    assert np.isclose(resultado['utilidad_total'], utilidad_calculada, atol=1e-3), \
        f"utilidad_total ({resultado['utilidad_total']:.2f}) no coincide con el cálculo manual ({utilidad_calculada:.2f})"
    
    # Verificar que los pesos suman 1.0
    suma_pesos = w_equidad + w_social + w_legal + w_consumo + w_reportes
    assert np.isclose(suma_pesos, 1.0, atol=1e-6), \
        f"Los pesos de la utilidad deben sumar 1.0, pero suman {suma_pesos}"
