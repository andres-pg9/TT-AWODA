"""
Pruebas unitarias para cálculo de rankings y resultados (ia/resultados.py).

Funciones testeadas:
    - calcular_rankings(): Genera rankings de colonias y edificaciones
    - imprimir_resultados_detallados(): Formatea resultados para salida

Tests incluidos:
    1. test_calcular_rankings_estructura: Estructura correcta de DataFrames
    2. test_calcular_rankings_orden: Rankings ordenados descendentemente
    3. test_imprimir_resultados_modo_json: Formato JSON válido
    4. test_imprimir_resultados_contenido: Contenido completo en JSON
"""

import pytest
import numpy as np
import pandas as pd
from ia.resultados import calcular_rankings, imprimir_resultados_detallados
from ia.normalizacion import normalizar_valores


# ============================================================================
# TESTS - calcular_rankings()
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
def test_calcular_rankings_estructura(pesos_validos, datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que calcular_rankings retorne DataFrames con la estructura correcta.
    
    Comportamiento esperado:
        - Debe retornar una tupla con 2 DataFrames
        - DataFrame de colonias: columnas ['Colonia', 'Prioridad', 'Ranking']
        - DataFrame de edificaciones: columnas ['Edificación', 'Prioridad', 'Ranking']
    """
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    df_colonias, df_edificaciones = calcular_rankings(
        alpha=pesos_validos['alpha'],
        beta=pesos_validos['beta'],
        gamma=pesos_validos['gamma'],
        delta=pesos_validos['delta'],
        consumo_norm=consumo_norm,
        reportes_norm=reportes_norm
    )
    
    # Verificar tipos
    assert isinstance(df_colonias, pd.DataFrame), \
        "df_colonias debe ser un DataFrame"
    assert isinstance(df_edificaciones, pd.DataFrame), \
        "df_edificaciones debe ser un DataFrame"
    
    # Verificar columnas de colonias
    assert list(df_colonias.columns) == ['Colonia', 'Prioridad', 'Ranking'], \
        f"Columnas incorrectas en df_colonias: {list(df_colonias.columns)}"
    
    # Verificar columnas de edificaciones
    assert list(df_edificaciones.columns) == ['Edificación', 'Prioridad', 'Ranking'], \
        f"Columnas incorrectas en df_edificaciones: {list(df_edificaciones.columns)}"
    
    # Verificar número de filas
    assert len(df_colonias) == 7, \
        f"Debe haber 7 colonias, pero hay {len(df_colonias)}"
    assert len(df_edificaciones) == 7, \
        f"Debe haber 7 edificaciones, pero hay {len(df_edificaciones)}"
    
    # Verificar que no hay valores nulos
    assert not df_colonias.isnull().any().any(), \
        "df_colonias no debe tener valores nulos"
    assert not df_edificaciones.isnull().any().any(), \
        "df_edificaciones no debe tener valores nulos"


@pytest.mark.unit
@pytest.mark.ia
def test_calcular_rankings_orden(pesos_validos, datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que los rankings estén ordenados correctamente.
    
    Comportamiento esperado:
        - Las prioridades deben estar en orden descendente
        - Los rankings deben ser consecutivos: 1, 2, 3, ...
        - Mayor prioridad = ranking más bajo (1 es el mejor)
    """
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    df_colonias, df_edificaciones = calcular_rankings(
        alpha=0.3, beta=0.3, gamma=0.2, delta=0.2,
        consumo_norm=consumo_norm,
        reportes_norm=reportes_norm
    )
    
    # Verificar orden descendente de prioridades en colonias
    prioridades_colonias = df_colonias['Prioridad'].tolist()
    assert prioridades_colonias == sorted(prioridades_colonias, reverse=True), \
        "Las prioridades de colonias deben estar en orden descendente"
    
    # Verificar orden descendente de prioridades en edificaciones
    prioridades_edificaciones = df_edificaciones['Prioridad'].tolist()
    assert prioridades_edificaciones == sorted(prioridades_edificaciones, reverse=True), \
        "Las prioridades de edificaciones deben estar en orden descendente"
    
    # Verificar rankings consecutivos en colonias
    rankings_colonias = df_colonias['Ranking'].tolist()
    assert rankings_colonias == list(range(1, len(df_colonias) + 1)), \
        f"Los rankings deben ser [1, 2, ..., 7], pero son {rankings_colonias}"
    
    # Verificar rankings consecutivos en edificaciones
    rankings_edificaciones = df_edificaciones['Ranking'].tolist()
    assert rankings_edificaciones == list(range(1, len(df_edificaciones) + 1)), \
        f"Los rankings deben ser [1, 2, ..., 7], pero son {rankings_edificaciones}"
    
    # Verificar que el ranking 1 tiene la mayor prioridad
    assert df_colonias.iloc[0]['Ranking'] == 1, \
        "El primer elemento debe tener ranking 1"
    assert df_colonias.iloc[0]['Prioridad'] == max(prioridades_colonias), \
        "El ranking 1 debe tener la mayor prioridad"


# ============================================================================
# TESTS - imprimir_resultados_detallados()
# ============================================================================

@pytest.mark.unit
@pytest.mark.ia
def test_imprimir_resultados_modo_json(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que imprimir_resultados_detallados retorne JSON válido en modo_json=True.
    
    Comportamiento esperado:
        - Debe retornar un diccionario
        - Debe tener las claves: utilidad_total, pesos_optimos, colonias, edificaciones
        - Los valores deben tener la estructura correcta
    """
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    pesos_optimos = np.array([0.25, 0.25, 0.25, 0.25])
    from ia.funciones import calcular_utilidad
    resultado = calcular_utilidad(*pesos_optimos, 
                                  consumo_norm=consumo_norm,
                                  reportes_norm=reportes_norm)
    
    json_resultado = imprimir_resultados_detallados(
        pesos_optimos=pesos_optimos,
        resultado=resultado,
        modo_json=True,
        consumo_norm=consumo_norm,
        reportes_norm=reportes_norm
    )
    
    # Verificar que es un diccionario
    assert isinstance(json_resultado, dict), \
        "El resultado en modo_json debe ser un diccionario"
    
    # Verificar claves principales
    claves_esperadas = {'utilidad_total', 'pesos_optimos', 'colonias', 'edificaciones'}
    assert set(json_resultado.keys()) == claves_esperadas, \
        f"Claves incorrectas. Esperadas: {claves_esperadas}, Recibidas: {set(json_resultado.keys())}"
    
    # Verificar estructura de pesos_optimos
    assert isinstance(json_resultado['pesos_optimos'], dict), \
        "pesos_optimos debe ser un diccionario"
    assert set(json_resultado['pesos_optimos'].keys()) == {'α', 'β', 'γ', 'δ'}, \
        "pesos_optimos debe tener claves α, β, γ, δ"
    
    # Verificar que los pesos son números
    for clave, valor in json_resultado['pesos_optimos'].items():
        assert isinstance(valor, (int, float)), \
            f"El peso {clave} debe ser numérico, pero es {type(valor)}"


@pytest.mark.unit
@pytest.mark.ia
def test_imprimir_resultados_contenido(datos_consumo_prueba, datos_reportes_prueba):
    """
    Verifica que el contenido del JSON sea completo y coherente.
    
    Comportamiento esperado:
        - Debe tener 7 colonias y 7 edificaciones
        - Cada elemento debe tener: nombre, prioridad, ranking
        - Los rankings deben ser válidos (1 a 7)
        - La utilidad_total debe coincidir con el resultado
    """
    consumo_norm = normalizar_valores(datos_consumo_prueba, piso=0.3)
    reportes_norm = normalizar_valores(datos_reportes_prueba, piso=0.3)
    
    pesos_optimos = np.array([0.3, 0.3, 0.2, 0.2])
    from ia.funciones import calcular_utilidad
    resultado = calcular_utilidad(*pesos_optimos,
                                  consumo_norm=consumo_norm,
                                  reportes_norm=reportes_norm)
    
    json_resultado = imprimir_resultados_detallados(
        pesos_optimos=pesos_optimos,
        resultado=resultado,
        modo_json=True,
        consumo_norm=consumo_norm,
        reportes_norm=reportes_norm
    )
    
    # Verificar número de colonias
    assert len(json_resultado['colonias']) == 7, \
        f"Debe haber 7 colonias, pero hay {len(json_resultado['colonias'])}"
    
    # Verificar número de edificaciones
    assert len(json_resultado['edificaciones']) == 7, \
        f"Debe haber 7 edificaciones, pero hay {len(json_resultado['edificaciones'])}"
    
    # Verificar estructura de cada colonia
    for colonia in json_resultado['colonias']:
        assert 'nombre' in colonia, "Cada colonia debe tener 'nombre'"
        assert 'prioridad' in colonia, "Cada colonia debe tener 'prioridad'"
        assert 'ranking' in colonia, "Cada colonia debe tener 'ranking'"
        assert 1 <= colonia['ranking'] <= 7, \
            f"El ranking debe estar entre 1 y 7, pero es {colonia['ranking']}"
    
    # Verificar estructura de cada edificación
    for edificacion in json_resultado['edificaciones']:
        assert 'nombre' in edificacion, "Cada edificación debe tener 'nombre'"
        assert 'prioridad' in edificacion, "Cada edificación debe tener 'prioridad'"
        assert 'ranking' in edificacion, "Cada edificación debe tener 'ranking'"
        assert 1 <= edificacion['ranking'] <= 7, \
            f"El ranking debe estar entre 1 y 7, pero es {edificacion['ranking']}"
    
    # Verificar que utilidad_total coincide
    assert np.isclose(json_resultado['utilidad_total'], 
                     resultado['utilidad_total'], atol=1e-3), \
        f"utilidad_total en JSON debe coincidir con el resultado"
    
    # Verificar que los pesos suman 1
    suma_pesos = sum(json_resultado['pesos_optimos'].values())
    assert np.isclose(suma_pesos, 1.0, atol=1e-4), \
        f"Los pesos deben sumar 1.0, pero suman {suma_pesos}"
    
    # Verificar que los rankings son únicos y consecutivos
    rankings_colonias = [c['ranking'] for c in json_resultado['colonias']]
    assert sorted(rankings_colonias) == list(range(1, 8)), \
        "Los rankings de colonias deben ser [1, 2, 3, 4, 5, 6, 7]"
    
    rankings_edificaciones = [e['ranking'] for e in json_resultado['edificaciones']]
    assert sorted(rankings_edificaciones) == list(range(1, 8)), \
        "Los rankings de edificaciones deben ser [1, 2, 3, 4, 5, 6, 7]"
