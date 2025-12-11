"""
Configuración de fixtures para pruebas unitarias.

Fixtures compartidas:
    - datos_consumo_prueba: Diccionario con consumos de prueba por colonia
    - datos_reportes_prueba: Diccionario con reportes de prueba por colonia
    - pesos_validos: Conjunto de pesos que suman 1.0
"""

import pytest
import numpy as np


# ============================================================================
# FIXTURES - DATOS DE PRUEBA
# ============================================================================

@pytest.fixture
def datos_consumo_prueba():
    """
    Datos de consumo de prueba para 7 colonias.
    Valores en litros, rango realista basado en datos reales.
    """
    return {
        'Capultitlán': 87980.69,
        'Villa GAM': 98406.67,
        'Residencial Zacatenco': 126827.67,
        'Tepeyac Insurgentes': 213349.26,
        'Lindavista I': 369692.29,
        'Magdalena de las Salinas': 440823.13,
        'Lindavista II': 523630.27
    }


@pytest.fixture
def datos_reportes_prueba():
    """
    Datos de reportes de fallas de prueba para 7 colonias.
    Valores enteros representando número de reportes.
    """
    return {
        'Capultitlán': 33,
        'Villa GAM': 79,
        'Residencial Zacatenco': 195,
        'Tepeyac Insurgentes': 319,
        'Lindavista I': 405,
        'Magdalena de las Salinas': 128,
        'Lindavista II': 265,
    }


@pytest.fixture
def pesos_validos():
    """
    Conjunto de pesos válidos para la heurística.
    Restricción: α + β + γ + δ = 1.0
    """
    return {
        'alpha': 0.25,   # Peso legal
        'beta': 0.25,    # Peso social
        'gamma': 0.25,   # Peso consumo
        'delta': 0.25    # Peso reportes
    }


@pytest.fixture
def datos_consumo_iguales():
    """
    Datos donde todos los valores son iguales.
    Útil para testear división por cero en normalización.
    """
    return {
        'Colonia A': 100.0,
        'Colonia B': 100.0,
        'Colonia C': 100.0
    }


@pytest.fixture
def datos_consumo_extremos():
    """
    Datos con valores extremos para testear normalización.
    """
    return {
        'Min': 0.0,
        'Medio': 50.0,
        'Max': 100.0
    }
