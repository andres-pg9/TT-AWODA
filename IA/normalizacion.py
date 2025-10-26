import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

# ============================================================================
# CONFIGURACIÓN DE DATOS
# ============================================================================

# Lista de colonias a analizar
COLONIAS = [
    "Capultitlán",
    "Villa GAM",
    "Residencial Zacatenco",
    "Tepeyac Insurgentes",
    "Lindavista I",
    "Magdalena de las Salinas",
    "Lindavista II"
]

# Tipos de edificaciones consideradas
EDIFICACIONES = [
    'Hospital',
    'Clínicas Particulares',
    'Escuelas',
    'Casas',
    'Dependencias de Gobierno',
    'Comercios',
    'Centros Comerciales'
]

# Prioridades según encuesta ciudadana (escala 1-7, donde 7 es más importante)
SOCIAL = {
    'Hospital': 7,
    'Clínicas Particulares': 6,
    'Escuelas': 5,
    'Casas': 4,
    'Dependencias de Gobierno': 3,
    'Comercios': 2,
    'Centros Comerciales': 1
}

# Prioridades según marco legal (escala 1-3, donde 3 es más importante)
# Basado en Ley de Aguas Nacionales y normativas de CDMX
LEGAL = {
    'Hospital': 3,
    'Casas': 3,
    'Clínicas Particulares': 3,
    'Comercios': 2,
    'Centros Comerciales': 2,
    'Escuelas': 1,
    'Dependencias de Gobierno': 1
}

CONSUMO = {
    'Capultitlán': 6,
    'Villa GAM': 5,
    'Residencial Zacatenco': 2,
    'Tepeyac Insurgentes': 1,
    'Lindavista I': 3,
    'Magdalena de las Salinas': 7,
    'Lindavista II': 4,
}

REPORTES = {
    'Capultitlán': 1,
    'Villa GAM': 2,
    'Residencial Zacatenco': 7,
    'Tepeyac Insurgentes': 4,
    'Lindavista I': 5,
    'Magdalena de las Salinas': 3,
    'Lindavista II': 6,
}

"""
# Prioridades según marco legal (escala 1-3, donde 3 es más importante)
# Basado en Ley de Aguas Nacionales y normativas de CDMX
LEGAL = {
    'Hospital': 3,
    'Casas': 3,
    'Clínicas Particulares': 3,
    'Comercios': 2,
    'Centros Comerciales': 2,
    'Escuelas': 1,
    'Dependencias de Gobierno': 1
}

# Consumo histórico promedio por colonia (en litros)
# Fuente: Datos abiertos CDMX
CONSUMO = {
    'Capultitlán': 87980.69,
    'Villa GAM': 98406.67,
    'Residencial Zacatenco': 126827.67,
    'Tepeyac Insurgentes': 213349.26,
    'Lindavista I': 369692.29,
    'Magdalena de las Salinas': 440823.13,
    'Lindavista II': 523630.27
}
"""

# ============================================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================================

def normalizar_valores(valores: dict, piso: float = 0.3) -> dict:
    """
    Normaliza con un piso o valor mínimo (30%) para evitar ceros absolutos pero mantener alta diferenciación
    Esta función escala los valores entre 'piso' y 1 para las variables de consumo y reportes.
    """
    min_val = min(valores.values())
    max_val = max(valores.values())
    
    if max_val == min_val:
        return {k: 1.0 for k in valores.keys()}
    
    norm = {k: (v - min_val) / (max_val - min_val) for k, v in valores.items()}
    return {k: piso + (1 - piso) * v for k, v in norm.items()}


def normalizar_prioridades(prioridades: dict) -> dict:
    """
    Normaliza prioridades dividiendo entre el valor máximo.

    Args:
        prioridades: Diccionario con valores de prioridad

    Returns:
        Diccionario con prioridades normalizadas entre 0 y 1
        
    Esta funcion escala las prioridades para que el valor más alto sea 1 
    manteniendo la proporción relativa entre ellas para las variables social y legal.
    """
    max_val = max(prioridades.values())
    return {k: v / max_val for k, v in prioridades.items()}


# Aplicar normalización a todos los datos
SOCIAL_NORM = normalizar_prioridades(SOCIAL)
LEGAL_NORM = normalizar_prioridades(LEGAL)
CONSUMO_NORM = normalizar_valores(CONSUMO)
REPORTES_NORM = normalizar_valores(REPORTES)
