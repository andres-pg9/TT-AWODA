import numpy as np
from typing import Dict, List
from normalizacion import (
    COLONIAS, EDIFICACIONES, SOCIAL_NORM, LEGAL_NORM, 
    CONSUMO_NORM, REPORTES_NORM
)

# ============================================================================
# FUNCIÓN HEURÍSTICA
# ============================================================================

def calcular_heuristica(alpha: float, beta: float, gamma: float, delta: float,
                        edificacion: str, colonia: str) -> float:
    """
    Calcula el valor heurístico para una combinación edificación-colonia.

    La heurística combina 4 factores:
    - alpha (α): Peso de la prioridad legal
    - beta (β): Peso de la prioridad social (encuesta ciudadana)
    - gamma (γ): Peso del consumo histórico
    - delta (δ): Peso del número de reportes

    Fórmula: H = α·x + β·y + γ·z + δ·w

    Donde:
        x = prioridad legal normalizada
        y = prioridad social normalizada
        z = consumo histórico normalizado
        w = reportes normalizados

    Args:
        alpha: Peso legal (0-1)
        beta: Peso social (0-1)
        gamma: Peso consumo (0-1)
        delta: Peso reportes (0-1)
        edificacion: Tipo de edificación
        colonia: Nombre de la colonia

    Returns:
        Valor heurístico (0-1), mayor valor = mayor prioridad
    """
    x = SOCIAL_NORM.get(edificacion, 0)
    y = LEGAL_NORM.get(edificacion, 0)
    z = CONSUMO_NORM.get(colonia, 0)
    w = REPORTES_NORM.get(colonia, 0)

    return alpha * x + beta * y + gamma * z + delta * w

# ============================================================================
# FUNCIÓN DE UTILIDAD (OBJETIVO A MAXIMIZAR)
# ============================================================================

def calcular_coeficiente_gini(valores: List[float]) -> float:
    """
    Calcula el coeficiente de Gini para medir desigualdad.

    Gini = 0: Distribución perfectamente equitativa
    Gini = 1: Desigualdad máxima

    En este contexto, medimos qué tan equitativa es la distribución
    de prioridades resultante de la heurística.

    Args:
        valores: Lista de valores a evaluar

    Returns:
        Coeficiente de Gini (0-1)
    """
    if len(valores) == 0:
        return 0

    valores = sorted(valores)
    n = len(valores)
    suma_total = sum(valores)

    if suma_total == 0:
        return 0

    # Fórmula de Gini: (2 * suma_ponderada) / (n * suma_total) - (n+1)/n
    suma_ponderada = sum((i + 1) * v for i, v in enumerate(valores))
    return (2 * suma_ponderada) / (n * suma_total) - (n + 1) / n