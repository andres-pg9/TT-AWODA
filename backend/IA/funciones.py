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


def calcular_utilidad(alpha: float, beta: float, gamma: float, delta: float) -> Dict:
    """
    Calcula la utilidad multiobjetivo de una configuración de pesos.

    La utilidad combina 5 componentes:
    1. EQUIDAD (25%): Distribución justa medida por coef. de Gini
    2. SATISFACCIÓN SOCIAL (30%): Alineación con preferencias ciudadanas
    3. CUMPLIMIENTO LEGAL (25%): Respeto a normativas oficiales
    4. ATENCIÓN A ALTO CONSUMO (10%): Prioriza zonas con mayor consumo histórico
    5. ATENCIÓN A REPORTES (10%): Prioriza zonas con más problemas reportados

    Utilidad Total = 0.25·Equidad + 0.30·Social + 0.25·Legal + 0.10·Consumo + 0.10·Reportes

    Args:
        alpha, beta, gamma, delta: Pesos de la heurística

    Returns:
        Diccionario con utilidad total y sus componentes (escala 0-100)
    """
    # Calcular heurística para todas las combinaciones colonia-edificación
    valores_heuristica = []
    ponderacion_social = []
    ponderacion_legal = []
    ponderacion_consumo = []
    ponderacion_reportes = []

    for colonia in COLONIAS:
        for edificacion in EDIFICACIONES:
            # Valor heurístico
            h = calcular_heuristica(alpha, beta, gamma, delta, edificacion, colonia)
            valores_heuristica.append(h)

            # Componentes individuales ponderados
            ponderacion_social.append(h * SOCIAL_NORM[edificacion])
            ponderacion_legal.append(h * LEGAL_NORM[edificacion])
            ponderacion_consumo.append(h * CONSUMO_NORM[colonia])
            ponderacion_reportes.append(h * REPORTES_NORM[colonia])

    # COMPONENTE 1: EQUIDAD (25%)
    # Medida mediante coeficiente de Gini invertido
    # Menor Gini = Mayor equidad
    gini = calcular_coeficiente_gini(valores_heuristica)
    equidad = 100 * (1 - gini)

    # COMPONENTE 2: SATISFACCIÓN SOCIAL (30%)
    # Qué tanto se alinean las prioridades con la encuesta ciudadana
    suma_social = sum(ponderacion_social)
    suma_total = sum(valores_heuristica)
    satisfaccion_social = 100 * (suma_social / suma_total if suma_total > 0 else 0)

    # COMPONENTE 3: CUMPLIMIENTO LEGAL (25%)
    # Qué tanto se respetan las prioridades legales
    suma_legal = sum(ponderacion_legal)
    cumplimiento_legal = 100 * (suma_legal / suma_total if suma_total > 0 else 0)

    # COMPONENTE 4: ATENCIÓN A ALTO CONSUMO (10%)
    # Qué tanto se prioriza a colonias con mayor consumo histórico
    suma_consumo = sum(ponderacion_consumo)
    atencion_consumo = 100 * (suma_consumo / suma_total if suma_total > 0 else 0)

    # COMPONENTE 5: ATENCIÓN A REPORTES (10%)
    # Qué tanto se prioriza a colonias con más reportes de fallas
    suma_reportes = sum(ponderacion_reportes)
    atencion_reportes = 100 * (suma_reportes / suma_total if suma_total > 0 else 0)

    # UTILIDAD TOTAL (ponderada)
    w_equidad, w_social, w_legal, w_consumo, w_reportes = 0.30, 0.25, 0.25, 0.10, 0.10 #Suma 1.0
    utilidad_total = (w_equidad * equidad + 
                    w_social * satisfaccion_social + 
                    w_legal * cumplimiento_legal +
                    w_consumo * atencion_consumo +
                    w_reportes * atencion_reportes)

    return {
        'utilidad_total': utilidad_total,
        'equidad': equidad,
        'satisfaccion_social': satisfaccion_social,
        'cumplimiento_legal': cumplimiento_legal,
        'atencion_consumo': atencion_consumo,
        'atencion_reportes': atencion_reportes,
        'coeficiente_gini': gini
    }
