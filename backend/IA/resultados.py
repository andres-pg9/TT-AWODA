import numpy as np
import pandas as pd
from typing import Tuple, Dict
from normalizacion import COLONIAS, EDIFICACIONES
from funciones import calcular_heuristica

# ============================================================================
# FUNCIONES DE RANKING
# ============================================================================

def calcular_rankings(alpha: float, beta: float, gamma: float, delta: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula rankings de prioridad para colonias y edificaciones.

    Para cada colonia: promedia la heurística de todas sus edificaciones
    Para cada edificación: promedia la heurística en todas las colonias

    Args:
        alpha, beta, gamma, delta: Pesos de la heurística

    Returns:
        Tupla con (df_colonias, df_edificaciones) ordenados por prioridad
    """

    # RANKING POR COLONIA
    # Para cada colonia, promediar prioridad de todas las edificaciones
    ranking_colonias = {}
    for colonia in COLONIAS:
        valores = [
            calcular_heuristica(alpha, beta, gamma, delta, edif, colonia)
            for edif in EDIFICACIONES
        ]
        ranking_colonias[colonia] = np.mean(valores)

    df_colonias = pd.DataFrame(
        list(ranking_colonias.items()),
        columns=['Colonia', 'Prioridad']
    )
    df_colonias = df_colonias.sort_values('Prioridad', ascending=False).reset_index(drop=True)
    df_colonias['Ranking'] = range(1, len(df_colonias) + 1)

    # RANKING POR EDIFICACIÓN
    # Para cada edificación, promediar prioridad en todas las colonias
    ranking_edificaciones = {}
    for edificacion in EDIFICACIONES:
        valores = [
            calcular_heuristica(alpha, beta, gamma, delta, edificacion, col)
            for col in COLONIAS
        ]
        ranking_edificaciones[edificacion] = np.mean(valores)

    df_edificaciones = pd.DataFrame(
        list(ranking_edificaciones.items()),
        columns=['Edificación', 'Prioridad']
    )
    df_edificaciones = df_edificaciones.sort_values('Prioridad', ascending=False).reset_index(drop=True)
    df_edificaciones['Ranking'] = range(1, len(df_edificaciones) + 1)

    return df_colonias, df_edificaciones

# ============================================================================
# FUNCIÓN PARA IMPRIMIR RESULTADOS DETALLADOS
# ============================================================================

def imprimir_resultados_detallados(pesos: np.ndarray, resultado: Dict):
    """
    Imprime resultados detallados de la optimización.

    Args:
        pesos: Array con [α, β, γ, δ]
        resultado: Diccionario con utilidad y componentes
    """
    print("\n" + "="*80)
    print("RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*80)

    # PESOS ÓPTIMOS    
    print("\n\nPESOS ÓPTIMOS DE LA HEURÍSTICA")
    nombres_pesos = ['α (Legal)', 'β (Social)', 'γ (Consumo)', 'δ (Reportes)']
    for nombre, peso in zip(nombres_pesos, pesos):
        print(f"   {nombre:15} = {peso:.4f}")
    print(f"   {'Suma':15} = {pesos.sum():.4f}  (debe ser 1.0)")

    # UTILIDAD TOTAL
    print(f"\n\nUTILIDAD TOTAL: {resultado['utilidad_total']:6.2f} / 100")

    # COMPONENTES
    print("\n\nCOMPONENTES DE LA UTILIDAD (5 FACTORES)")
    print(f"   Equidad (35%):           {resultado['equidad']:6.2f}/100")
    print(f"   Coef. Gini:           {resultado['coeficiente_gini']:.4f}")
    print(f"   Social (25%):            {resultado['satisfaccion_social']:6.2f}/100")
    print(f"   Legal (25%):             {resultado['cumplimiento_legal']:6.2f}/100")
    print(f"   Atención Consumo (10%):  {resultado['atencion_consumo']:6.2f}/100")
    print(f"   Atención Reportes (10%): {resultado['atencion_reportes']:6.2f}/100")

    # RANKINGS
    df_colonias, df_edificaciones = calcular_rankings(*pesos)

    print("\n\nDISTRIBUCIÓN DE PRIORIDAD POR COLONIA: ")
    print("   " + " → ".join(df_colonias['Colonia'].tolist()))

    print("\n\nDISTRIBUCIÓN DE PRIORIDAD POR EDIFICACIÓN")
    print("   " + " → ".join(df_edificaciones['Edificación'].tolist()))

    print("\n\nTABLA DETALLADA - COLONIAS")
    print(df_colonias.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print("\n\nTABLA DETALLADA - EDIFICACIONES")
    print(df_edificaciones.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

def imprimir_resultados_detallados(pesos_optimos, resultado, modo_json=False):
    # ... tu código existente que imprime todo ...

    if modo_json:
        return {
            "utilidad_total": resultado.get("utilidad_total", 0),
            "pesos_optimos": {
                "α": round(pesos_optimos[0], 4),
                "β": round(pesos_optimos[1], 4),
                "γ": round(pesos_optimos[2], 4),
                "δ": round(pesos_optimos[3], 4)
            },
            "colonias": resultado.get("colonias", []),
            "edificaciones": resultado.get("edificaciones", [])
        }
