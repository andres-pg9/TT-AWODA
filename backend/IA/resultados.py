import numpy as np
import pandas as pd
from typing import Tuple, Dict
from .normalizacion import COLONIAS, EDIFICACIONES
from .funciones import calcular_heuristica

# ============================================================================ #
# FUNCIONES DE RANKING
# ============================================================================ #

def calcular_rankings(alpha: float, beta: float, gamma: float, delta: float,
                     consumo_norm: Dict[str, float] = None,
                     reportes_norm: Dict[str, float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula rankings de prioridad para colonias y edificaciones.
    
    Args:
        alpha, beta, gamma, delta: Pesos de la heurística
        consumo_norm: Diccionario opcional con valores de consumo normalizados
        reportes_norm: Diccionario opcional con valores de reportes normalizados
    
    Returns:
        Tupla con DataFrames de colonias y edificaciones ordenados por prioridad
    """
    # RANKING POR COLONIA
    ranking_colonias = {}
    for colonia in COLONIAS:
        valores = [
            calcular_heuristica(alpha, beta, gamma, delta, edif, colonia,
                              consumo_norm, reportes_norm)
            for edif in EDIFICACIONES
        ]
        ranking_colonias[colonia] = np.mean(valores)

    df_colonias = pd.DataFrame(
        list(ranking_colonias.items()), columns=['Colonia', 'Prioridad']
    ).sort_values('Prioridad', ascending=False).reset_index(drop=True)
    df_colonias['Ranking'] = range(1, len(df_colonias) + 1)

    # RANKING POR EDIFICACIÓN
    ranking_edificaciones = {}
    for edificacion in EDIFICACIONES:
        valores = [
            calcular_heuristica(alpha, beta, gamma, delta, edificacion, col,
                              consumo_norm, reportes_norm)
            for col in COLONIAS
        ]
        ranking_edificaciones[edificacion] = np.mean(valores)

    df_edificaciones = pd.DataFrame(
        list(ranking_edificaciones.items()), columns=['Edificación', 'Prioridad']
    ).sort_values('Prioridad', ascending=False).reset_index(drop=True)
    df_edificaciones['Ranking'] = range(1, len(df_edificaciones) + 1)

    return df_colonias, df_edificaciones


# ============================================================================ #
# FUNCIÓN PARA IMPRIMIR RESULTADOS DETALLADOS
# ============================================================================ #

def imprimir_resultados_detallados(pesos_optimos, resultado, modo_json=False,
                                  consumo_norm: Dict[str, float] = None,
                                  reportes_norm: Dict[str, float] = None):
    """
    Imprime o devuelve resultados detallados de la optimización.
    
    Args:
        pesos_optimos: Array con los pesos óptimos encontrados
        resultado: Diccionario con los resultados de la utilidad
        modo_json: Si True, retorna un diccionario en lugar de imprimir
        consumo_norm: Diccionario opcional con valores de consumo normalizados
        reportes_norm: Diccionario opcional con valores de reportes normalizados
    """
    print("\n" + "="*80)
    print("RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*80)

    # PESOS ÓPTIMOS    
    print("\nPESOS ÓPTIMOS DE LA HEURÍSTICA")
    nombres_pesos = ['α (Legal)', 'β (Social)', 'γ (Consumo)', 'δ (Reportes)']
    for nombre, peso in zip(nombres_pesos, pesos_optimos):
        print(f"   {nombre:15} = {peso:.4f}")
    print(f"   {'Suma':15} = {pesos_optimos.sum():.4f}  (debe ser 1.0)")

    # UTILIDAD TOTAL
    print(f"\nUTILIDAD TOTAL: {resultado['utilidad_total']:6.2f} / 100")

    # COMPONENTES
    print("\nCOMPONENTES DE LA UTILIDAD (5 FACTORES)")
    print(f"   Equidad (35%):           {resultado['equidad']:6.2f}/100")
    print(f"   Coef. Gini:              {resultado['coeficiente_gini']:.4f}")
    print(f"   Social (25%):            {resultado['satisfaccion_social']:6.2f}/100")
    print(f"   Legal (25%):             {resultado['cumplimiento_legal']:6.2f}/100")
    print(f"   Atención Consumo (10%):  {resultado['atencion_consumo']:6.2f}/100")
    print(f"   Atención Reportes (10%): {resultado['atencion_reportes']:6.2f}/100")

    # RANKINGS usando los valores normalizados
    df_colonias, df_edificaciones = calcular_rankings(*pesos_optimos,
                                                      consumo_norm=consumo_norm,
                                                      reportes_norm=reportes_norm)

    print("\nDISTRIBUCIÓN DE PRIORIDAD POR COLONIA: ")
    print("   " + " → ".join(df_colonias['Colonia'].tolist()))

    print("\nDISTRIBUCIÓN DE PRIORIDAD POR EDIFICACIÓN")
    print("   " + " → ".join(df_edificaciones['Edificación'].tolist()))

    print("\nTABLA DETALLADA - COLONIAS")
    print(df_colonias.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print("\nTABLA DETALLADA - EDIFICACIONES")
    print(df_edificaciones.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # ======================================================
    # MODO JSON (para FastAPI)
    # ======================================================
    if modo_json:
        colonias_data = [
            {"nombre": row["Colonia"], "prioridad": row["Prioridad"], "ranking": int(row["Ranking"])}
            for _, row in df_colonias.iterrows()
        ]
        edificaciones_data = [
            {"nombre": row["Edificación"], "prioridad": row["Prioridad"], "ranking": int(row["Ranking"])}
            for _, row in df_edificaciones.iterrows()
        ]

        return {
            "utilidad_total": resultado.get("utilidad_total", 0),
            "pesos_optimos": {
                "α": round(pesos_optimos[0], 4),
                "β": round(pesos_optimos[1], 4),
                "γ": round(pesos_optimos[2], 4),
                "δ": round(pesos_optimos[3], 4)
            },
            "colonias": colonias_data,
            "edificaciones": edificaciones_data
        }
