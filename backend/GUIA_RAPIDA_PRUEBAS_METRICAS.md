# Guía Rápida: Ejecutar Pruebas de Métricas de Tiempo

## Resumen
Se implementaron métricas de tiempo de ejecución para evaluar el rendimiento del algoritmo PSO. Las métricas miden tiempos en diferentes fases del proceso de optimización.

## Pruebas Implementadas

### 1. Pruebas Unitarias (10 pruebas)
Ubicación: `backend/tests/unit/test_metricas_tiempo.py`

### 2. Pruebas de Integración
Todas las pruebas existentes fueron actualizadas y siguen funcionando.

## Ejecutar las Pruebas

### Requisitos Previos
```powershell
# Navegar al directorio backend
cd backend

# Asegurarse de que pytest está instalado
pip install pytest pytest-cov
```

### Ejecutar Todas las Pruebas de Métricas
```powershell
# Ejecución básica con salida detallada
python -m pytest tests/unit/test_metricas_tiempo.py -v

# Con cobertura de código
python -m pytest tests/unit/test_metricas_tiempo.py --cov=ia.pso --cov-report=html

# Solo una clase de pruebas específica
python -m pytest tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO -v
```

### Ejecutar Prueba Individual
```powershell
# Ejemplo: prueba específica
python -m pytest tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_pso_retorna_metricas_tiempo -v
```

### Verificar Pruebas Existentes
```powershell
# Pruebas de PSO (verificar compatibilidad)
python -m pytest tests/unit/test_pso.py -v

# Todas las pruebas unitarias
python -m pytest tests/unit/ -v

# Todas las pruebas del proyecto
python -m pytest -v
```

## Demostración Interactiva

### Script de Demostración
Ubicación: `backend/demo_metricas_tiempo.py`

```powershell
# Prueba rápida (10 partículas, 20 iteraciones)
python demo_metricas_tiempo.py rapido

# Configuración estándar (30 partículas, 150 iteraciones)
python demo_metricas_tiempo.py estandar

# Comparar múltiples configuraciones
python demo_metricas_tiempo.py comparar

# Análisis de escalabilidad
python demo_metricas_tiempo.py escalabilidad
```

## Resultados Esperados

### Pruebas Unitarias
```
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_pso_retorna_metricas_tiempo PASSED [ 10%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_metricas_tiempo_son_numericas PASSED [ 20%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_tiempo_total_es_suma_de_componentes PASSED [ 30%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_tiempo_promedio_por_iteracion_es_correcto PASSED [ 40%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_metricas_incluyen_metadatos PASSED [ 50%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_mas_iteraciones_toma_mas_tiempo PASSED [ 60%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_mas_particulas_toma_mas_tiempo PASSED [ 70%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_tiempo_normalizacion_es_constante PASSED [ 80%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoIntegracion::test_consistencia_de_resultados_con_metricas PASSED [ 90%]
tests/unit/test_metricas_tiempo.py::TestMetricasTiempoIntegracion::test_reproducibilidad_con_seed PASSED [100%]

10 passed in ~1.5s
```

### Demo Rápido
```
MÉTRICAS DE TIEMPO:
  Normalización:     29.00µs
  Inicialización:    2.05ms
  Iteraciones:       29.16ms
  Por iteración:     1.46ms
  Total PSO:         31.44ms
  Total medido:      31.47ms

DISTRIBUCIÓN DEL TIEMPO:
  Normalización:    0.09%
  Inicialización:   6.52%
  Iteraciones:     92.74%

RENDIMIENTO:
  Evaluaciones totales:      200
  Evaluaciones por segundo:  6,858
```

## Uso del API con Métricas

### Iniciar el Servidor
```powershell
# Terminal 1: Backend
cd backend
uvicorn main:app --reload

# Terminal 2: Frontend (opcional)
cd frontend
npm run dev
```

### Consultar Métricas vía API

#### Ejecutar Optimización (retorna métricas)
```powershell
# PowerShell
$body = @{
    consumo = @{
        "Lindavista I" = 100000
        "Lindavista II" = 150000
    }
    reportes = @{
        "Lindavista I" = 50
        "Lindavista II" = 30
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/optimize" -Method Post -Body $body -ContentType "application/json"
```

#### Obtener Estadísticas de Rendimiento
```powershell
# PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/api/optimize/stats/performance"
```

## Estructura de Métricas Retornadas

```json
{
  "metricas_tiempo": {
    "tiempo_total": 2.1534,
    "tiempo_normalizacion": 0.0002,
    "tiempo_inicializacion": 0.1234,
    "tiempo_iteraciones": 2.0298,
    "tiempo_promedio_por_iteracion": 0.0135,
    "iteraciones_totales": 150,
    "particulas_totales": 30,
    "tiempo_validacion": 0.0001,
    "tiempo_guardado_entrada": 0.0523,
    "tiempo_procesamiento_resultados": 0.0234,
    "tiempo_guardado_bd": 0.0523,
    "tiempo_procesamiento_api": 2.3456
  }
}
```

## Troubleshooting

### Error: "pytest no reconocido"
```powershell
# Usar python -m pytest en su lugar
python -m pytest tests/unit/test_metricas_tiempo.py -v
```

### Error: "ModuleNotFoundError"
```powershell
# Verificar entorno virtual
cd backend
python -c "import sys; print(sys.prefix)"

# Reinstalar dependencias
pip install -r requirements.txt
```

### Pruebas lentas
```powershell
# Ejecutar solo pruebas rápidas
python -m pytest tests/unit/test_metricas_tiempo.py::TestMetricasTiempoPSO::test_pso_retorna_metricas_tiempo -v
```

## Documentación Adicional

- **Guía Completa**: `backend/GUIA_METRICAS_TIEMPO.md`
- **Resumen de Implementación**: `backend/RESUMEN_IMPLEMENTACION_METRICAS.md`
- **Guía General de Pruebas**: `backend/tests/GUIA_PRUEBAS.md`

## Verificación Rápida

Ejecutar esto para verificar que todo funciona:
```powershell
cd backend
python -m pytest tests/unit/test_metricas_tiempo.py -v && python demo_metricas_tiempo.py rapido
```

Si ambos comandos se ejecutan sin errores, la implementación está funcionando correctamente.
