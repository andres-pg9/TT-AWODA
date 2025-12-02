import React, { useState, useEffect } from 'react';
import { ResponsiveLine } from '@nivo/line';
import './MapaColonias.css'; // Reutilizamos los estilos existentes
import { API_URL } from '../config';

// 🔧 URL del backend (cambiar si usas otro puerto)
//const API_URL = 'http://localhost:8000';

/**
 * Componente Graficas
 * 
 * Muestra gráficas de historial de consumo y reportes por colonia
 * usando Nivo Line Charts con el mismo diseño del dashboard principal
 */
const Graficas = () => {
    // Estados
    const [colonias, setColonias] = useState([]);
    const [coloniaSeleccionada, setColoniaSeleccionada] = useState(null);
    const [limite, setLimite] = useState(10);
    const [datosHistorial, setDatosHistorial] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Cargar lista de colonias al montar
    useEffect(() => {
        cargarColonias();
    }, []);

    // Cargar historial cuando cambia la colonia o el límite
    useEffect(() => {
        if (coloniaSeleccionada) {
            cargarHistorial(coloniaSeleccionada, limite);
        }
    }, [coloniaSeleccionada, limite]);

    /**
     * Obtener lista de colonias disponibles
     */
    const cargarColonias = async () => {
        try {
            const response = await fetch(`${API_URL}/api/colonias/`);
            if (!response.ok) throw new Error('Error al cargar colonias');

            const data = await response.json();
            setColonias(data.colonias);

            // Seleccionar primera colonia por defecto
            if (data.colonias.length > 0) {
                setColoniaSeleccionada(data.colonias[0]);
            }

            setLoading(false);
        } catch (err) {
            console.error('Error al cargar colonias:', err);
            setError('No se pudieron cargar las colonias');
            setLoading(false);
        }
    };

    /**
     * Obtener historial de una colonia específica
     */
    const cargarHistorial = async (nombreColonia, limit) => {
        try {
            setLoading(true);
            setError(null);

            const response = await fetch(
                `${API_URL}/api/colonias/${encodeURIComponent(nombreColonia)}/historial?limit=${limit}`
            );

            if (!response.ok) {
                throw new Error('Error al cargar historial');
            }

            const data = await response.json();
            setDatosHistorial(data);
            setLoading(false);
        } catch (err) {
            console.error('Error al cargar historial:', err);
            setError('No se pudo cargar el historial de la colonia');
            setLoading(false);
        }
    };

    /**
     * Formatear datos para Nivo Line Chart
     */
    const formatearDatosParaNivo = (tipo) => {
        if (!datosHistorial || !datosHistorial.formato_nivo) return [];

        const datos = tipo === 'consumo'
            ? datosHistorial.formato_nivo.consumo
            : datosHistorial.formato_nivo.reportes;

        return [{
            id: tipo === 'consumo' ? 'Consumo (m³)' : 'Número de Reportes',
            color: tipo === 'consumo' ? '#1976d2' : '#e74c3c',
            data: datos.map(item => ({
                x: new Date(item.x).toLocaleDateString("es-MX", {
                    timeZone: "America/Mexico_City"
                }) + "\n" +
                    new Date(item.x).toLocaleTimeString("es-MX", {
                        timeZone: "America/Mexico_City",
                        hour: "2-digit",
                        minute: "2-digit"
                    })
                ,
                y: item.y
            }))
        }];
    };


    /**
     * Configuración común para ambas gráficas
     */
    const configuracionGraficaBase = {
        margin: { top: 30, right: 30, bottom: 70, left: 80 },
        xScale: { type: 'point' },
        yScale: {
            type: 'linear',
            min: 'auto',
            max: 'auto',
            stacked: false,
            reverse: false
        },
        curve: 'linear',
        axisTop: null,
        axisRight: null,
        axisBottom: {
            tickSize: 5,
            tickPadding: 10,
            tickRotation: 0,
            format: value => value,
            legend: 'Fecha',
            legendOffset: 50,
            legendPosition: 'middle'
        },
        axisLeft: {
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legendOffset: -60,
            legendPosition: 'middle'
        },
        pointSize: 8,
        pointColor: { theme: 'background' },
        pointBorderWidth: 2,
        pointBorderColor: { from: 'serieColor' },
        pointLabelYOffset: -12,
        useMesh: true,
        enableSlices: 'x',
        enableGridX: false,
        enableGridY: true,
        lineWidth: 3,
        enableArea: false,
        areaOpacity: 0.1
    };

    // Pantalla de carga inicial
    if (loading && colonias.length === 0) {
        return (
            <div className="awoda-loading">
                <div className="awoda-spinner"></div>
                <p>Cargando datos de colonias...</p>
            </div>
        );
    }

    return (
        <div className="awoda-container">
            {/* ========================================
                SIDEBAR IZQUIERDO - Lista de Colonias
                ======================================== */}
            <aside className="awoda-sidebar">
                <div className="awoda-sidebar-header">
                    <h3>Colonias disponibles:</h3>
                </div>

                <div className="awoda-colonias-lista">
                    {colonias.map((colonia, index) => (
                        <button
                            key={index}
                            className={`awoda-colonia-item ${coloniaSeleccionada === colonia ? 'active' : ''
                                }`}
                            onClick={() => setColoniaSeleccionada(colonia)}
                        >
                            <span className="awoda-colonia-icono">→</span>
                            <span className="awoda-colonia-nombre">{colonia}</span>
                        </button>
                    ))}
                </div>
            </aside>

            {/* ========================================
                ÁREA PRINCIPAL - Gráficas
                ======================================== */}
            <main className="awoda-main">
                <div className="awoda-map-header">
                    <h2>
                        Gráficas - {coloniaSeleccionada || 'Seleccione una colonia'}
                    </h2>
                </div>

                <div style={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '20px',
                    padding: '20px',
                    backgroundColor: '#f5f5f5',
                    overflowY: 'auto'
                }}>
                    {error && (
                        <div style={{
                            padding: '15px',
                            backgroundColor: '#ffebee',
                            border: '1px solid #ef5350',
                            borderRadius: '6px',
                            color: '#c62828'
                        }}>
                            ⚠️ {error}
                        </div>
                    )}

                    {loading && coloniaSeleccionada && (
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '40px',
                            gap: '15px'
                        }}>
                            <div className="awoda-spinner" style={{ width: '40px', height: '40px' }}></div>
                            <p style={{ color: '#666' }}>Cargando datos de {coloniaSeleccionada}...</p>
                        </div>
                    )}

                    {!loading && datosHistorial && (
                        <>
                            {/* GRÁFICA 1: CONSUMO */}
                            <div style={{
                                backgroundColor: 'white',
                                borderRadius: '8px',
                                padding: '20px',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                                height: '350px'
                            }}>
                                <h3 style={{
                                    margin: '0 0 15px 0',
                                    fontSize: '18px',
                                    fontWeight: '600',
                                    color: '#2c3e50',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '10px'
                                }}>
                                    Consumo de Agua (m³)
                                </h3>
                                <div style={{ height: 'calc(100% - 40px)' }}>
                                    <ResponsiveLine
                                        data={formatearDatosParaNivo('consumo')}
                                        {...configuracionGraficaBase}
                                        axisLeft={{
                                            ...configuracionGraficaBase.axisLeft,
                                            legend: 'Consumo (m³)',
                                        }}
                                        colors={['#1976d2']}
                                        theme={{
                                            axis: {
                                                ticks: {
                                                    text: { fontSize: 11, fill: '#666' }
                                                },
                                                legend: {
                                                    text: { fontSize: 12, fontWeight: 600, fill: '#333' }
                                                }
                                            },
                                            grid: {
                                                line: { stroke: '#e0e0e0', strokeWidth: 1 }
                                            }
                                        }}
                                        sliceTooltip={({ slice }) => (
                                            <div style={{
                                                background: 'white',
                                                padding: '12px 16px',
                                                border: '2px solid #1976d2',
                                                borderRadius: '8px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                                                minWidth: '180px'
                                            }}>
                                                <div style={{ 
                                                    fontSize: '13px',
                                                    color: '#666',
                                                    marginBottom: '6px'
                                                }}>
                                                    {slice.points[0].data.xFormatted}
                                                </div>
                                                <div style={{ 
                                                    fontSize: '16px',
                                                    fontWeight: '700',
                                                    color: '#1976d2',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '6px'
                                                }}>
                                                    {slice.points[0].data.yFormatted} m³
                                                </div>
                                            </div>
                                        )}
                                    />
                                </div>
                            </div>

                            {/* GRÁFICA 2: REPORTES */}
                            <div style={{
                                backgroundColor: 'white',
                                borderRadius: '8px',
                                padding: '20px',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                                height: '350px'
                            }}>
                                <h3 style={{
                                    margin: '0 0 15px 0',
                                    fontSize: '18px',
                                    fontWeight: '600',
                                    color: '#2c3e50',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '10px'
                                }}>
                                    Número de Reportes
                                </h3>
                                <div style={{ height: 'calc(100% - 40px)' }}>
                                    <ResponsiveLine
                                        data={formatearDatosParaNivo('reportes')}
                                        {...configuracionGraficaBase}
                                        axisLeft={{
                                            ...configuracionGraficaBase.axisLeft,
                                            legend: 'Número de Reportes',
                                        }}
                                        colors={['#e74c3c']}
                                        theme={{
                                            axis: {
                                                ticks: {
                                                    text: { fontSize: 11, fill: '#666' }
                                                },
                                                legend: {
                                                    text: { fontSize: 12, fontWeight: 600, fill: '#333' }
                                                }
                                            },
                                            grid: {
                                                line: { stroke: '#e0e0e0', strokeWidth: 1 }
                                            }
                                        }}
                                        sliceTooltip={({ slice }) => (
                                            <div style={{
                                                background: 'white',
                                                padding: '12px 16px',
                                                border: '2px solid #e74c3c',
                                                borderRadius: '8px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                                                minWidth: '180px'
                                            }}>
                                                <div style={{ 
                                                    fontSize: '13px',
                                                    color: '#666',
                                                    marginBottom: '6px'
                                                }}>
                                                    {slice.points[0].data.xFormatted}
                                                </div>
                                                <div style={{ 
                                                    fontSize: '16px',
                                                    fontWeight: '700',
                                                    color: '#e74c3c',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '6px'
                                                }}>
                                                    {slice.points[0].data.yFormatted} reportes
                                                </div>
                                            </div>
                                        )}
                                    />
                                </div>
                            </div>
                        </>
                    )}

                    {!loading && !datosHistorial && !error && (
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '60px',
                            color: '#999',
                            gap: '10px'
                        }}>
                            <span style={{ fontSize: '48px' }}>📊</span>
                            <p style={{ fontSize: '16px', margin: 0 }}>
                                Selecciona una colonia para ver sus gráficas
                            </p>
                        </div>
                    )}
                </div>
            </main>

            {/* ========================================
                SIDEBAR DERECHO - Controles y Detalles
                ======================================== */}
            <aside className="awoda-sidebar-right">
                <div className="awoda-distribucion-header">
                    <h2>Configuración</h2>
                </div>

                {/* Mensaje informativo */}
                <div className="awoda-disclaimer-footer">
                    <p>
                        <strong>Información:</strong> Las gráficas muestran el historial
                        de consumo de agua y número de reportes de incidencias para la colonia
                        seleccionada. Los datos se actualizan automáticamente al cambiar de colonia
                        o modificar el número de registros a mostrar.
                    </p>
                </div>

                {/* Selector de límite */}
                <div style={{ padding: '20px 25px' }}>
                    <label style={{
                        display: 'block',
                        marginBottom: '10px',
                        fontSize: '14px',
                        fontWeight: '600',
                        color: '#2c3e50'
                    }}>
                        Número de datos a mostrar:
                    </label>
                    <select
                        value={limite}
                        onChange={(e) => setLimite(Number(e.target.value))}
                        style={{
                            width: '100%',
                            padding: '12px',
                            fontSize: '14px',
                            border: '2px solid #e0e0e0',
                            borderRadius: '6px',
                            backgroundColor: 'white',
                            color: '#000',
                            cursor: 'pointer',
                            transition: 'border-color 0.2s',
                            outline: 'none'
                        }}
                        onFocus={(e) => e.target.style.borderColor = '#1976d2'}
                        onBlur={(e) => e.target.style.borderColor = '#e0e0e0'}
                    >
                        <option value={5}>5 registros</option>
                        <option value={7}>7 registros</option>
                        <option value={10}>10 registros</option>
                    </select>
                </div>

                {/* Información de la colonia seleccionada */}
                {datosHistorial && (
                    <div className="awoda-colonia-detalles">
                        <h3>{coloniaSeleccionada}</h3>

                        <div className="awoda-detalle-item">
                            <span>Total de registros:</span>
                            <strong>{datosHistorial.total_registros}</strong>
                        </div>

                        <div className="awoda-detalle-item">
                            <span>Mostrando:</span>
                            <strong>{datosHistorial.limite_aplicado} registros</strong>
                        </div>

                        {datosHistorial.datos && datosHistorial.datos.length > 0 && (
                            <>
                                <div style={{
                                    marginTop: '20px',
                                    paddingTop: '15px',
                                    borderTop: '1px solid #ddd'
                                }}>
                                    <p style={{
                                        fontSize: '13px',
                                        fontWeight: '600',
                                        color: '#2c3e50',
                                        marginBottom: '10px'
                                    }}>
                                        Datos más recientes:
                                    </p>

                                    <div className="awoda-detalle-item">
                                        <span>Consumo:</span>
                                        <strong>{datosHistorial.datos[0].consumo.toFixed(2)} m³</strong>
                                    </div>

                                    <div className="awoda-detalle-item">
                                        <span>Reportes:</span>
                                        <strong>{datosHistorial.datos[0].reportes}</strong>
                                    </div>

                                    <div className="awoda-detalle-item" style={{ fontSize: '12px', color: '#666' }}>
                                        <span>Fecha:</span>
                                        <span>{new Date(datosHistorial.datos[0].fecha).toLocaleString('es-MX')}</span>
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                )}


            </aside>
        </div>
    );
};

export default Graficas;