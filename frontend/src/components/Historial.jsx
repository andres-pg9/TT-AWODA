import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import './Historial.css';
import { API_URL } from '../config';

/**
 * COMPONENTE HISTORIAL
 * Muestra el historial de sugerencias de optimización calculadas previamente
 */
const Historial = () => {
  const navigate = useNavigate();
  const [historial, setHistorial] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [detalleResultado, setDetalleResultado] = useState(null);
  const [loadingDetalle, setLoadingDetalle] = useState(false);
  const [usuario, setUsuario] = useState(null);

  // Cargar datos del usuario
  useEffect(() => {
    const userData = localStorage.getItem('usuario');
    if (userData) {
      setUsuario(JSON.parse(userData));
    }
  }, []);

  // Cargar historial al montar el componente
  useEffect(() => {
    cargarHistorial();
  }, []);

  const cargarHistorial = async () => {
    try {
      setLoading(true);
      setError(null);

      const token = localStorage.getItem('token');
      const response = await fetch(`${API_URL}/api/optimize/historial?limit=20`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Error al cargar el historial');
      }

      const data = await response.json();
      //console.log('Datos recibidos del historial:', data);
      //console.log('Primer resultado:', data.resultados?.[0]);
      setHistorial(data.resultados || []);
    } catch (err) {
      setError(err.message);
      console.error('Error al cargar historial:', err);
    } finally {
      setLoading(false);
    }
  };

  const cargarDetalle = async (resultadoId) => {
    try {
      setLoadingDetalle(true);
      
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_URL}/api/optimize/${resultadoId}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Error al cargar detalles');
      }

      const data = await response.json();
      setDetalleResultado(data);
    } catch (err) {
      console.error('Error al cargar detalles:', err);
      alert('Error al cargar los detalles del resultado');
    } finally {
      setLoadingDetalle(false);
    }
  };

  const handleToggleDetalle = async (resultadoId) => {
    if (expandedId === resultadoId) {
      // Cerrar detalle
      setExpandedId(null);
      setDetalleResultado(null);
    } else {
      // Abrir detalle
      setExpandedId(resultadoId);
      await cargarDetalle(resultadoId);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const formatearFecha = (fechaISO) => {
    const fecha = new Date(fechaISO);
    // Restar 6 horas para convertir UTC a hora local CDMX
    fecha.setHours(fecha.getHours() - 6);
    return fecha.toLocaleString('es-MX', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatearNumero = (numero) => {
    if (numero === undefined || numero === null || isNaN(numero)) {
      return '0.00';
    }
    return Number(numero).toLocaleString('es-MX', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const formatearPorcentaje = (numero) => {
    if (numero === undefined || numero === null || isNaN(numero)) {
      return '0%';
    }
    return `${(Number(numero) * 100).toFixed(0)}%`;
  };

  const exportarAPDF = async () => {
    try {
      const doc = new jsPDF();
      
      // Título
      doc.setFontSize(18);
      doc.setTextColor(26, 35, 50);
      doc.text('AWODA - Historial de Optimización', 14, 20);
      
      // Información de exportación
      doc.setFontSize(10);
      doc.setTextColor(100, 100, 100);
      const fechaExportacion = new Date().toLocaleString('es-MX', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
      doc.text(`Fecha de exportación: ${fechaExportacion}`, 14, 28);
      if (usuario) {
        doc.text(`Generado por: ${usuario.nombre_empleado}`, 14, 34);
      }
      
      // Obtener últimos 10 resultados con sus rankings
      const ultimos10 = historial.slice(0, 10);
      const token = localStorage.getItem('token');
      let yPos = 40;
      
      for (let i = 0; i < ultimos10.length; i++) {
        const resultado = ultimos10[i];
        
        // Nueva página para cada resultado (excepto el primero)
        if (i > 0) {
          doc.addPage();
          yPos = 20;
        }
        
        // Encabezado del resultado
        doc.setFontSize(14);
        doc.setTextColor(26, 35, 50);
        doc.text(`Resultado ${i + 1}`, 14, yPos);
        yPos += 5;
        
        // Información del resultado (Fecha, Utilidad, Pesos)
        const infoData = [
          ['Fecha de Cálculo', formatearFecha(resultado.fecha_calculo)],
          ['Utilidad Total', formatearNumero(resultado.utilidad_total)],
          ['Peso Social', formatearPorcentaje(resultado.pesos_heuristica?.beta_social)],
          ['Peso Legal', formatearPorcentaje(resultado.pesos_heuristica?.alfa_legal)],
          ['Peso Consumo', formatearPorcentaje(resultado.pesos_heuristica?.gamma_consumo)],
          ['Peso Reportes', formatearPorcentaje(resultado.pesos_heuristica?.delta_reportes)]
        ];
        
        autoTable(doc, {
          startY: yPos,
          body: infoData,
          theme: 'plain',
          styles: { 
            fontSize: 10, 
            cellPadding: 2,
            lineColor: [200, 200, 200],
            lineWidth: 0.1
          },
          columnStyles: {
            0: { cellWidth: 50, fontStyle: 'bold', textColor: [52, 73, 94] },
            1: { cellWidth: 130 }
          }
        });
        
        yPos = doc.lastAutoTable.finalY + 10;
        
        // Cargar detalles del resultado
        const response = await fetch(`${API_URL}/api/optimize/${resultado.id}`, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });
        
        if (!response.ok) continue;
        
        const detalle = await response.json();
        
        // Ranking de Colonias
        if (detalle.ranking_colonias && detalle.ranking_colonias.length > 0) {
          doc.setFontSize(12);
          doc.setTextColor(52, 73, 94);
          doc.text('Orden de prioridad en Colonias', 14, yPos);
          yPos += 6;
          
          const coloniasData = detalle.ranking_colonias.slice(0, 10).map((col, idx) => [
            idx + 1,
            col.nombre || 'N/A'
          ]);
          
          autoTable(doc, {
            startY: yPos,
            head: [['#', 'Colonia']],
            body: coloniasData,
            theme: 'striped',
            headStyles: { fillColor: [52, 73, 94], fontSize: 10 },
            styles: { fontSize: 9, cellPadding: 3 },
            columnStyles: {
              0: { cellWidth: 15, halign: 'center' },
              1: { cellWidth: 165 }
            }
          });
          
          yPos = doc.lastAutoTable.finalY + 10;
        }
        
        // Ranking de Edificaciones
        if (detalle.ranking_edificaciones && detalle.ranking_edificaciones.length > 0) {
          // Verificar si hay espacio suficiente, si no, nueva página
          if (yPos > 220) {
            doc.addPage();
            yPos = 20;
          }
          
          doc.setFontSize(12);
          doc.setTextColor(52, 73, 94);
          doc.text('Orden de prioridad en Edificaciones', 14, yPos);
          yPos += 6;
          
          const edificacionesData = detalle.ranking_edificaciones.slice(0, 10).map((edif, idx) => [
            idx + 1,
            edif.nombre || 'N/A'
          ]);
          
          autoTable(doc, {
            startY: yPos,
            head: [['#', 'Edificación']],
            body: edificacionesData,
            theme: 'striped',
            headStyles: { fillColor: [52, 73, 94], fontSize: 10 },
            styles: { fontSize: 9, cellPadding: 3 },
            columnStyles: {
              0: { cellWidth: 15, halign: 'center' },
              1: { cellWidth: 165 }
            }
          });
        }
      }
      
      // Guardar PDF con fecha local (CDMX, UTC-6)
      const fechaLocal = new Date();
      fechaLocal.setHours(fechaLocal.getHours() - 6);
      const fechaFormato = fechaLocal.toISOString().split('T')[0];
      doc.save(`AWODA_Historial_${fechaFormato}.pdf`);
      
    } catch (error) {
      console.error('Error al generar PDF:', error);
      alert('Error al generar el PDF. Por favor intenta de nuevo.');
    }
  };

  if (loading) {
    return (
      <div className="historial-container">
        <nav className="awoda-navbar">
          <div className="awoda-navbar-brand">
            <img src="/isotipo-awoda.png" alt="AWODA" className="awoda-icon" />
            <span className="awoda-title">AWODA</span>
          </div>
          <div className="awoda-navbar-menu">
            <button onClick={() => navigate('/dashboard')} className="awoda-nav-link">
              PRINCIPAL
            </button>
            <a href="#graficas" className="awoda-nav-link">GRÁFICAS</a>
            <span className="awoda-nav-link active">HISTORIAL</span>
            <a href="#entrenamiento" className="awoda-nav-link">ENTRENAMIENTO</a>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            {usuario && (
              <span style={{ color: 'white', fontSize: '14px' }}>
                👤 {usuario.nombre_empleado}
              </span>
            )}
            <button className="awoda-navbar-logout" onClick={handleLogout}>
              Cerrar sesión
            </button>
          </div>
        </nav>
        <div className="historial-content">
          <div className="historial-loading">
            <div className="spinner"></div>
            <p>Cargando historial...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="historial-container">

      {/* Contenido principal */}
      <div className="historial-content">
        <div className="historial-header">
          <h1>Historial de Sugerencias de Optimización</h1>
          <p>Consulta las distribuciones de agua previas</p>
        </div>

        {error && (
          <div className="historial-error">
            <p>⚠️ {error}</p>
            <button onClick={cargarHistorial}>Reintentar</button>
          </div>
        )}

        {!error && historial.length === 0 && (
          <div className="historial-empty">
            <p>No hay resultados de optimización guardados</p>
            <button onClick={() => navigate('/dashboard')}>
              Ir al Mapa para Calcular
            </button>
          </div>
        )}

        {!error && historial.length > 0 && (
          <>
            <div className="historial-actions">
              <button className="btn-exportar-pdf" onClick={exportarAPDF}>
                📄 Exportar a PDF
              </button>
            </div>
            <div className="historial-table-container">
              <table className="historial-table">
              <thead>
                <tr>
                  <th>Fecha de Cálculo</th>
                  <th>Utilidad Total</th>
                  <th>Pesos de Heurística</th>
                  <th>Acciones</th>
                </tr>
              </thead>
              <tbody>
                {historial.map((resultado) => (
                  <React.Fragment key={resultado.id}>
                    <tr className="historial-row">
                      <td>{formatearFecha(resultado.fecha_calculo)}</td>
                      <td className="utilidad-cell">
                        {formatearNumero(resultado.utilidad_total)}
                      </td>
                      <td className="pesos-cell">
                        <div className="pesos-container">
                          <span className="peso-badge">
                            Social: {formatearPorcentaje(resultado.pesos_heuristica?.beta_social)}
                          </span>
                          <span className="peso-badge">
                            Legal: {formatearPorcentaje(resultado.pesos_heuristica?.alfa_legal)}
                          </span>
                          <span className="peso-badge">
                            Consumo: {formatearPorcentaje(resultado.pesos_heuristica?.gamma_consumo)}
                          </span>
                          <span className="peso-badge">
                            Reportes: {formatearPorcentaje(resultado.pesos_heuristica?.delta_reportes)}
                          </span>
                        </div>
                      </td>
                      <td className="actions-cell">
                        <button 
                          className="btn-ver-detalle"
                          onClick={() => handleToggleDetalle(resultado.id)}
                        >
                          {expandedId === resultado.id ? '▲ Ocultar' : '▼ Ver Detalles'}
                        </button>
                      </td>
                    </tr>
                    
                    {/* Fila expandible con detalles */}
                    {expandedId === resultado.id && (
                      <tr className="detalle-row">
                        <td colSpan="4">
                          <div className="detalle-container">
                            {loadingDetalle ? (
                              <div className="detalle-loading">
                                <div className="spinner-small"></div>
                                <p>Cargando detalles...</p>
                              </div>
                            ) : detalleResultado ? (
                              <div className="detalle-content">
                                {/* Rankings */}
                                <div className="detalle-rankings">
                                  {/* Ranking de Colonias */}
                                  {detalleResultado.ranking_colonias && detalleResultado.ranking_colonias.length > 0 && (
                                    <div className="detalle-section">
                                      <h3>Ranking de Colonias</h3>
                                      <div className="ranking-table-wrapper">
                                        <table className="ranking-table">
                                          <thead>
                                            <tr>
                                              <th>#</th>
                                              <th>Colonia</th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {detalleResultado.ranking_colonias.slice(0, 10).map((colonia, index) => (
                                              <tr key={index}>
                                                <td className="rank-cell">{index + 1}</td>
                                                <td>{colonia.nombre || 'N/A'}</td>
                                              </tr>
                                            ))}
                                          </tbody>
                                        </table>
                                        {detalleResultado.ranking_colonias.length > 10 && (
                                          <p className="ranking-note">
                                            Mostrando 10 de {detalleResultado.ranking_colonias.length} colonias
                                          </p>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Ranking de Edificaciones */}
                                  {detalleResultado.ranking_edificaciones && detalleResultado.ranking_edificaciones.length > 0 && (
                                    <div className="detalle-section">
                                      <h3>Ranking de Edificaciones</h3>
                                      <div className="ranking-table-wrapper">
                                        <table className="ranking-table">
                                          <thead>
                                            <tr>
                                              <th>#</th>
                                              <th>Edificación</th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {detalleResultado.ranking_edificaciones.slice(0, 10).map((edificacion, index) => (
                                              <tr key={index}>
                                                <td className="rank-cell">{index + 1}</td>
                                                <td>{edificacion.nombre || 'N/A'}</td>
                                              </tr>
                                            ))}
                                          </tbody>
                                        </table>
                                        {detalleResultado.ranking_edificaciones.length > 10 && (
                                          <p className="ranking-note">
                                            Mostrando 10 de {detalleResultado.ranking_edificaciones.length} edificaciones
                                          </p>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            ) : (
                              <p>No se pudieron cargar los detalles</p>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Historial;
