import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './MapaColonias.css';
import L from 'leaflet';
import ColoniaEdificaciones from "./ColoniaEdificaciones";
import { API_URL } from '../config';


const COLORES_RANKING = {
  1: '#FF0000',
  2: '#FF6600',
  3: '#FFAA00',
  4: '#FFFF00',
  5: '#FFFF99',
  6: '#99FF99',
  7: '#00CC00',
};

const ForceMapResize = () => {
  const map = useMap();

  useEffect(() => {
    setTimeout(() => {
      map.invalidateSize();
    }, 200);
  }, [map]);

  return null;
};

const FitBounds = ({ bounds }) => {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, {
        padding: [50, 50],
        maxZoom: 15,
        animate: true,
        duration: 0.5
      });
    }
  }, [bounds, map]);

  return null;
};

const LeyendaColores = () => {
  const map = useMap();

  useEffect(() => {
    const legend = L.control({ position: 'bottomright' });

    legend.onAdd = () => {
      const div = L.DomUtil.create('div', 'info legend');
      const rangos = [
        '1 prioridad máxima',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7 prioridad mínima'
      ];
      const coloresRango = [
        '#FF0000',
        '#FF6600',
        '#FFAA00',
        '#FFFF00',
        '#FFFF99',
        '#99FF99',
        '#00CC00',
      ];

      let labels = '<h4 style="margin:4px 0;">Orden de prioridad</h4>';
      for (let i = 0; i < rangos.length; i++) {
        labels +=
          `<i style="background:${coloresRango[i]}; width:18px; height:18px; display:inline-block; margin-right:6px; border:1px solid #999;"></i>` +
          rangos[i] + '<br>';
      }

      div.innerHTML = labels;
      return div;
    };

    legend.addTo(map);
    return () => {
      legend.remove();
    };
  }, [map]);

  return null;
};

const ModalParametros = ({ isOpen, onClose, colonias, onGuardar }) => {
  const [parametros, setParametros] = useState({
    consumo: {},
    reportes: {}
  });
  const [cargando, setCargando] = useState(false);

  useEffect(() => {
    if (isOpen && colonias.length > 0) {
      const consumoInicial = {};
      const reportesInicial = {};

      colonias.forEach((colonia) => {
        consumoInicial[colonia.colonia] = 0;
        reportesInicial[colonia.colonia] = 0;
      });

      setParametros({
        consumo: consumoInicial,
        reportes: reportesInicial
      });
    }
  }, [isOpen, colonias]);

  const handleConsumoChange = (colonia, valor) => {
    setParametros(prev => ({
      ...prev,
      consumo: {
        ...prev.consumo,
        [colonia]: valor === '' ? '' : Number(valor)
      }
    }));
  };

  const handleReportesChange = (colonia, valor) => {
    setParametros(prev => ({
      ...prev,
      reportes: {
        ...prev.reportes,
        [colonia]: valor === '' ? '' : Number(valor)
      }
    }));
  };

  const handleGuardar = async () => {
    setCargando(true);
    try {
      await onGuardar(parametros);
      onClose();
    } catch (error) {
      console.error('Error al guardar parámetros:', error);
    } finally {
      setCargando(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>AJUSTAR PARÁMETROS</h2>
          <button className="modal-close-btn" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          <div className="modal-info">
            <p>Configure los parámetros de consumo y reportes para cada colonia. Los valores se utilizarán para recalcular la priorización.</p>
          </div>

          <div className="modal-table-container">
            <table className="modal-table">
              <thead>
                <tr>
                  <th>COLONIA</th>
                  <th>NÚM. DE REPORTES</th>
                  <th>CONSUMO TOTAL DE AGUA POR COLONIA</th>
                </tr>
              </thead>
              <tbody>
                {colonias.map((colonia, index) => (
                  <tr key={index}>
                    <td className="colonia-nombre-cell">{colonia.colonia}</td>
                    <td>
                      <input
                        type="number"
                        min="0"
                        className="modal-input"
                        value={parametros.reportes[colonia.colonia] ?? ''}
                        onChange={(e) => handleReportesChange(colonia.colonia, e.target.value)}
                      />
                    </td>
                    <td>
                      <div className="input-with-unit">
                        <input
                          type="number"
                          min="0"
                          className="modal-input"
                          value={parametros.consumo[colonia.colonia] ?? ''}
                          onChange={(e) => handleConsumoChange(colonia.colonia, e.target.value)}
                        />
                        <span className="input-unit">m³</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="modal-footer">
          <button
            className="modal-btn modal-btn-cancel"
            onClick={onClose}
            disabled={cargando}
          >
            CANCELAR
          </button>
          <button
            className="modal-btn modal-btn-save"
            onClick={handleGuardar}
            disabled={cargando}
          >
            {cargando ? 'GUARDANDO...' : 'GUARDAR'}
          </button>
        </div>
      </div>
    </div>
  );
};

const MapaColonias = () => {
  const [geoJsonData, setGeoJsonData] = useState(null);
  const [datosPrioridad, setDatosPrioridad] = useState(null);
  const [coloniaSeleccionada, setColoniaSeleccionada] = useState(null);
  const [boundsSeleccionado, setBoundsSeleccionado] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [modalAbierto, setModalAbierto] = useState(false);
  const geoJsonLayerRef = useRef(null);
  const [geoKey, setGeoKey] = useState(0);
  const [boundsIniciales, setBoundsIniciales] = useState(null);
  const [menuMovilAbierto, setMenuMovilAbierto] = useState(false);
  const [usuario, setUsuario] = useState(null);

  useEffect(() => {
    cargarDatos();
    cargarUsuario();
  }, []);

  const cargarUsuario = () => {
    try {
      const usuarioGuardado = localStorage.getItem('usuario');
      if (usuarioGuardado) {
        setUsuario(JSON.parse(usuarioGuardado));
      }
    } catch (error) {
      console.error('Error al cargar usuario:', error);
    }
  };

  const cargarDatos = async () => {
    try {
      const responseGeo = await fetch('/Colonias_GAM_GeoJSON.geojson');
      const geoData = await responseGeo.json();
      const bounds = L.geoJSON(geoData).getBounds();

      setGeoJsonData(geoData);
      setBoundsSeleccionado(bounds);
      setBoundsIniciales(bounds);
      // GET para obtener los datos del backend
      const responsePrioridad = await fetch(`${API_URL}/api/optimize/`);
      const prioridadData = await responsePrioridad.json();

      prioridadData.colonias = prioridadData.colonias.map(c => ({
        colonia: c.colonia ?? c.nombre,
        prioridad: c.prioridad,
        ranking: c.ranking
      }));
      setDatosPrioridad(prioridadData);
      setGeoKey(prev => prev + 1); 

      setCargando(false);
    } catch (error) {
      console.error('Error cargando datos:', error);
      setCargando(false);
    }
  };

  const guardarParametros = async (parametros) => {
    try {
      setCargando(true);
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_URL}/api/optimize/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify(parametros)
      });

      if (!response.ok) {
        throw new Error('Error al enviar parámetros al servidor');
      }

      const prioridadData = await response.json();
      prioridadData.colonias = prioridadData.colonias.map(c => ({
        colonia: c.colonia ?? c.nombre,
        prioridad: c.prioridad,
        ranking: Number(c.ranking)
      }));
      setDatosPrioridad(prioridadData);
      setGeoKey(prev => prev + 1);
      setCargando(false);

      setCargando(false);
    } catch (error) {
      console.error('Error guardando parámetros:', error);
      setCargando(false);
      throw error;
    }
  };

  const obtenerRankingColonia = (nombreColonia) => {
    if (!datosPrioridad || !datosPrioridad.colonias) return null;
    const colonia = datosPrioridad.colonias.find(
      c => c?.colonia?.toLowerCase() === nombreColonia?.toLowerCase()
    );
    return colonia ? colonia.ranking : null;
  };

  const obtenerDatosColonia = (nombreColonia) => {
    if (!datosPrioridad || !datosPrioridad.colonias) return null;
    return datosPrioridad.colonias.find(
      c => c?.colonia?.toLowerCase() === nombreColonia?.toLowerCase()
    );
  };

  const style = (feature) => {
    if (!datosPrioridad?.colonias) {
      return {
        fillColor: '#CCCCCC',
        color: "#666",
        fillOpacity: 0.7,
        weight: 1,
      };
    }

    const nombreColonia = feature?.properties?.name || '';
    const ranking = obtenerRankingColonia(nombreColonia);

    return {
      fillColor: COLORES_RANKING[ranking] ?? '#CCCCCC',
      weight: 1.5,
      opacity: 1,
      color: '#555555',
      fillOpacity: 0.9,
    };
  };

  const onEachFeature = (feature, layer) => {
    const nombreColonia = feature?.properties?.name || '';
    const datosColonia = obtenerDatosColonia(nombreColonia);

    if (datosColonia) {
      const tooltipContent = `
        <div style="text-align: center; font-family: Arial, sans-serif;">
          <strong style="font-size: 14px;">${datosColonia.colonia}</strong><br/>
          <span style="color: ${COLORES_RANKING[datosColonia.ranking]}; font-size: 18px;">●</span><br/>
          <span style="font-size: 12px;">Ranking: ${datosColonia.ranking}/7</span><br/>
          <span style="font-size: 12px;">Prioridad: ${(datosColonia.prioridad * 100).toFixed(1)}%</span>
        </div>
      `;
      layer.bindTooltip(tooltipContent);
    }

    layer.on('mouseover', function (e) {
      const targetLayer = e.target;
      targetLayer.setStyle({
        weight: 5,
        color: '#000000',
        fillOpacity: 0.8,
      });
      if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
        targetLayer.bringToFront();
      }
    });

    layer.on('mouseout', function (e) {
      if (coloniaSeleccionada !== nombreColonia) {
        geoJsonLayerRef.current.resetStyle(e.target);
      }
    });

    layer.on('click', function (e) {
      const targetLayer = e.target;
      setColoniaSeleccionada(nombreColonia);
      const bounds = targetLayer.getBounds();
      setBoundsSeleccionado(bounds);
    });
  };

  const coloniasOrdenadas = datosPrioridad?.colonias
    ? [...datosPrioridad.colonias].sort((a, b) => a.ranking - b.ranking)
    : [];

  if (cargando) {
    return (
      <div className="awoda-loading">
        <div className="awoda-spinner"></div>
        <p>Cargando sistema de distribución de agua AWODA...</p>
      </div>
    );
  }

  return (
    <div className="awoda-container">
      {usuario && usuario.rol_usuario !== 'administrador' && (
        <ModalParametros
          isOpen={modalAbierto}
          onClose={() => setModalAbierto(false)}
          colonias={coloniasOrdenadas}
          onGuardar={guardarParametros}
        />
      )}

      {/* 🆕 Botón hamburguesa (solo visible en móvil) */}
      <button 
        className="awoda-hamburger-btn"
        onClick={() => setMenuMovilAbierto(!menuMovilAbierto)}
        aria-label="Abrir menú de colonias"
      >
        <span></span>
        <span></span>
        <span></span>
      </button>

      {/* 🆕 Overlay para cerrar menú al tocar fuera */}
      {menuMovilAbierto && (
        <div 
          className="awoda-menu-overlay" 
          onClick={() => setMenuMovilAbierto(false)}
        ></div>
      )}

      {/* Sidebar izquierdo con clase condicional */}
      <aside className={`awoda-sidebar ${menuMovilAbierto ? 'menu-abierto' : ''}`}>
        <div className="awoda-sidebar-header">
          <p className="awoda-disclaimer">
            La priorización de suministro está delimitada a un conjunto
            de colonias previamente seleccionadas. Las sugerencias generadas
            se basan en los datos disponibles de esta área y no deben
            extrapolarse a otras regiones sin el análisis correspondiente
          </p>
        </div>

        <h3 style={{ marginTop: "10px", padding: "0 20px" }}>Colonias disponibles:</h3>
        <div className="awoda-colonias-lista">
          <button
            className={`awoda-colonia-item ${!coloniaSeleccionada ? "active" : ""}`}
            onClick={() => {
              setColoniaSeleccionada(null);
              if (boundsIniciales) setBoundsSeleccionado(boundsIniciales);
              setMenuMovilAbierto(false); // 🆕 Cerrar menú al seleccionar
            }}
          >
            <span className="awoda-colonia-icono">↻</span>
            <span className="awoda-colonia-nombre">Vista General</span>
          </button>

          {coloniasOrdenadas.map((colonia, index) => (
            <button
              key={index}
              className={`awoda-colonia-item ${coloniaSeleccionada === colonia.colonia ? "active" : ""}`}
              onClick={() => {
                setColoniaSeleccionada(colonia.colonia);

                const feature = geoJsonData.features.find(
                  f => f.properties.name.toLowerCase() === colonia.colonia.toLowerCase()
                );

                if (feature && geoJsonLayerRef.current) {
                  const layer = Object.values(geoJsonLayerRef.current._layers).find(
                    l => l.feature?.properties?.name === colonia.colonia
                  );
                  if (layer) {
                    const bounds = layer.getBounds();
                    setBoundsSeleccionado(bounds);
                  }
                }
                setMenuMovilAbierto(false); // 🆕 Cerrar menú al seleccionar
              }}
            >
              <span className="awoda-colonia-icono">→</span>
              <span className="awoda-colonia-nombre">{colonia.colonia}</span>
            </button>
          ))}
        </div>

        <div className="awoda-sidebar-footer">
          <p className="awoda-disclaimer">
            Sistema desarrollado conforme a la Ley de Aguas Nacionales
          </p>
        </div>
      </aside>

      <main className="awoda-main" style={{ backgroundColor: "#fafafa", padding: "0px 0" }}>
        <div className="awoda-map-header" style={{ textAlign: "center", marginBottom: "20px" }}>
          <h2 style={{ fontWeight: "700", color: "#2c3e50" }}>Mapa de Prioridad</h2>
        </div>

        <div style={{ display: "flex", justifyContent: "center" }}>
          <div style={{
            width: "80%",
            maxWidth: "1200px",
            height: "600px",
            borderRadius: "10px",
            overflow: "hidden",
            boxShadow: "0 2px 10px rgba(0, 0, 0, 0.15)",
            border: "1px solid #ddd",
          }}>
            <MapContainer
              center={[19.4855, -99.1189]}
              zoom={12}
              style={{ height: "100%", width: "100%" }}
              className="awoda-leaflet-map"
            >
              <ForceMapResize />
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              />

              {geoJsonData && datosPrioridad?.colonias?.length > 0 && (
                <GeoJSON
                  key={geoKey}
                  ref={geoJsonLayerRef}
                  data={geoJsonData}
                  style={style}
                  onEachFeature={onEachFeature}
                />
              )}

              {boundsSeleccionado && <FitBounds bounds={boundsSeleccionado} />}
              <LeyendaColores />
            </MapContainer>
          </div>
        </div>
      </main>

      <aside className="awoda-sidebar-right">
        <div className="awoda-sidebar-right-content">
          <div className="awoda-distribucion-header">
            <h2>Distribución Sugerida</h2>
          </div>

          <div className="awoda-disclaimer-footer">
            <p>
              Esta propuesta de distribución fue generada por la IA de AWODA y está
              sujeta a validación de autoridades de la SEGIAGUA CDMX.
            </p>
          </div>

          {!coloniaSeleccionada && (
            <div className="awoda-distribucion-lista">
              {coloniasOrdenadas.map((colonia, index) => (
                <div key={index} className="awoda-distribucion-item">
                  <span className="awoda-distribucion-numero">{index + 1}.</span>
                  <span className="awoda-distribucion-nombre">{colonia.colonia}</span>
                </div>
              ))}
            </div>
          )}

          {coloniaSeleccionada && (
            <>
              <div className="awoda-distribucion-lista">
                <ColoniaEdificaciones
                  edificaciones={datosPrioridad.edificaciones}
                />
              </div>

              <div className="awoda-colonia-detalles">
                <h3>Detalles de la Colonia</h3>

                <div className="awoda-detalle-item">
                  <strong>Nombre:</strong> {coloniaSeleccionada}
                </div>

                <div className="awoda-detalle-item">
                  <strong>Ranking:</strong>{" "}
                  {obtenerDatosColonia(coloniaSeleccionada).ranking}/7
                </div>

                <div className="awoda-detalle-item">
                  <strong>Prioridad:</strong>{" "}
                  {(obtenerDatosColonia(coloniaSeleccionada).prioridad * 100).toFixed(2)}%
                </div>

                <div
                  className="awoda-color-indicator"
                  style={{
                    backgroundColor:
                      COLORES_RANKING[
                        obtenerDatosColonia(coloniaSeleccionada).ranking
                      ],
                  }}
                ></div>
              </div>
            </>
          )}

          {usuario && usuario.rol_usuario !== 'administrador' && (
            <button
              className="awoda-btn-ajustar"
              onClick={() => setModalAbierto(true)}
            >
              AJUSTAR PARÁMETROS
            </button>
          )}
        </div>
      </aside>
    </div>
  );
};

export default MapaColonias;
