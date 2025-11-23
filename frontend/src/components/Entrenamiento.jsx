import { useState } from 'react';
import './Entrenamiento.css';

const Entrenamiento = () => {
    const [diapositivaActual, setDiapositivaActual] = useState(0);

    const diapositivas = [
        {
            titulo: "El Problema que Resuelve PSO",
            contenido: (
                <div className="entrenamiento-contenido">
                    <p className="entrenamiento-intro">
                        Imagina que tienes que distribuir agua en una ciudad durante una sequia.
                        Necesitas decidir:
                    </p>
                    <div className="grid-parametros">
                        <div className="parametro-card parametro-alfa">
                            <p className="parametro-nombre">α (Alfa)</p>
                            <p className="parametro-desc">¿Cuanto peso darle a las normas legales?</p>
                        </div>
                        <div className="parametro-card parametro-beta">
                            <p className="parametro-nombre">β (Beta)</p>
                            <p className="parametro-desc">¿Cuanto peso darle a las preferencias ciudadanas?</p>
                        </div>
                        <div className="parametro-card parametro-gamma">
                            <p className="parametro-nombre">γ (Gamma)</p>
                            <p className="parametro-desc">¿Cuanto peso darle al consumo historico?</p>
                        </div>
                        <div className="parametro-card parametro-delta">
                            <p className="parametro-nombre">δ (Delta)</p>
                            <p className="parametro-desc">¿Cuanto peso darle a los reportes de fallas?</p>
                        </div>
                    </div>
                    <div className="desafio-box">
                        <p className="desafio-titulo">Desafio:</p>
                        <p className="desafio-texto">Encontrar el balance optimo entre estos 4 factores que maximice la "calidad" de la distribucion.</p>
                    </div>
                </div>
            )
        },
        {
            titulo: "¿Que es PSO?",
            contenido: (
                <div className="entrenamiento-contenido">
                    <div className="pso-definicion">
                        <h3 className="pso-titulo">Particle Swarm Optimization</h3>
                        <p className="pso-descripcion">
                            Algoritmo de optimizacion inspirado en el comportamiento de bandadas de aves buscando comida.
                        </p>
                    </div>

                    <div className="analogia-box">
                        <h4 className="analogia-titulo">Analogia Simple:</h4>
                        <p className="analogia-intro">
                            Imagina que tienes 30 drones buscando el punto mas alto de una montaña en la niebla:
                        </p>
                        <ul className="analogia-lista">
                            <li>Cada dron (particula) explora una zona diferente</li>
                            <li>Todos comparten informacion sobre la mejor altura encontrada</li>
                            <li>Se mueven influenciados por:</li>
                        </ul>
                        <div className="influencias-lista">
                            <p>✓ Su propia mejor experiencia</p>
                            <p>✓ La mejor experiencia del grupo</p>
                            <p>✓ Su inercia actual</p>
                        </div>
                    </div>

                    <div className="resultado-box">
                        <p className="resultado-texto">
                            Despues de muchas iteraciones, convergen hacia la cima de la montaña.
                        </p>
                    </div>
                </div>
            )
        },
        {
            titulo: "El Espacio de Busqueda",
            contenido: (
                <div className="entrenamiento-contenido">
                    <p className="intro-texto">
                        En lugar de buscar en una montaña fisica, PSO busca en un espacio matematico de <span className="highlight">4 dimensiones</span>:
                    </p>

                    <div className="puntos-box">
                        <h4 className="puntos-titulo">Puntos en el espacio (configuraciones de pesos):</h4>
                        <div className="puntos-lista">
                            <div className="punto-item">
                                <span className="punto-label">Punto A =</span> [α=0.25, β=0.25, γ=0.25, δ=0.25]
                                <span className="punto-utilidad punto-normal">→ Utilidad: 72.45</span>
                            </div>
                            <div className="punto-item">
                                <span className="punto-label">Punto B =</span> [α=0.50, β=0.30, γ=0.15, δ=0.05]
                                <span className="punto-utilidad punto-bueno">→ Utilidad: 74.12</span>
                            </div>
                            <div className="punto-item punto-mejor">
                                <span className="punto-label">Punto C =</span> [α=0.35, β=0.31, γ=0.21, δ=0.13]
                                <span className="punto-utilidad punto-optimo">→ Utilidad: 85.34 ⭐ ¡Este es el mejor!</span>
                            </div>
                        </div>
                    </div>

                    <div className="restricciones-grid">
                        <div className="restriccion-card">
                            <p className="restriccion-titulo">Restriccion:</p>
                            <p className="restriccion-formula">α + β + γ + δ = 1.0</p>
                            <p className="restriccion-nota">(los pesos suman 100%)</p>
                        </div>
                        <div className="objetivo-card">
                            <p className="objetivo-titulo">Objetivo:</p>
                            <p className="objetivo-texto">Encontrar el punto con la mayor utilidad</p>
                        </div>
                    </div>
                </div>
            )
        },
        {
            titulo: "Componentes del Algoritmo",
            contenido: (
                <div className="entrenamiento-contenido">
                    <div className="componente-box componente-particulas">
                        <h4 className="componente-titulo">1️⃣ Particulas (Soluciones Candidatas)</h4>
                        <p className="componente-intro">Cada particula representa una posible solucion:</p>
                        <div className="codigo-box">
                            <p><span className="codigo-var">posicion</span> = [0.30, 0.28, 0.25, 0.17]  <span className="codigo-comentario">// Sus pesos actuales</span></p>
                            <p><span className="codigo-var">velocidad</span> = [-0.05, +0.03, +0.02, 0.0]  <span className="codigo-comentario">// Hacia donde se mueve</span></p>
                            <p><span className="codigo-var">mejor_personal</span> = [0.32, 0.26, 0.24, 0.18]  <span className="codigo-comentario">// Su mejor historico</span></p>
                            <p><span className="codigo-var">fitness_personal</span> = 82.5  <span className="codigo-comentario">// Utilidad de su mejor</span></p>
                        </div>
                    </div>

                    <div className="componente-box componente-global">
                        <h4 className="componente-titulo">2️⃣ Mejor Global (Gbest)</h4>
                        <p className="componente-intro">La mejor solucion encontrada por TODO el enjambre:</p>
                        <div className="codigo-box">
                            <p><span className="codigo-var">mejor_global_posicion</span> = [0.35, 0.31, 0.21, 0.13]</p>
                            <p><span className="codigo-var">mejor_global_utilidad</span> = 85.34</p>
                        </div>
                        <p className="componente-nota">Como un faro que todos los drones pueden ver</p>
                    </div>

                    <div className="componente-box componente-heuristica">
                        <h4 className="componente-titulo">3️⃣ La Heuristica</h4>
                        <p className="componente-intro">Formula que calcula la prioridad de cada combinacion:</p>
                        <div className="formula-box">
                            <p className="formula">H = α·x + β·y + γ·z + δ·w</p>
                        </div>
                    </div>
                </div>
            )
        },
        {
            titulo: "Los Parametros Explicados",
            contenido: (
                <div className="entrenamiento-contenido">
                    <div className="parametro-detalle parametro-detalle-azul">
                        <h4 className="parametro-detalle-titulo">n_particles = 30</h4>
                        <p className="parametro-detalle-desc">Tamaño del enjambre</p>
                        <p className="parametro-detalle-nota">
                            <span className="negrita">Balance tipico:</span> 30-50 particulas para explorar simultaneamente
                        </p>
                    </div>

                    <div className="parametro-detalle parametro-detalle-purpura">
                        <h4 className="parametro-detalle-titulo">n_iterations = 150</h4>
                        <p className="parametro-detalle-desc">Numero de ciclos de refinamiento</p>
                        <p className="parametro-detalle-nota">
                            Suficiente para que el algoritmo converja al optimo
                        </p>
                    </div>

                    <div className="parametro-detalle parametro-detalle-verde">
                        <h4 className="parametro-detalle-titulo">w = 0.7 (Inercia)</h4>
                        <p className="parametro-detalle-desc">Porcentaje de direccion actual que mantiene cada particula</p>
                        <div className="codigo-box">
                            nueva_velocidad = <span className="codigo-var">0.7</span> * velocidad_actual + otras_fuerzas
                        </div>
                        <p className="parametro-detalle-nota">70% mantiene rumbo, 30% se adapta</p>
                    </div>

                    <div className="parametro-doble-grid">
                        <div className="parametro-detalle parametro-detalle-naranja">
                            <h4 className="parametro-detalle-titulo-sm">c1 = 1.5</h4>
                            <p className="parametro-detalle-desc-sm">Coeficiente Cognitivo</p>
                            <p className="parametro-detalle-nota-sm">Confianza en su propia memoria</p>
                        </div>
                        <div className="parametro-detalle parametro-detalle-rosa">
                            <h4 className="parametro-detalle-titulo-sm">c2 = 1.5</h4>
                            <p className="parametro-detalle-desc-sm">Coeficiente Social</p>
                            <p className="parametro-detalle-nota-sm">Atraccion hacia el mejor global</p>
                        </div>
                    </div>

                    <div className="parametro-detalle parametro-detalle-gris">
                        <h4 className="parametro-detalle-titulo">seed = 42</h4>
                        <p className="parametro-detalle-desc">Semilla aleatoria para reproducibilidad</p>
                        <p className="parametro-detalle-nota">
                            Con semilla: siempre da el mismo resultado (util para pruebas)
                        </p>
                    </div>
                </div>
            )
        },
        {
            titulo: "Resultado Final",
            contenido: (
                <div className="entrenamiento-contenido">
                    <div className="resultado-final-box">
                        <h3 className="resultado-final-titulo">Pesos Optimos Encontrados</h3>
                        <div className="resultado-pesos-grid">
                            <div className="peso-card peso-alfa">
                                <p className="peso-valor">35.12%</p>
                                <p className="peso-nombre">α (Legal)</p>
                            </div>
                            <div className="peso-card peso-beta">
                                <p className="peso-valor">30.87%</p>
                                <p className="peso-nombre">β (Social)</p>
                            </div>
                            <div className="peso-card peso-gamma">
                                <p className="peso-valor">21.34%</p>
                                <p className="peso-nombre">γ (Consumo)</p>
                            </div>
                            <div className="peso-card peso-delta">
                                <p className="peso-valor">12.67%</p>
                                <p className="peso-nombre">δ (Reportes)</p>
                            </div>
                        </div>
                    </div>

                    <div className="utilidad-box">
                        <div className="utilidad-contenido">
                            <span className="utilidad-icono">📈</span>
                            <p className="utilidad-valor">Utilidad: 85.34/100</p>
                        </div>
                        <p className="utilidad-descripcion">El mejor valor encontrado despues de 150 iteraciones</p>
                    </div>

                    <div className="calculo-box">
                        <h4 className="calculo-titulo">Con estos pesos se calculan:</h4>
                        <ul className="calculo-lista">
                            <li>
                                <span className="calculo-punto"></span>
                                <span><strong>Ranking de colonias:</strong> Lindavista II, Magdalena Salinas, Lindavista I...</span>
                            </li>
                            <li>
                                <span className="calculo-punto"></span>
                                <span><strong>Ranking de edificaciones:</strong> Hospital, Clinicas Particulares, Escuelas...</span>
                            </li>
                        </ul>
                    </div>

                    <div className="conclusion-box">
                        <p className="conclusion-texto">
                            Estos pesos optimizados se usan para distribuir el agua de manera justa y eficiente
                        </p>
                    </div>
                </div>
            )
        }
    ];

    const siguienteDiapositiva = () => {
        if (diapositivaActual < diapositivas.length - 1) {
            setDiapositivaActual(diapositivaActual + 1);
        }
    };

    const anteriorDiapositiva = () => {
        if (diapositivaActual > 0) {
            setDiapositivaActual(diapositivaActual - 1);
        }
    };

    const irADiapositiva = (indice) => {
        setDiapositivaActual(indice);
    };

    return (
        <div className="entrenamiento-container">
            <div className="awoda-map-header" style={{ textAlign: "center", marginBottom: "20px", background: "white" }}>
          <h2 style={{ fontSize: "2rem", fontWeight: "700", color: "#2c3e50" }}>Entrenamiento</h2>
            </div>

            <div className="diapositiva-wrapper">
                <div className="diapositiva-header">
                    <span className="diapositiva-icono">{diapositivas[diapositivaActual].icono}</span>
                    <h3 className="diapositiva-titulo">{diapositivas[diapositivaActual].titulo}</h3>
                </div>

                <div className="diapositiva-contenido">
                    {diapositivas[diapositivaActual].contenido}
                </div>

                <div className="diapositiva-navegacion">
                    <button
                        className="btn-navegacion btn-anterior"
                        onClick={anteriorDiapositiva}
                        disabled={diapositivaActual === 0}
                    >
                        ← Anterior
                    </button>

                    <div className="indicadores-progreso">
                        {diapositivas.map((_, indice) => (
                            <button
                                key={indice}
                                className={`indicador ${indice === diapositivaActual ? 'activo' : ''}`}
                                onClick={() => irADiapositiva(indice)}
                                aria-label={`Ir a diapositiva ${indice + 1}`}
                            />
                        ))}
                    </div>

                    <button
                        className="btn-navegacion btn-siguiente"
                        onClick={siguienteDiapositiva}
                        disabled={diapositivaActual === diapositivas.length - 1}
                    >
                        Siguiente →
                    </button>
                </div>

                <div className="contador-diapositivas">
                    {diapositivaActual + 1} / {diapositivas.length}
                </div>
            </div>
        </div>
    );
};

export default Entrenamiento;
