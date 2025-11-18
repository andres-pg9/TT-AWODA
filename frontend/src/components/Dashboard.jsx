import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import MapaColonias from './MapaColonias';
import Graficas from './Graficas';
import Historial from './Historial';
import './Dashboard.css';

/**
 * DASHBOARD - Componente principal después del login
 * Contiene el navbar y las diferentes vistas (Mapa, Gráficas, Historial)
 */

const Dashboard = () => {
    const navigate = useNavigate();
    const [usuario, setUsuario] = useState(null);
    const [cargando, setCargando] = useState(true);
    const [vistaActual, setVistaActual] = useState('principal');

    // Validar token al cargar el componente
    useEffect(() => {
        const validarSesion = async () => {
            const token = localStorage.getItem('token');
            const usuarioGuardado = localStorage.getItem('usuario');

            if (!token) {
                console.log('❌ No hay token, redirigiendo a login...');
                navigate('/login');
                return;
            }

            if (usuarioGuardado) {
                try {
                    setUsuario(JSON.parse(usuarioGuardado));
                    setCargando(false);
                } catch (err) {
                    console.error('Error al parsear usuario:', err);
                    localStorage.removeItem('usuario');
                }
            }

            try {
                const response = await fetch('http://127.0.0.1:8000/api/auth/validate-token', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (!response.ok) {
                    console.log('❌ Token inválido, redirigiendo a login...');
                    localStorage.removeItem('token');
                    localStorage.removeItem('usuario');
                    navigate('/login');
                    return;
                }

                const data = await response.json();
                console.log('✅ Token válido:', data);
                setCargando(false);

            } catch (err) {
                console.error('Error al validar token:', err);
                setCargando(false);
            }
        };

        validarSesion();
    }, [navigate]);

    const handleLogout = async () => {
        const token = localStorage.getItem('token');

        try {
            await fetch('http://127.0.0.1:8000/api/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            console.log('✅ Logout exitoso');
        } catch (err) {
            console.error('Error al hacer logout:', err);
        } finally {
            localStorage.removeItem('token');
            localStorage.removeItem('usuario');
            navigate('/login');
        }
    };

    const cambiarVista = (vista) => {
        setVistaActual(vista);
    };

    if (cargando) {
        return (
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100vh',
                flexDirection: 'column',
                gap: '20px'
            }}>
                <div style={{
                    width: '50px',
                    height: '50px',
                    border: '5px solid #f3f3f3',
                    borderTop: '5px solid #3498db',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                }}></div>
                <p>Validando sesión...</p>
            </div>
        );
    }

    return (
        <div className="dashboard-container">
            {/* Navbar superior */}
            <nav className="awoda-navbar">
                <div className="awoda-navbar-brand">
                    <img src="/isotipo-awoda.png" alt="AWODA" className="awoda-icon" />
                    <span className="awoda-title">AWODA</span>
                </div>
                <div className="awoda-navbar-menu">
                    <button
                        onClick={() => cambiarVista('principal')}
                        className={`awoda-nav-link ${vistaActual === 'principal' ? 'active' : ''}`}
                    >
                        PRINCIPAL
                    </button>
                    <button
                        onClick={() => cambiarVista('graficas')}
                        className={`awoda-nav-link ${vistaActual === 'graficas' ? 'active' : ''}`}
                    >
                        GRÁFICAS
                    </button>
                    <button
                        onClick={() => cambiarVista('historial')}
                        className={`awoda-nav-link ${vistaActual === 'historial' ? 'active' : ''}`}
                    >
                        HISTORIAL
                    </button>
                    <button
                        onClick={() => cambiarVista('entrenamiento')}
                        className={`awoda-nav-link ${vistaActual === 'entrenamiento' ? 'active' : ''}`}
                    >
                        ENTRENAMIENTO
                    </button>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                    {usuario && (
                        <span style={{ color: 'white', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <svg width="16" height="16" fill="white" viewBox="0 0 16 16">
                                <path d="M8 8a3 3 0 1 0 0-6a3 3 0 0 0 0 6z" />
                                <path fillRule="evenodd" d="M14 14s-1-4-6-4s-6 4-6 4s2 2 6 2s6-2 6-2z" />
                            </svg>
                            {usuario.nombre_empleado}
                        </span>

                    )}
                    <button className="awoda-navbar-logout" onClick={handleLogout}>
                        Cerrar sesión
                    </button>
                </div>
            </nav>

            {/* Contenido principal según la vista */}
            {vistaActual === 'principal' && <MapaColonias />}
            {vistaActual === 'graficas' && <Graficas />}
            {vistaActual === 'historial' && <Historial />}
            {vistaActual === 'entrenamiento' && (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: 'calc(100vh - 60px)',
                    flexDirection: 'column',
                    gap: '20px',
                    color: '#666'
                }}>
                    <span style={{ fontSize: '64px' }}>🤖</span>
                    <h2>Entrenamiento</h2>
                    <p>Esta sección estará disponible próximamente</p>
                </div>
            )}

            {/* Animación de spinner */}
            <style>
                {`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                `}
            </style>
        </div>
    );
};

export default Dashboard;