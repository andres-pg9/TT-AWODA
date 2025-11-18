import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import MapaColonias from './MapaColonias';
import './Dashboard.css';

/**
 * DASHBOARD - Componente principal después del login
 * Contiene el navbar y el mapa de colonias
 * 🔥 NUEVO: Valida que el usuario tenga token antes de mostrar contenido
 */

const Dashboard = () => {
    const navigate = useNavigate();
    const [usuario, setUsuario] = useState(null);
    const [cargando, setCargando] = useState(true);

    // 🔥 CAMBIO 1: Validar token al cargar el componente
    useEffect(() => {
        const validarSesion = async () => {
            const token = localStorage.getItem('token');
            const usuarioGuardado = localStorage.getItem('usuario');

            // Si no hay token, redirigir al login
            if (!token) {
                console.log('❌ No hay token, redirigiendo a login...');
                navigate('/login');
                return;
            }

            // Si hay usuario guardado, usarlo
            if (usuarioGuardado) {
                try {
                    setUsuario(JSON.parse(usuarioGuardado));
                    setCargando(false);
                } catch (err) {
                    console.error('Error al parsear usuario:', err);
                    localStorage.removeItem('usuario');
                }
            }

            // 🔥 OPCIONAL: Validar token con el backend
            try {
                const response = await fetch('http://127.0.0.1:8000/api/auth/validate-token', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (!response.ok) {
                    // Token inválido o expirado
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
                // Si hay error de red pero tenemos token, continuar
                setCargando(false);
            }
        };

        validarSesion();
    }, [navigate]);

    // 🔥 CAMBIO 2: Hacer logout con el backend
    const handleLogout = async () => {
        const token = localStorage.getItem('token');

        try {
            // Llamar al endpoint de logout del backend
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
            // Siempre eliminar token y redirigir
            localStorage.removeItem('token');
            localStorage.removeItem('usuario');
            navigate('/login');
        }
    };

    // 🔥 CAMBIO 3: Mostrar indicador de carga mientras valida
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
                    <a href="#principal" className="awoda-nav-link active">PRINCIPAL</a>
                    <a href="#graficas" className="awoda-nav-link">GRÁFICAS</a>
                    <button onClick={() => navigate('/historial')} className="awoda-nav-link" style={{cursor: 'pointer'}}>
                        HISTORIAL
                    </button>
                    <a href="#entrenamiento" className="awoda-nav-link">ENTRENAMIENTO</a>
                </div>
                
                {/* 🔥 CAMBIO 4: Mostrar nombre del usuario */}
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

            {/* Componente principal del mapa */}
            <MapaColonias />

            {/* Footer */}
            <footer className="awoda-footer">
                <p>
                    <a href="/aviso-privacidad">Aviso de privacidad</a> | © 2025 AWODA ESCOM IPN
                </p>
            </footer>

            {/* 🔥 Agregar animación de spinner */}
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