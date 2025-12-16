import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';
import { API_URL } from '../config';
import ModalAvisoPrivacidad from './ModalAvisoPrivacidad';

const Login = () => {
    const [formData, setFormData] = useState({
        empleado: '',
        password: '',
        rememberMe: false
    });
    const [error, setError] = useState('');
    const [cargando, setCargando] = useState(false);
    const [modalPrivacidadAbierto, setModalPrivacidadAbierto] = useState(false);
    const navigate = useNavigate();

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setCargando(true);

        // Validación básica
        if (!formData.empleado || !formData.password) {
            setError('Por favor, completa todos los campos');
            setCargando(false);
            return;
        }

        try {
            // 🔥 CAMBIO 1: Llamada REAL al backend
            const response = await fetch(`${API_URL}/api/auth/login`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json' 
                },
                body: JSON.stringify({
                    numero_empleado: parseInt(formData.empleado), // Convertir a número
                    password: formData.password
                })
            });

            // 🔥 CAMBIO 2: Manejar respuesta del backend
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Credenciales incorrectas');
            }

            const data = await response.json();

            // 🔥 CAMBIO 3: Guardar token REAL y datos del usuario
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('usuario', JSON.stringify(data.usuario));

            // Si seleccionó "Recordarme", guardar número de empleado
            if (formData.rememberMe) {
                localStorage.setItem('rememberedEmpleado', formData.empleado);
            } else {
                localStorage.removeItem('rememberedEmpleado');
            }

            console.log('✅ Login exitoso');

            // Redirigir al dashboard
            navigate('/dashboard');

        } catch (err) {
            console.error('❌ Error de login:', err);
            setError(err.message || 'Error al iniciar sesión. Intenta nuevamente.');
        } finally {
            setCargando(false);
        }
    };

    // Cargar número de empleado guardado al montar el componente
    React.useEffect(() => {
        const rememberedEmpleado = localStorage.getItem('rememberedEmpleado');
        if (rememberedEmpleado) {
            setFormData(prev => ({
                ...prev,
                empleado: rememberedEmpleado,
                rememberMe: true
            }));
        }
    }, []);

    return (
        <div className="login-container">
            {/* Panel izquierdo con información */}
            <div className="login-panel-izquierdo">
                <div className="login-contenido-izquierdo">
                    <h1 className="login-titulo">¡Bienvenido a AWODA!</h1>
                    <p className="login-descripcion">
                        AWODA es un proyecto desarrollado en la Escuela Superior de Cómputo (ESCOM) del Instituto Politécnico Nacional (IPN).
                        Esta aplicación web emplea Inteligencia Artificial para generar sugerencias sobre la priorización en la distribución de agua potable,
                        integrando factores sociales, legales y de consumo con el fin de apoyar una gestión más equitativa y sustentable del recurso hídrico.
                    </p>
                    <p className="login-descripcion">
                        Las recomendaciones que ofrece AWODA son consultivas y están sujetas a validación y decisión final por parte de las autoridades competentes.
                        El sistema promueve la toma de decisiones informadas, alineadas con la Ley de Aguas Nacionales, el Plan Nacional Hídrico y el Derecho Humano al Agua
                        establecido en la Constitución Política de los Estados Unidos Mexicanos.
                    </p>
                </div>
            </div>


            {/* Panel derecho con formulario */}
            <div className="login-panel-derecho">
                <div className="login-formulario-container">
                    {/* Logo */}
                    <div className="login-logo-container">
                        <img
                            src="/isologo-awoda.png"
                            alt="AWODA Logo"
                            className="login-logo"
                        />
                    </div>

                    {/* Formulario */}
                    <form onSubmit={handleSubmit} className="login-form">
                        {error && (
                            <div className="login-error">
                                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                </svg>
                                {error}
                            </div>
                        )}

                        <div className="login-input-group">
                            <input
                                type="text"
                                name="empleado"
                                placeholder="Número de empleado"
                                value={formData.empleado}
                                onChange={handleChange}
                                className="login-input"
                                required
                            />
                        </div>

                        <div className="login-input-group">
                            <input
                                type="password"
                                name="password"
                                placeholder="Contraseña"
                                value={formData.password}
                                onChange={handleChange}
                                className="login-input"
                                required
                            />
                        </div>

                        <div className="login-opciones">
                            <label className="login-checkbox-label">
                                <input
                                    type="checkbox"
                                    name="rememberMe"
                                    checked={formData.rememberMe}
                                    onChange={handleChange}
                                    className="login-checkbox"
                                />
                                <span>Recordarme</span>
                            </label>

                        </div>

                        <button
                            type="submit"
                            className="login-btn-submit"
                            disabled={cargando}
                        >
                            {cargando ? 'Iniciando sesión...' : 'Iniciar sesión'}
                        </button>

                    </form>
                </div>
            </div>

            <footer className="login-footer">
                <p>
                    <button 
                        onClick={() => setModalPrivacidadAbierto(true)} 
                        className="login-footer-link"
                        type="button"
                    >
                        Aviso de privacidad
                    </button> | © 2025 AWODA ESCOM IPN
                </p>
            </footer>

            {/* Modal de Aviso de Privacidad */}
            <ModalAvisoPrivacidad 
                isOpen={modalPrivacidadAbierto}
                onClose={() => setModalPrivacidadAbierto(false)}
            />

        </div>
    );
};

export default Login;