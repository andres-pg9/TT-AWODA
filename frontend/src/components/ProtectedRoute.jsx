import React from 'react';
import { Navigate } from 'react-router-dom';

/**
 * Componente ProtectedRoute
 * Protege rutas que requieren autenticación
 * Si no hay token, redirige al login
 */

const ProtectedRoute = ({ children }) => {
    const token = localStorage.getItem('token');

    if (!token) {
        console.log('🚫 Acceso denegado - No hay token');
        return <Navigate to="/login" replace />;
    }

    return children;
};

export default ProtectedRoute;
