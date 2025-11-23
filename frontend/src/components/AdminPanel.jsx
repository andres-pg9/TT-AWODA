import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './AdminPanel.css';

const AdminPanel = () => {
    const navigate = useNavigate();
    const [usuarios, setUsuarios] = useState([]);
    const [cargando, setCargando] = useState(true);
    const [error, setError] = useState('');
    const [mostrarFormulario, setMostrarFormulario] = useState(false);
    const [modoEdicion, setModoEdicion] = useState(false);
    const [mensaje, setMensaje] = useState('');
    const [modalEliminar, setModalEliminar] = useState({ abierto: false, usuario: null });
    
    const [formularioUsuario, setFormularioUsuario] = useState({
        numero_empleado: '',
        password: '',
        nombre_empleado: '',
        rol_usuario: 'trabajador'
    });

    useEffect(() => {
        validarAdmin();
        cargarUsuarios();
    }, []);

    const validarAdmin = () => {
        const usuario = JSON.parse(localStorage.getItem('usuario'));
        if (!usuario || usuario.rol_usuario !== 'administrador') {
            navigate('/dashboard');
        }
    };

    const cargarUsuarios = async () => {
        try {
            setCargando(true);
            const token = localStorage.getItem('token');
            
            const response = await fetch('http://127.0.0.1:8000/api/usuarios/', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                throw new Error('Error al cargar usuarios');
            }

            const data = await response.json();
            setUsuarios(data);
            setError('');
        } catch (err) {
            setError('Error al cargar la lista de usuarios');
            console.error('Error:', err);
        } finally {
            setCargando(false);
        }
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormularioUsuario(prev => ({
            ...prev,
            [name]: name === 'numero_empleado' ? parseInt(value) || '' : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMensaje('');
        setError('');

        try {
            const token = localStorage.getItem('token');
            
            const response = await fetch('http://127.0.0.1:8000/api/usuarios/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formularioUsuario)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error al crear usuario');
            }

            setMensaje('Usuario creado exitosamente');
            setFormularioUsuario({
                numero_empleado: '',
                password: '',
                nombre_empleado: '',
                rol_usuario: 'trabajador'
            });
            setMostrarFormulario(false);
            cargarUsuarios();
            
            // Limpiar mensaje despues de 5 segundos
            setTimeout(() => {
                setMensaje('');
            }, 5000);
        } catch (err) {
            setError(err.message);
        }
    };

    const abrirModalEliminar = (numeroEmpleado, nombreEmpleado) => {
        setModalEliminar({
            abierto: true,
            usuario: { numeroEmpleado, nombreEmpleado }
        });
    };

    const cerrarModalEliminar = () => {
        setModalEliminar({ abierto: false, usuario: null });
    };

    const confirmarEliminar = async () => {
        const { numeroEmpleado, nombreEmpleado } = modalEliminar.usuario;

        try {
            const token = localStorage.getItem('token');
            
            const response = await fetch(`http://127.0.0.1:8000/api/usuarios/${numeroEmpleado}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error al eliminar usuario');
            }

            setMensaje(`Usuario ${nombreEmpleado} eliminado exitosamente`);
            cargarUsuarios();
            cerrarModalEliminar();
            
            // Limpiar mensaje despues de 5 segundos
            setTimeout(() => {
                setMensaje('');
            }, 5000);
        } catch (err) {
            setError(err.message);
            cerrarModalEliminar();
            
            // Limpiar error despues de 5 segundos
            setTimeout(() => {
                setError('');
            }, 5000);
        }
    };

    return (
        <div className="admin-panel-container">
            <div className="awoda-map-header" style={{ textAlign: "center", marginBottom: "20px" }}>
          <h2 style={{ fontWeight: "700", color: "#2c3e50" }}>Gestion de Usuarios</h2>
        </div>

            {mensaje && (
                <div className="mensaje-exito">
                    {mensaje}
                </div>
            )}

            {error && (
                <div className="mensaje-error">
                    {error}
                </div>
            )}

            <div className="admin-panel-acciones">
                <button 
                    onClick={() => setMostrarFormulario(!mostrarFormulario)}
                    className="btn-nuevo-usuario"
                >
                    {mostrarFormulario ? 'Cancelar' : 'Nuevo Usuario'}
                </button>
            </div>

            {mostrarFormulario && (
                <div className="formulario-usuario">
                    <h2>Crear Nuevo Usuario</h2>
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="numero_empleado">Numero de Empleado:</label>
                            <input
                                type="number"
                                id="numero_empleado"
                                name="numero_empleado"
                                value={formularioUsuario.numero_empleado}
                                onChange={handleInputChange}
                                required
                                min="1"
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="nombre_empleado">Nombre Completo:</label>
                            <input
                                type="text"
                                id="nombre_empleado"
                                name="nombre_empleado"
                                value={formularioUsuario.nombre_empleado}
                                onChange={handleInputChange}
                                required
                                minLength="3"
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="password">Contraseña:</label>
                            <input
                                type="password"
                                id="password"
                                name="password"
                                value={formularioUsuario.password}
                                onChange={handleInputChange}
                                required
                                minLength="6"
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="rol_usuario">Rol:</label>
                            <select
                                id="rol_usuario"
                                name="rol_usuario"
                                value={formularioUsuario.rol_usuario}
                                onChange={handleInputChange}
                                required
                            >
                                <option value="trabajador">Trabajador</option>
                                <option value="administrador">Administrador</option>
                            </select>
                        </div>

                        <div className="form-acciones">
                            <button type="submit" className="btn-guardar">
                                Crear Usuario
                            </button>
                            <button 
                                type="button" 
                                onClick={() => setMostrarFormulario(false)}
                                className="btn-cancelar"
                            >
                                Cancelar
                            </button>
                        </div>
                    </form>
                </div>
            )}

            <div className="tabla-usuarios">
                <h2>Lista de Usuarios</h2>
                {cargando ? (
                    <p>Cargando usuarios...</p>
                ) : usuarios.length === 0 ? (
                    <p>No hay usuarios registrados</p>
                ) : (
                    <table>
                        <thead>
                            <tr>
                                <th>Numero Empleado</th>
                                <th>Nombre</th>
                                <th>Rol</th>
                                <th>Acciones</th>
                            </tr>
                        </thead>
                        <tbody>
                            {usuarios.map(usuario => (
                                <tr key={usuario.id}>
                                    <td>{usuario.numero_empleado}</td>
                                    <td>{usuario.nombre_empleado}</td>
                                    <td>
                                        <span className={`badge badge-${usuario.rol_usuario}`}>
                                            {usuario.rol_usuario}
                                        </span>
                                    </td>
                                    <td>
                                        <button
                                            onClick={() => abrirModalEliminar(
                                                usuario.numero_empleado,
                                                usuario.nombre_empleado
                                            )}
                                            className="btn-eliminar"
                                        >
                                            Eliminar
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Modal de confirmacion de eliminacion */}
            {modalEliminar.abierto && (
                <div className="modal-overlay-eliminar" onClick={cerrarModalEliminar}>
                    <div className="modal-container-eliminar" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header-eliminar">
                            <h3>Confirmar Eliminación</h3>
                        </div>
                        <div className="modal-body-eliminar">
                            <p>¿Estás seguro de eliminar al usuario <strong>{modalEliminar.usuario?.nombreEmpleado}</strong>?</p>
                            <p className="modal-advertencia">Esta acción no se puede deshacer.</p>
                        </div>
                        <div className="modal-footer-eliminar">
                            <button 
                                onClick={cerrarModalEliminar}
                                className="btn-modal-cancelar"
                            >
                                Cancelar
                            </button>
                            <button 
                                onClick={confirmarEliminar}
                                className="btn-modal-eliminar"
                            >
                                Eliminar
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AdminPanel;
