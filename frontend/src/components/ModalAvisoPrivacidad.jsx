import React, { useEffect } from 'react';
import './ModalAvisoPrivacidad.css';

const ModalAvisoPrivacidad = ({ isOpen, onClose }) => {
    // Cerrar modal con tecla ESC
    useEffect(() => {
        const handleEsc = (event) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleEsc);
            // Prevenir scroll del body cuando el modal está abierto
            document.body.style.overflow = 'hidden';
        }

        return () => {
            document.removeEventListener('keydown', handleEsc);
            document.body.style.overflow = 'unset';
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div 
            className="modal-privacidad-overlay" 
            onClick={onClose}
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-privacidad-titulo"
        >
            <div 
                className="modal-privacidad-container" 
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="modal-privacidad-header">
                    <h2 id="modal-privacidad-titulo">Aviso de Privacidad</h2>
                    <button 
                        className="modal-privacidad-btn-cerrar"
                        onClick={onClose}
                        aria-label="Cerrar modal"
                    >
                        ✕
                    </button>
                </div>

                {/* Body */}
                <div className="modal-privacidad-body">
                    <section className="modal-privacidad-seccion">
                        <h3>Identificación del Responsable</h3>
                        <p>
                            <strong>AWODA</strong> es un proyecto desarrollado en la <strong>Escuela Superior de Cómputo (ESCOM) </strong> 
                            del <strong>Instituto Politécnico Nacional (IPN)</strong>, ubicado en Av. Juan de Dios Bátiz s/n, 
                            Unidad Profesional Adolfo López Mateos, Gustavo A. Madero, Ciudad de México, C.P. 07738.
                        </p>
                        <p>
                            Para cualquier consulta relacionada con el tratamiento de sus datos personales, puede contactarnos 
                            a través del correo electrónico: 
                            <p><strong>aperezg1707@alumno.ipn.mx</strong> </p>
                            <p><strong>cmedinaa1700@alumno.ipn.mx</strong> </p>
                            <p><strong>obrionesr1700@alumno.ipn.mx</strong> </p>
                        </p>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Finalidades del Tratamiento de Datos Personales</h3>
                        <p>
                            Los datos personales que recabamos de usted serán utilizados para las siguientes finalidades 
                            que son necesarias para el servicio que solicita:
                        </p>
                        <ul>
                            <li>Identificación y autenticación de usuarios del sistema AWODA</li>
                            <li>Control de acceso a la plataforma y sus funcionalidades</li>
                            <li>Registro de actividades dentro del sistema para fines de auditoría y seguridad</li>
                            <li>Generación de recomendaciones sobre distribución de agua potable basadas en análisis de datos</li>
                            <li>Comunicación de avisos importantes relacionados con el funcionamiento del sistema</li>
                            <li>Cumplimiento de obligaciones legales y normativas aplicables</li>
                        </ul>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Datos Personales Recabados</h3>
                        <p>
                            Para llevar a cabo las finalidades descritas en el presente aviso de privacidad, 
                            utilizaremos los siguientes datos personales:
                        </p>
                        <ul>
                            <li>Número de empleado</li>
                            <li>Nombre completo</li>
                            <li>Correo electrónico institucional</li>
                            <li>Contraseña de acceso (almacenada de forma encriptada)</li>
                            <li>Rol o perfil dentro del sistema (administrador, usuario operativo, etc.)</li>
                            <li>Registro de accesos y actividades dentro de la plataforma</li>
                        </ul>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Transferencia de Datos Personales</h3>
                        <p>
                            Le informamos que sus datos personales <strong>NO</strong> serán compartidos, transferidos o divulgados 
                            a terceros, salvo en los siguientes casos:
                        </p>
                        <ul>
                            <li>Cuando sea requerido por autoridades competentes en el ejercicio de sus atribuciones legales</li>
                            <li>Cuando sea necesario para cumplir con disposiciones legales o reglamentarias aplicables</li>
                            <li>Por orden judicial o administrativa</li>
                        </ul>
                        <p>
                            En cualquier otro caso, se requerirá su consentimiento previo, expreso e informado.
                        </p>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Seguridad de los Datos Personales</h3>
                        <p>
                            AWODA ha implementado medidas de seguridad administrativas, técnicas y físicas para proteger 
                            sus datos personales contra daño, pérdida, alteración, destrucción o el uso, acceso o tratamiento 
                            no autorizado, incluyendo:
                        </p>
                        <ul>
                            <li>Encriptación de contraseñas mediante algoritmos seguros</li>
                            <li>Control de acceso basado en roles y permisos</li>
                            <li>Protección de comunicaciones mediante protocolos HTTPS</li>
                            <li>Respaldos periódicos de la información</li>
                            <li>Auditoría y registro de accesos al sistema</li>
                        </ul>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Uso de Cookies y Tecnologías de Rastreo</h3>
                        <p>
                            Le informamos que en nuestra página web utilizamos cookies, web beacons u otras tecnologías 
                            de almacenamiento y seguimiento de datos, a través de las cuales es posible monitorear su 
                            comportamiento como usuario de internet, con la finalidad de brindarle un mejor servicio y 
                            experiencia al navegar en nuestra plataforma.
                        </p>
                        <p>
                            Los datos personales que obtenemos de estas tecnologías de rastreo son: sesión de usuario, 
                            preferencias de configuración y tokens de autenticación temporal.
                        </p>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Modificaciones al Aviso de Privacidad</h3>
                        <p>
                            El presente aviso de privacidad puede sufrir modificaciones, cambios o actualizaciones derivadas 
                            de nuevos requerimientos legales; de nuestras propias necesidades por los servicios que ofrecemos; 
                            de nuestras prácticas de privacidad; o por otras causas.
                        </p>
                        <p>
                            Nos comprometemos a mantenerlo informado sobre los cambios que pueda sufrir el presente aviso 
                            de privacidad, a través de la propia plataforma AWODA.
                        </p>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Marco Legal</h3>
                        <p>
                            El presente Aviso de Privacidad se emite en cumplimiento a lo dispuesto por la 
                            <strong> Ley Federal de Protección de Datos Personales en Posesión de los Particulares </strong> 
                            y su Reglamento, así como los <strong>Lineamientos del Aviso de Privacidad</strong> emitidos por el 
                            Instituto Nacional de Transparencia, Acceso a la Información y Protección de Datos Personales (INAI).
                        </p>
                        <p>
                            Asimismo, el tratamiento de datos se realiza en concordancia con:
                        </p>
                        <ul>
                            <li>Constitución Política de los Estados Unidos Mexicanos (Artículo 4° - Derecho Humano al Agua)</li>
                            <li>Ley de Aguas Nacionales</li>
                            <li>Plan Nacional Hídrico</li>
                            <li>Normativa aplicable del Instituto Politécnico Nacional</li>
                        </ul>
                    </section>

                    <section className="modal-privacidad-seccion">
                        <h3>Consentimiento</h3>
                        <p>
                            Al utilizar la plataforma AWODA y proporcionar sus datos personales, usted acepta y consiente 
                            expresamente los términos y condiciones del presente Aviso de Privacidad.
                        </p>
                    </section>

                    <div className="modal-privacidad-fecha">
                        <p><strong>Fecha de última actualización:</strong> 15 de diciembre de 2025</p>
                    </div>
                </div>

                {/* Footer */}
                <div className="modal-privacidad-footer">
                    <button 
                        className="modal-privacidad-btn-aceptar"
                        onClick={onClose}
                    >
                        Aceptar
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ModalAvisoPrivacidad;
