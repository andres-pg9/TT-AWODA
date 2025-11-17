# TT-AWODA

AWODA es una aplicación web desarrollada en ESCOM-IPN que utiliza IA para generar sugerencias sobre la distribución priorizada de agua potable en la CDMX, considerando datos históricos, reportes ciudadanos, población y edificaciones críticas, apoyando a las autoridades en la toma de decisiones bajo el marco legal vigente.

Se utilizó el algoritmo Particle Swarm Optimization (PSO) para sugerir la distribución óptima de agua en la CDMX, priorizando sectores esenciales como hospitales, escuelas y casas de acuerdo con la Ley de Aguas Nacionales.

---

## Tabla de Contenidos

- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [API Endpoints](#-api-endpoints)
- [Documentación](#-documentación)

---

## Características

### Backend
- **Algoritmo PSO** para optimización de distribución de agua
- **API REST** completa con FastAPI
- **Base de datos MongoDB** con persistencia de datos
- **Autenticación JWT** con usuarios y roles
- **Historial** completo de optimizaciones
- **Trazabilidad** de cambios por usuario

### Frontend
- **Mapa interactivo** con Leaflet mostrando colonias
- **Visualización** de prioridades por colores
- **Dashboard** intuitivo con 7 columnas
- **Ajuste de parámetros** actualizable
- **Sistema de login** integrado

---

## Tecnologías

### Backend
- **FastAPI** 0.120.0 - Framework web async
- **MongoDB** - Base de datos NoSQL
- **Motor** 3.3.2 - Driver async de MongoDB
- **JWT** - Autenticación con tokens
- **Bcrypt** - Hashing seguro de passwords
- **NumPy** & **Pandas** - Computación científica
- **Python** 3.11+

### Frontend
- **React** 19.1.1 - Framework de UI
- **Vite** 7.1.7 - Build tool
- **Leaflet** 1.9.4 - Mapas interactivos
- **React Router** 7.9.5 - Navegación

---

## Requisitos Previos

### Software Necesario

1. **Python 3.11 o superior**
   - Descargar: https://www.python.org/downloads/

2. **Node.js 18 o superior**
   - Descargar: https://nodejs.org/

3. **MongoDB**
   - **Opción A (Local)**: https://www.mongodb.com/try/download/community
   - **Opción B (Cloud)**: MongoDB Atlas (gratis) - https://www.mongodb.com/cloud/atlas

---

## Instalación

### 1️. Clonar el Repositorio

```bash
git clone https://github.com/andres-pg9/TT-AWODA.git
cd TT-AWODA
```

---

### 2️. Configurar Backend

```bash
# Navegar a la carpeta backend
cd backend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu configuración de MongoDB y JWT_SECRET_KEY

# Inicializar base de datos (crea usuarios de prueba y datos iniciales)
python init_database.py
```

**Credenciales de prueba creadas:**
- Admin: `215646` / `admin123`
- Trabajador: `215647` / `trabajador123`

---

### 3️. Configurar Frontend

```bash
# Navegar a la carpeta frontend (desde la raíz del proyecto)
cd ../frontend

# Instalar dependencias
npm install

# (Opcional) Configurar URL del backend
echo "VITE_API_URL=http://localhost:8000" > .env
```

---

## Uso

### Iniciar Backend

```bash
cd backend
# Asegurarse de tener el entorno virtual activado
uvicorn main:app --reload
```

El backend estará disponible en: **http://localhost:8000**

- Docs interactivos: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

### Iniciar Frontend

```bash
cd frontend
npm run dev
```

El frontend estará disponible en: **http://localhost:5173**

---

## Estructura del Proyecto

```
awoda/
├── backend/
│   ├── api/
│   │   └── routes/
│   │       ├── auth.py           # Autenticación
│   │       ├── config.py         # Configuración
│   │       └── optimize.py       # Optimización PSO
│   ├── core/
│   │   └── settings.py           # Configuración del proyecto
│   ├── database/
│   │   ├── connection.py         # Conexión a MongoDB
│   │   ├── models.py             # Modelos Pydantic
│   │   └── repositories.py       # CRUD operations
│   ├── ia/
│   │   ├── funciones.py          # Funciones de utilidad
│   │   ├── normalizacion.py     # Normalización de datos
│   │   ├── pso.py                # Algoritmo PSO
│   │   └── resultados.py         # Procesamiento de resultados
│   ├── models/
│   │   └── schemas.py            # Schemas de API
│   ├── .env                      # Variables de entorno (no subir a git)
│   ├── .env.example              # Plantilla de variables
│   ├── init_database.py          # Script de inicialización
│   ├── main.py                   # Punto de entrada
│   └── requirements.txt          # Dependencias Python
│
└── frontend/
    ├── public/
    ├── src/
    │   ├── components/           # Componentes React
    │   ├── pages/                # Páginas
    │   ├── App.jsx               # Componente principal
    │   └── main.jsx              # Punto de entrada
    ├── index.html
    ├── package.json              # Dependencias Node
    └── vite.config.js            # Configuración Vite
```

---

## API Endpoints

### Autenticación
```
POST   /api/auth/login              # Login de usuarios
GET    /api/auth/me                 # Usuario actual
GET    /api/auth/validate-token     # Validar token JWT
POST   /api/auth/logout             # Cerrar sesión
```

### Optimización
```
GET    /api/optimize                # Obtener último ranking
POST   /api/optimize                # Ejecutar optimización
GET    /api/optimize/historial      # Historial de resultados
GET    /api/optimize/{id}           # Resultado específico
```

### Configuración
```
GET    /api/config                  # Configuración del sistema
```

### Sistema
```
GET    /                            # Status del backend
GET    /health                      # Health check
```

---

## Base de Datos

### Colecciones de MongoDB

**1. usuarios**
- Login y autenticación
- Roles (administrador/trabajador)
- Passwords hasheados con bcrypt

**2. datos_colonias**
- Consumo histórico por colonia
- Número de reportes de fallas
- Timestamp de cada consulta

**3. resultados_optimizacion**
- Pesos óptimos (α, β, γ, δ)
- Utilidad total calculada
- Rankings de colonias y edificaciones
- Asociado al usuario que lo generó

---

## Testing

### Backend

```bash
cd backend

# Test de repositorios
python test_repositories.py

# Test de autenticación (requiere requests)
pip install requests
python test_auth.py
```

### Frontend

```bash
cd frontend

# Linting
npm run lint

# Build de producción
npm run build
```

---

## Seguridad

### Implementado
- Passwords hasheados con bcrypt
- JWT tokens con expiración
- CORS configurado
- Variables sensibles en .env
- Validación de datos con Pydantic

### Recomendaciones para Producción
- Usar HTTPS siempre
- Cambiar JWT_SECRET_KEY por una clave segura
- Implementar rate limiting
- Agregar refresh tokens
- Configurar MongoDB con autenticación
- Usar secrets manager en producción

---


## Licencia

Este proyecto es para fines educativos y está basado en:
- Ley de Aguas Nacionales (México)
- Artículos 9, 13, 16, 23, 41 del Reglamento

---

## Autores

- **Equipo AWODA** - *Trabajo Terminal 2025-B005*
- [Briones Rayo Oscar](https://github.com/OscarBR7)
- [Medina Ascencio Carlos Armando](https://github.com/CarlosAMedina)
- [Perez Gomez Andres](https://github.com/andres-pg9)

---

---

## 🔗 Enlaces Útiles

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Documentación de React](https://react.dev/)
- [Documentación de MongoDB](https://www.mongodb.com/docs/)
- [Leaflet](https://leafletjs.com/)
- [PSO Algorithm](https://www.geeksforgeeks.org/machine-learning/particle-swarm-optimization-pso-an-overview/)

---

**Última actualización:** Noviembre 2025