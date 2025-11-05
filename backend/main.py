from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import optimize, config

app = FastAPI(title="AWODA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(optimize.router, prefix="/api/optimize", tags=["Optimización"])
app.include_router(config.router, prefix="/api/config", tags=["Configuración"])

@app.get("/")
def home():
    return {"status": "Backend de AWODA activo"}
