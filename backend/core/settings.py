import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "AWODA Backend"
    VERSION = "1.0"
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

settings = Settings()
