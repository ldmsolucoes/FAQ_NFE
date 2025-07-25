import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from nfe import nfe

app = FastAPI()


# Caminho absoluto para a pasta 'nfe/imagens' a partir do arquivo main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imagens_dir = os.path.join(BASE_DIR, "imagens")
scripts_dir = os.path.join(BASE_DIR, "scripts")
styles_dir = os.path.join(BASE_DIR, "styles")


# Serve a pasta "imagens" que está dentro de "nfe"
app.mount("/imagens", StaticFiles(directory=imagens_dir), name="imagens")
app.mount("/scripts", StaticFiles(directory=scripts_dir), name="scripts")
app.mount("/styles", StaticFiles(directory=styles_dir), name="styles")


# Inclui seu router
app.include_router(nfe.router)