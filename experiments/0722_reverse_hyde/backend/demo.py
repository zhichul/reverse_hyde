from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

app = FastAPI()

app.mount(
    "/annotations",
    StaticFiles(directory=str("annotations"), html=True),
    name="annotations",
)
uvicorn.run(app, host='0.0.0.0', port=8000)
