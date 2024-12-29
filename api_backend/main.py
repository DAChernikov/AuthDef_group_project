import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from api.v1.api_route import router as model_router

app = FastAPI(
    title="api_backend",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )

@app.get("/")
async def root():
    return [{"status": "App ready"}]

app.include_router(model_router, prefix="/api/v1/models")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)