from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import BASE_DIR
from core.database import Base, engine
from ranking.scheduler import start_scheduler, scheduler

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    start_scheduler()
    yield
    scheduler.shutdown(wait=False)
    await engine.dispose()


app = FastAPI(title="良华量化", version="0.2.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")

from app.routes import router  # noqa: E402
app.include_router(router)


# ── 全局错误页 ──────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return HTMLResponse(
        templates.get_template("404.html").render(request=request),
        status_code=404,
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return HTMLResponse(
        templates.get_template("500.html").render(request=request),
        status_code=500,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=3456, reload=True)
