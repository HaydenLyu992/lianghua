import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import BASE_DIR
from core.database import Base, engine
from ranking.scheduler import start_scheduler, scheduler

logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    start_scheduler()
    # 后台预热关键缓存，避免首次请求冷启动
    asyncio.create_task(_warmup_caches())
    yield
    scheduler.shutdown(wait=False)
    await engine.dispose()


async def _warmup_caches():
    """后台预热股票列表 + 资金流向，避免首次请求冷启动等待。"""
    from core.akshare_client import AkShareClient
    client = AkShareClient()
    warmups = [
        ("stock_list", client.get_stock_list),
        ("all_fund_flow", client.get_all_fund_flow),
    ]
    for name, fn in warmups:
        try:
            df = await asyncio.to_thread(fn)
            logger.info("Warmup %s: %d rows", name, len(df))
        except Exception as e:
            logger.warning("Warmup %s failed (non-critical): %s", name, e)


app = FastAPI(title="良华分析", version="0.2.0", lifespan=lifespan)

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
