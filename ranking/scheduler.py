import logging
from datetime import datetime, time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import RANKING_REFRESH_MINUTES, RANKING_TOP_N
from core.akshare_client import AkShareClient
from core.database import AsyncSession, FundFlowSnapshot
from ranking.fund_ranking import FundRanking

logger = logging.getLogger(__name__)

inflow_ranking: list[dict] = []
outflow_ranking: list[dict] = []
northbound_ranking: list[dict] = []
last_update: datetime | None = None


def is_trading_time() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return time(9, 0) <= t <= time(15, 5)


async def _persist_snapshots(rows: list[dict]):
    if not rows:
        return
    try:
        async with AsyncSession() as session:
            now = datetime.now()
            for r in rows:
                stmt = pg_insert(FundFlowSnapshot).values(
                    stock_code=r.get("code", ""),
                    stock_name=r.get("name", ""),
                    main_net_inflow=r.get("main_net", 0),
                    super_large_net=r.get("super_large_net"),
                    large_net=r.get("large_net"),
                    medium_net=r.get("medium_net"),
                    small_net=r.get("small_net"),
                    snapshot_time=now,
                )
                stmt = stmt.on_conflict_do_nothing()
                await session.execute(stmt)
            await session.commit()
    except Exception as e:
        logger.warning("Failed to persist fund flow snapshots: %s", e)


async def refresh_rankings():
    global inflow_ranking, outflow_ranking, northbound_ranking, last_update
    if not is_trading_time():
        return

    try:
        faker = AkShareClient()
        ranking = FundRanking(faker)
        inflow, outflow = ranking.refresh(RANKING_TOP_N)
        inflow_ranking = inflow
        outflow_ranking = outflow

        nb = ranking.refresh_northbound(20)
        northbound_ranking = nb

        last_update = datetime.now()
        logger.info("Rankings refreshed: %d inflow, %d outflow, %d northbound", len(inflow), len(outflow), len(nb))

        await _persist_snapshots(inflow)
        await _persist_snapshots(outflow)
    except Exception as e:
        logger.error("Ranking refresh failed: %s", e)


scheduler = AsyncIOScheduler()


def start_scheduler():
    scheduler.add_job(
        refresh_rankings,
        CronTrigger(minute=f"*/{RANKING_REFRESH_MINUTES}"),
        id="fund_ranking",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Ranking scheduler started, interval=%dmin", RANKING_REFRESH_MINUTES)
