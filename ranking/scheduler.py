import logging
from datetime import datetime, time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import RANKING_REFRESH_MINUTES, RANKING_TOP_N
from core.akshare_client import AkShareClient
from core.database import AsyncSession, FundFlowSnapshot
from ranking.fund_ranking import FundRanking
from ranking.trend_ranking import TrendRanking

logger = logging.getLogger(__name__)

inflow_ranking: list[dict] = []
surge_ranking: list[dict] = []
rising_trend: list[dict] = []
falling_trend: list[dict] = []
last_update: dict = {"value": None}


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


async def refresh_rankings(force: bool = False):
    global inflow_ranking, surge_ranking, rising_trend, falling_trend, last_update
    if not force and not is_trading_time():
        return

    try:
        client = AkShareClient()
        ranking = FundRanking(client)
        inflow, _ = ranking.refresh(RANKING_TOP_N)
        inflow_ranking.clear()
        inflow_ranking.extend(inflow)

        surge = ranking.refresh_surge(RANKING_TOP_N)
        surge_ranking.clear()
        surge_ranking.extend(surge)

        # 趋势排行
        trend = TrendRanking(client)
        rising, falling = trend.refresh(RANKING_TOP_N)
        rising_trend.clear()
        rising_trend.extend(rising)
        falling_trend.clear()
        falling_trend.extend(falling)

        last_update["value"] = datetime.now()
        logger.info("Rankings refreshed: %d inflow, %d surge, %d rising, %d falling",
                    len(inflow), len(surge), len(rising), len(falling))

        await _persist_snapshots(inflow)
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
