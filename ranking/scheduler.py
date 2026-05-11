import logging
from datetime import datetime, time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import RANKING_REFRESH_MINUTES, RANKING_TOP_N
from core.akshare_client import AkShareClient
from ranking.fund_ranking import FundRanking

logger = logging.getLogger(__name__)

# 模块级共享数据，路由直接读取
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
