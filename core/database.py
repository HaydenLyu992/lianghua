import os
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:lvhang123@localhost:5432/lianghua",
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSession = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class StockInfo(Base):
    __tablename__ = "stock_info"
    code = Column(String(10), primary_key=True)
    name = Column(String(50))
    market = Column(String(2))
    industry = Column(String(50))
    updated_at = Column(DateTime, default=datetime.now)


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False)
    score_total = Column(Integer)
    score_fund = Column(Integer)
    score_ind = Column(Integer)
    score_macro = Column(Integer)
    score_flow = Column(Integer)
    score_sent = Column(Integer)
    score_geo = Column(Integer)
    signal = Column(String(10))
    report_json = Column(JSONB)
    created_at = Column(DateTime, default=datetime.now)


class FundFlowSnapshot(Base):
    __tablename__ = "fund_flow_snapshot"
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False)
    stock_name = Column(String(50))
    main_net_inflow = Column(Float)
    super_large_net = Column(Float)
    large_net = Column(Float)
    medium_net = Column(Float)
    small_net = Column(Float)
    snapshot_time = Column(DateTime, default=datetime.now)


class NewsCache(Base):
    __tablename__ = "news_cache"
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False)
    title = Column(Text)
    content = Column(Text)
    source = Column(String(50))
    sentiment = Column(String(10))
    impact_score = Column(Integer)
    pub_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
