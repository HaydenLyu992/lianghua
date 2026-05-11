import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# --- Database ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:lvhang123@localhost:5432/lianghua",
)

# --- Data Sources ---
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
AKSHARE_CACHE_TTL = int(os.getenv("AKSHARE_CACHE_TTL", "300"))  # seconds

# --- LLM (DeepSeek，强制启用) ---
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

if not LLM_API_KEY:
    raise RuntimeError(
        "LLM_API_KEY 未设置。请在 .env 文件或环境变量中配置 DeepSeek API Key。\n"
        "示例: LLM_API_KEY=sk-your-deepseek-key"
    )

# --- Trading Calendar ---
TRADING_START = "09:30"
TRADING_END = "15:00"

# --- Ranking ---
RANKING_REFRESH_MINUTES = 5
RANKING_TOP_N = 50

# --- Factor Weights (must sum to 100) ---
FACTOR_WEIGHTS = {
    "fundamental": 25,
    "fund_flow": 25,
    "sentiment": 20,
    "macro": 15,
    "industry": 10,
    "geo_external": 5,
}
