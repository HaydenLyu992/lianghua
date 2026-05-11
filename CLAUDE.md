# CLAUDE.md — 良华

> A 股多因子分析系统。6 维因子研判涨跌 + 资金流向排行榜 + 在线回测。**不做自动交易。**
>
> 详细设计见 [`docs/DESIGN.md`](docs/DESIGN.md)。

## 命令

```bash
# 开发
pip install -r requirements.txt
python run.py                        # uvicorn app.main:app --reload，端口 8000

# Docker
docker-compose up -d                 # 一键部署（构建 app 镜像 + 启动 PG + app）
docker-compose up -d --build         # 代码更新后重新构建
docker-compose down                  # 停止
docker-compose logs -f app           # 查看应用日志

# 数据库备份
docker exec postgres pg_dump -U postgres lianghua > backup.sql
```

## 技术栈

Python 3.11+ / FastAPI / Jinja2 / HTMX / Chart.js · 数据：AkShare（主力，免费实时）+ Tushare（辅助） · 存储：PostgreSQL 16.1（已有容器 `postgres`，DB `lianghua`） · ORM：SQLAlchemy 2.0 async + asyncpg · 调度：APScheduler · 回测：Backtrader · 技术指标：TA-Lib · LLM：DeepSeek（强制启用，启动时校验 Key）

## 架构

```
app/           Web 层：FastAPI 路由 + Jinja2 模板 + static/ 静态文件
core/          数据采集层：AkShare / Tushare 统一封装，上层不直接调 API
factors/       6 个因子模块，各自独立，统一继承 FactorBase → 返回 FactorResult
analysis/      综合分析引擎：StockAnalyzer 聚合加权 + limit_analyzer 涨跌停 + llm_analyzer
backtest/      Backtrader 封装 + 内置示例策略
ranking/       资金流向排行榜 + APScheduler 定时刷新
```

## 数据库

连接串从环境变量 `DATABASE_URL` 读取，开发默认值：
```
postgresql+asyncpg://postgres:lvhang123@localhost:5432/lianghua
```

PG 容器 `postgres`（docker-compose 内服务名 `pg`），所有 DB 操作异步，4 张表：`stock_info`、`analysis_history`、`fund_flow_snapshot`、`news_cache`。

## 编码约定

- **因子模块**：继承 `factors/base.py` 的 `FactorBase`，返回 `FactorResult(score=0-100, detail={...})`
- **股票代码**：自动识别 `000001` / `sh.000001` / `平安银行` 三种格式
- **异常兜底**：所有外部数据获取必须 try/except，接口挂了返回缓存或空值，不允许前端崩
- **前端**：Jinja2 模板 + HTMX 局部刷新，不写 SPA。Chart.js / HTMX 通过 CDN 引入
- **静态文件**：FastAPI `StaticFiles` 挂载 `app/static/`，无需单独部署
- **数据库操作**：全部用 SQLAlchemy async session，禁止同步调用
- **不写注释解释 WHAT**：函数名和变量名已经说明 WHAT，只在 WHY 非显而易见时写一行注释
- **不引入新依赖**：除非必要，优先用标准库或已有依赖

## 安全红线

- **禁止自动下单**：本项目只做数据分析和趋势研判，不得接入任何券商交易接口
- 数据库密码、API Token 走环境变量，不硬编码
- `.env` 和备份文件不提交 git

## 6 因子 → AkShare 接口速查

| # | 因子 | 关键接口 |
|---|------|---------|
| 1 | 基本面 | `stock_financial_abstract`, `stock_notice_report`, `stock_restricted_release_queue`, `stock_jgdy` |
| 2 | 行业 | `stock_board_industry_cons`, `stock_board_industry_index_ths` |
| 3 | 宏观 | `macro_china_pmi`, `macro_china_cpi`, `macro_china_money_supply`, `macro_china_lpr` |
| 4 | 资金 | `stock_hsgt_hist`, `stock_margin_detail`, `stock_fund_flow_individual`, `stock_lhb_detail` |
| 5 | 情绪 | `stock_zt_pool`, `stock_hot_rank` + TA-Lib |
| 6 | 地缘 | `news_economic_baidu` + LLM 过滤分类 |

## 已知陷阱

- **AkShare 不稳定**：它是爬虫封装，极端行情时限流/挂掉是常态，必须做重试 + 本地缓存兜底
- **TA-Lib 安装**：macOS 需 `brew install ta-lib`；Dockerfile 内需 `apt-get install libta-lib0`
- **交易时段判断**：定时任务仅在 A 股交易时段（9:30-15:00，工作日）高频刷新，其他时间低频或停跑
- **LLM 强制**：未配置 `LLM_API_KEY` 启动即报错，不做关键词退化
