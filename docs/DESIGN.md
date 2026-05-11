# 良华量化 — 详细设计方案

> 版本: v1.0 | 日期: 2026-05-11

---

## 一、系统目标

用户输入一支 A 股股票代码，系统自动从 6 个维度聚合数据并给出综合分析报告：
- 涨跌趋势研判（多/空/中性 + 评分）
- 涨跌停概率估算
- 关键风险与利好事件列表
- 同时在另一个页面展示全市场资金流向排行榜
- 提供在线回测工具验证策略

---

## 二、技术选型及理由

| 组件 | 选择 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | 异步支持好、自动生成 API 文档、性能高 |
| 模板 | Jinja2 | FastAPI 内置支持，前后端一体无需 SPA |
| 数据库 | PostgreSQL 16 (Docker) | 复用已有容器 `postgres`，数据库名 `lianghua` |
| 定时任务 | APScheduler | 进程内调度，不需要 Celery/Redis 额外服务 |
| 数据源 | AkShare + Tushare | AkShare 免费实时、接口丰富；Tushare 历史数据质量高 |
| 回测 | Backtrader | Python 生态最成熟的回测框架 |
| 技术指标 | TA-Lib | C 底层计算快，覆盖所有常见指标 |
| LLM | DeepSeek (deepseek-chat) | OpenAI 兼容协议，强制启用。启动时校验 API Key，无 Key 直接报错 |
| 前端交互 | HTMX | 局部刷新，避免写重型 JS 框架 |

---

## 三、目录结构

```
lianghua/
├── README.md                     # 项目说明
├── CLAUDE.md                     # AI 协作指南
├── requirements.txt              # 依赖清单
├── Dockerfile                    # 应用镜像构建
├── docker-compose.yml            # 一键部署（app + PG）
├── .env.example                  # 环境变量模板
├── run.py                        # 开发启动入口
├── config.py                     # 全局配置
│
├── data/                         # 本地缓存文件
│
├── logs/                         # 日志目录
│
├── app/                          # ── Web 应用层 ──
│   ├── __init__.py
│   ├── main.py                   # FastAPI app 实例 + 中间件 + 生命周期
│   ├── routes.py                 # 所有路由定义
│   ├── templates/
│   │   ├── base.html             # 基础布局（导航栏 + 页脚）
│   │   ├── index.html            # 首页：输入框 + 快速导航
│   │   ├── analysis.html         # 个股分析报告（6 维详情 + 雷达图）
│   │   ├── ranking.html          # 资金流向排行榜
│   │   └── backtest.html         # 回测页面
│   └── static/
│       ├── css/style.css
│       └── js/charts.js          # Chart.js 画雷达图/K 线图
│
├── core/                         # ── 数据采集层 ──
│   ├── __init__.py
│   ├── akshare_client.py         # AkShare 统一封装，所有 ak.xxx() 调用归口于此
│   ├── tushare_client.py         # Tushare 封装（可选启用）
│   └── news_fetcher.py           # 新闻采集聚合（财联社、东方财富、百度经济）
│
├── factors/                      # ── 六因子模块 ──
│   ├── __init__.py
│   ├── base.py                   # 因子基类 FactorBase + FactorResult
│   ├── fundamental.py            # 因子1: 基本面
│   ├── industry.py               # 因子2: 行业/产业链
│   ├── macro.py                  # 因子3: 宏观/政策
│   ├── fund_flow.py              # 因子4: 资金流向
│   ├── sentiment.py              # 因子5: 情绪/技术面
│   └── geo_external.py           # 因子6: 地缘/外部冲击
│
├── analysis/                     # ── 综合分析引擎 ──
│   ├── __init__.py
│   ├── stock_analyzer.py         # 聚合 6 因子 + 加权打分 + 生成报告
│   ├── llm_analyzer.py           # LLM 新闻情感与影响分析（强制）
│   └── limit_analyzer.py         # 涨跌停专项分析
│
├── backtest/                     # ── 回测模块 ──
│   ├── __init__.py
│   ├── engine.py                 # Backtrader 封装
│   └── strategies.py             # 示例策略（均线金叉、动量突破、均值回归）
│
├── ranking/                      # ── 排行榜模块 ──
│   ├── __init__.py
│   ├── fund_ranking.py           # 资金流入流出 Top N
│   └── scheduler.py              # APScheduler 定时刷新任务
│
└── docs/
    └── DESIGN.md                 # 本文件
```

---

## 四、数据库设计（PostgreSQL）

### 运行环境

两种模式：

**开发模式** — 本地 `python run.py`，连接已有 PG 容器：

| 项目 | 值 |
|------|-----|
| 容器名 | `postgres` |
| 版本 | PostgreSQL 16.1 |
| 端口 | `5432` |
| 用户 | `postgres` |
| 密码 | `lvhang123` |
| 数据库 | `lianghua` |
| 连接串 | `postgresql+asyncpg://postgres:lvhang123@localhost:5432/lianghua` |

**生产部署** — `docker-compose.yml` 一键启动 app + PG：

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:lvhang123@pg:5432/lianghua
      TUSHARE_TOKEN: ${TUSHARE_TOKEN:-}
      LLM_API_KEY: ${LLM_API_KEY:-}
      LLM_BASE_URL: ${LLM_BASE_URL:-}
    depends_on:
      pg:
        condition: service_healthy
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

  pg:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: lvhang123
      POSTGRES_DB: lianghua
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d lianghua"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libta-lib0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 表结构

```sql
-- 个股基本信息缓存
CREATE TABLE stock_info (
    code        VARCHAR(10) PRIMARY KEY,  -- '000001'
    name        VARCHAR(50),             -- '平安银行'
    market      VARCHAR(2),              -- 'sz' / 'sh'
    industry    VARCHAR(50),             -- 申万一级行业
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- 因子评分历史（每次查询存一条，用于回看）
CREATE TABLE analysis_history (
    id          SERIAL PRIMARY KEY,
    stock_code  VARCHAR(10) NOT NULL,
    score_total SMALLINT,           -- 0-100
    score_fund  SMALLINT,           -- 基本面
    score_ind   SMALLINT,           -- 行业
    score_macro SMALLINT,           -- 宏观
    score_flow  SMALLINT,           -- 资金流向
    score_sent  SMALLINT,           -- 情绪技术
    score_geo   SMALLINT,           -- 地缘
    signal      VARCHAR(10),        -- 'bullish' / 'bearish' / 'neutral'
    report_json JSONB,              -- 完整分析报告
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_analysis_code ON analysis_history(stock_code);
CREATE INDEX idx_analysis_time ON analysis_history(created_at DESC);

-- 资金流向快照（排行榜数据源）
CREATE TABLE fund_flow_snapshot (
    id              SERIAL PRIMARY KEY,
    stock_code      VARCHAR(10) NOT NULL,
    stock_name      VARCHAR(50),
    main_net_inflow  NUMERIC(16,2),    -- 主力净流入（万元）
    super_large_net  NUMERIC(16,2),    -- 超大单净流入
    large_net        NUMERIC(16,2),    -- 大单净流入
    medium_net       NUMERIC(16,2),    -- 中单净流入
    small_net        NUMERIC(16,2),    -- 小单净流入
    snapshot_time    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ff_snapshot_time ON fund_flow_snapshot(snapshot_time DESC);
CREATE INDEX idx_ff_code ON fund_flow_snapshot(stock_code);

-- 新闻缓存
CREATE TABLE news_cache (
    id           SERIAL PRIMARY KEY,
    stock_code   VARCHAR(10) NOT NULL,
    title        TEXT,
    content      TEXT,
    source       VARCHAR(50),        -- 'cls' / 'eastmoney' / 'xueqiu'
    sentiment    VARCHAR(10),        -- 'positive' / 'negative' / 'neutral'
    impact_score SMALLINT,           -- 1-5 影响级别
    pub_time     TIMESTAMP,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_news_code ON news_cache(stock_code);
CREATE INDEX idx_news_time ON news_cache(pub_time DESC);
```

---

## 五、核心数据流

### 5.1 个股分析流程（/analyze?code=000001）

```
                    用户输入 stock_code
                           │
            ┌──────────────▼──────────────┐
            │  StockAnalyzer.analyze()     │
            │  (analysis/stock_analyzer.py)│
            └──────────────┬──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │      并行调用 6 个因子模块             │
        ▼       ▼       ▼       ▼       ▼      ▼
    [基本面] [行业]  [宏观]  [资金]  [情绪]  [地缘]
     score   score   score   score   score   score
        │       │       │       │       │       │
        └───────┴───────┴───┬───┴───────┴───────┘
                            │
                    ┌───────▼────────┐
                    │  加权算法汇总    │
                    │  各因子权重可配   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  涨跌停分析      │
                    │  limit_analyzer │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  LLM 新闻分析   │
                    │  (强制)         │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  生成综合报告    │
                    │  → analysis.html│
                    └────────────────┘
```

### 5.2 排行榜数据流

```
APScheduler (每5分钟触发, 仅交易时段)
    │
    ▼
fund_ranking.refresh()
    │
    ├── 调用 ak.stock_fund_flow_individual() 取全市场
    ├── 计算 main_net_inflow = 超大单 + 大单
    ├── 排序取 Top 50 流入 + Top 50 流出
    └── 写入 fund_flow_snapshot 表
    │
    ▼
/ranking 页面查询 fund_flow_snapshot → 渲染排名
```

### 5.3 回测流程

```
用户输入: 股票代码 + 回测区间 + 初始资金 + 策略选择
    │
    ▼
engine.run(code, start, end, cash, strategy)
    │
    ├── 从 AkShare/Tushare 下载历史 K 线
    ├── 注入 Backtrader Cerebro
    ├── 运行回测
    └── 输出: 收益曲线图 + 夏普比率 + 最大回撤 + 胜率 + 年化收益
    │
    ▼
backtest.html 展示结果图表
```

---

## 六、加权评分算法

### 默认权重

| 维度 | 权重 | 说明 |
|------|------|------|
| 基本面 | 25% | A 股最核心驱动力 |
| 资金流向 | 25% | 短期走势关键指标 |
| 情绪技术 | 20% | 市场情绪 + 技术形态 |
| 宏观 | 15% | 大环境影响整体估值 |
| 行业 | 10% | 板块轮动效应 |
| 地缘 | 5% | 偶发但冲击大 |

### 评分公式

```
total_score = Σ (factor_score × weight)   // 每个因子 0-100 分

signal:
  80-100  → 强烈看多
  60-79   → 中性偏多
  40-59   → 中性
  20-39   → 中性偏空
  0-19    → 强烈看空
```

### 涨跌停概率估算

```
涨停概率 = f(连板高度排名, 封板强度, 资金净流入, 利好新闻密度, 板块热度)
跌停概率 = f(跌幅排名, 利空新闻密度, 基本面恶化信号, 板块退潮)

具体算法在 limit_analyzer.py 中实现，
采用规则引擎 + 历史回测校准阈值。
```

---

## 七、API 路由设计

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页，股票输入框 |
| `/analyze` | GET | 个股分析报告（`?code=000001`） |
| `/api/analyze/{code}` | GET | 分析报告 JSON API |
| `/ranking` | GET | 资金流向排行榜页面 |
| `/api/ranking/inflow` | GET | 净流入 TOP 50 JSON |
| `/api/ranking/outflow` | GET | 净流出 TOP 50 JSON |
| `/api/ranking/northbound` | GET | 北向资金净买入 TOP 20 JSON |
| `/backtest` | GET | 回测页面 |
| `/api/backtest/run` | POST | 执行回测，返回结果 |
| `/api/news/{code}` | GET | 个股关联新闻列表 |
| `/api/limit/{code}` | GET | 涨跌停分析结果 |

---

## 八、前端设计

### 8.1 技术方案

| 项 | 选择 | 说明 |
|-----|------|------|
| 渲染方式 | 服务端渲染 (SSR) | FastAPI + Jinja2 模板，不搞前后端分离 |
| 样式 | 手写 CSS | 不引入 Tailwind/Bootstrap，保持轻量 |
| 图表 | Chart.js | 雷达图、K 线图、收益曲线 |
| 局部刷新 | HTMX | 表单提交、分页、轮询等无需写 JS |
| 静态资源 | FastAPI StaticFiles | `app.mount("/static", StaticFiles(directory="app/static"))` |
| 部署 | 无需单独部署 | app 容器内 uvicorn 同时服务模板 + 静态文件 |

### 8.2 页面总览

```
lianghua/
├── base.html          ← 公共骨架（导航栏、页脚、全局样式）
│
├── index.html         ← 首页: 股票输入框 + 快捷入口
│
├── analysis.html      ← 个股分析报告: 雷达图 + 6维详情 + 涨跌停 + 新闻
│
├── ranking.html       ← 资金流向排行榜: 流入/流出 TOP 50 表格
│
└── backtest.html      ← 在线回测: 参数表单 + 收益曲线图

static/
├── css/style.css      ← 全局样式
└── js/charts.js       ← Chart.js 封装（雷达图、K线图、曲线图）
```

### 8.3 `base.html` — 公共布局

```
┌─────────────────────────────────────────────┐
│  良华量化          首页 │ 排行榜 │ 回测        │  ← 导航栏
├─────────────────────────────────────────────┤
│                                             │
│         {% block content %}{% endblock %}   │  ← 各页面内容区
│                                             │
├─────────────────────────────────────────────┤
│  免责声明：不构成投资建议                     │  ← 页脚
└─────────────────────────────────────────────┘
```

### 8.4 `index.html` — 首页

```
┌─────────────────────────────────────────────┐
│                                             │
│         A股多因子分析系统                     │
│    从6个维度综合分析个股涨跌趋势               │
│                                             │
│   ┌─────────────────────────┐ ┌───────────┐ │
│   │  输入股票代码或名称...    │ │   开始分析  │ │
│   └─────────────────────────┘ └───────────┘ │
│                                             │
│   快捷入口:                                  │
│   [资金流向排行榜]  [在线回测]                │
│                                             │
│   示例: 000001(平安银行)  600519(贵州茅台)    │
└─────────────────────────────────────────────┘
```

### 8.5 `analysis.html` — 个股分析报告（核心页面）

```
┌──────────────────────────────────────────────┐
│  000001 平安银行         综合评分: 72/100      │
│  ████████████░░░░░░░░  中性偏多               │
├──────────────┬──────────┬───────────────────┤
│              │ 基本面   │ 近期关键事件        │
│   雷达图      │ ████░ 65│ • 昨日北向净买入    │
│   (6维)      │ 行业     │ • 央行降准预期      │
│              │ ███░░ 55│ • 板块轮动加速      │
│              │ 宏观     │                    │
│              │ ██░░░ 40│                    │
│              │ 资金流向 │                    │
│              │ █████ 82│                    │
│              │ 情绪技术 │                    │
│              │ ███░░ 48│                    │
│              │ 地缘风险 │                    │
│              │ █░░░░ 15│                    │
├──────────────┴──────────┴───────────────────┤
│  涨跌停分析                                  │
│  ┌──────────────────────────────────────┐   │
│  │ 今日涨停概率: 12%  │ 跌停概率: 3%     │   │
│  │ 封板强度: 中等     │ 连板高度: 2板    │   │
│  └──────────────────────────────────────┘   │
├──────────────────────────────────────────────┤
│  资金流向详情                                │
│  ┌──────────────────────────────────────┐   │
│  │ 近5日主力净流入 +2.3亿 (柱状图)        │   │
│  │ 北向资金: +0.8亿  融资余额: 15.2亿    │   │
│  └──────────────────────────────────────┘   │
├──────────────────────────────────────────────┤
│  关联新闻 (LLM 分类)                         │
│  ┌──────────────────────────────────────┐   │
│  │ 🟢 利好  平安银行一季度净利增8%        │   │
│  │ 🔴 利空  银行业净息差收窄超预期        │   │
│  │ ⚪ 中性  央行开展1000亿MLF操作        │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

### 8.6 `ranking.html` — 资金流向排行榜

```
┌──────────────────────────────────────────────┐
│  资金流向排行榜    最后更新: 14:35:02          │
├──────────────────────────────────────────────┤
│  [主力净流入 TOP 50]  [主力净流出 TOP 50]     │
│  [北向资金 TOP 20]                            │
├──────────────────────────────────────────────┤
│  排名  │ 代码    │ 名称     │ 主力净流入(万)  │
│  1    │ 000001 │ 平安银行  │ +52,380        │
│  2    │ 600519 │ 贵州茅台  │ +41,200        │
│  ...  │ ...    │ ...      │ ...            │
│  50   │ ...    │ ...      │ ...            │
├──────────────────────────────────────────────┤
│  [上一页] [下一页]                            │
└──────────────────────────────────────────────┘
```

### 8.7 `backtest.html` — 在线回测

```
┌──────────────────────────────────────────────┐
│  在线回测                                    │
├──────────────────────────────────────────────┤
│  股票代码: [000001]  回测区间: [2023-01] → [2025-12]  │
│  初始资金: [100000]  策略: [均线金叉 ▼]      │
│  快线: [5] 日  慢线: [20] 日                 │
│  [开始回测]                                  │
├──────────────────────────────────────────────┤
│  回测结果 (运行后展示)                        │
│  ┌──────────────────────────────────────┐   │
│  │          收益曲线图                    │   │
│  │    ╱╲    ╱╲                          │   │
│  │   ╱  ╲  ╱  ╲    ╱                    │   │
│  │  ╱    ╲╱    ╲  ╱                     │   │
│  │ ╱            ╲╱                      │   │
│  └──────────────────────────────────────┘   │
│  年化收益: +18.5%  夏普比率: 1.32            │
│  最大回撤: -12.3%  胜率: 56.8%               │
│  总收益率: +62.7%  交易次数: 47              │
└──────────────────────────────────────────────┘
```

### 8.8 用户交互流程

```
[首页 index.html]
    │
    ├── 输入股票代码 → 提交 GET /analyze?code=000001
    │       │
    │       └── → 后端 StockAnalyzer.analyze("000001")
    │               ├── 并行调 6 个因子模块
    │               ├── 涨跌停分析 (limit_analyzer)
    │               ├── LLM 新闻分析 (强制)
    │               └── 渲染 analysis.html 返回
    │
    ├── 点击「排行榜」→ GET /ranking
    │       └── → 查询 fund_flow_snapshot 表 → 渲染 ranking.html
    │
    └── 点击「回测」→ GET /backtest
            │
            └── 填参数 → POST /api/backtest/run
                    └── → Backtrader 引擎运行 → 返回图表数据
```

### 8.9 静态资源说明

**不需要单独部署。** FastAPI 通过 `StaticFiles` 中间件直接挂载 `app/static/` 目录：

```python
# app/main.py
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="app/static"), name="static")
```

- `style.css` — 全局样式，所有页面共用，由 `base.html` 引入
- `charts.js` — Chart.js 封装，仅在需要图表的页面（analysis.html、backtest.html）引入
- Chart.js 本身通过 CDN 引入，不打包本地（`<script src="https://cdn.jsdelivr.net/npm/chart.js@4">`）
- HTMX 同样 CDN 引入（`<script src="https://unpkg.com/htmx.org@1.9">`）

Docker 容器内 uvicorn 同时服务模板渲染 + 静态文件，一个容器搞定全部。

---

## 九、实施阶段

### Phase 1: 基础框架 + 数据层（最先做）

**目标**: 项目能跑起来，首页能打开，AkShare 能拉数据

- [ ] 创建目录结构
- [ ] `requirements.txt` 写入依赖
- [ ] `Dockerfile` + `docker-compose.yml` 部署配置
- [ ] `.env.example` 环境变量模板
- [ ] `config.py` 配置模块（含 DB 连接）
- [ ] `core/akshare_client.py` AkShare 核心接口封装
- [ ] `core/tushare_client.py` Tushare 封装（可选）
- [ ] `core/news_fetcher.py` 新闻采集
- [ ] `app/main.py` FastAPI app 骨架 + 数据库生命周期
- [ ] `app/routes.py` 基础路由 `/` → index.html
- [ ] `app/templates/base.html` 基础模板
- [ ] `app/templates/index.html` 首页

**验证**: `python run.py` → 浏览器访问 `http://localhost:8000` 看到首页

### Phase 2: 六因子 + 分析引擎

**目标**: 输入股票代码能看到完整 6 维分析报告

- [ ] `factors/base.py` 因子基类 `FactorBase` + `FactorResult`
- [ ] `factors/fundamental.py` 基本面因子
- [ ] `factors/industry.py` 行业因子
- [ ] `factors/macro.py` 宏观因子
- [ ] `factors/fund_flow.py` 资金流向因子
- [ ] `factors/sentiment.py` 情绪技术因子
- [ ] `factors/geo_external.py` 地缘因子
- [ ] `analysis/stock_analyzer.py` 综合聚合引擎
- [ ] `analysis/limit_analyzer.py` 涨跌停分析
- [ ] `app/templates/analysis.html` 分析报告页
- [ ] `app/static/js/charts.js` 雷达图

**验证**: 输入 `000001` → 看到完整分析报告 + 评分

### Phase 3: 排行榜 + 回测

**目标**: 排行榜实时展示，回测可跑出结果

- [ ] `ranking/fund_ranking.py` 资金流向排名逻辑
- [ ] `ranking/scheduler.py` APScheduler 定时刷新
- [ ] `app/templates/ranking.html` 排行榜页面
- [ ] `backtest/engine.py` Backtrader 封装
- [ ] `backtest/strategies.py` 示例策略
- [ ] `app/templates/backtest.html` 回测页面

**验证**: 排行榜有数据，回测跑完出收益曲线

### Phase 4: LLM + 打磨

**目标**: 新闻智能分析，界面完善

- [ ] `analysis/llm_analyzer.py` LLM 接入
- [ ] 前端图表优化（K 线图、资金流图）
- [ ] 错误兜底与缓存策略完善
- [ ] Docker 镜像构建 + docker-compose 一键部署测试
- [ ] 整体测试

---

## 十、关键依赖

```
# requirements.txt 核心依赖
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
jinja2>=3.1.0
akshare>=1.14.0
tushare>=1.4.0
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29.0
psycopg2-binary>=2.9       # 备用同步驱动
apscheduler>=3.10.0
backtrader>=1.9.0
ta-lib>=0.4.0
pandas>=2.2.0
numpy>=1.26.0
httpx>=0.27.0           # AkShare 底层用
openai>=1.0.0           # LLM 调用（DeepSeek，强制）
```

## 十一、注意事项

1. **AkShare 是爬虫封装**，依赖东方财富等网站的接口稳定性。极端行情时官网可能限流，需做重试 + 缓存。
2. **交易时段判断**：A 股交易日 9:30-15:00，定时任务只在此区间高频刷新，其他时间低频或停跑。
3. **Tushare Token**：需用户自行注册获取，配置在 `config.py` 中，不硬编码。
4. **LLM 强制**：必须配置 `LLM_API_KEY`，启动时校验，无 Key 直接报错退出。使用 DeepSeek（`deepseek-chat`，OpenAI 兼容协议）。
5. **不做交易下单**：本项目所有功能均为只读分析，不涉及任何券商接口或订单执行。

---

## 十二、Docker 部署

### 目标环境要求

目标机器仅需安装 **Docker + Docker Compose**，无需 Python、PostgreSQL、TA-Lib 等任何额外依赖。

### 部署步骤

```bash
# 1. 将项目拷贝到目标机器
scp -r lianghua/ user@目标IP:/opt/lianghua

# 2. 在目标机器上
cd /opt/lianghua

# 3. 一键构建并启动所有服务
docker-compose up -d

# 4. 验证
docker-compose ps
curl http://localhost:8000
```

### 服务组成

| 服务 | 镜像 | 说明 |
|------|------|------|
| `app` | 本地 Dockerfile 构建 | FastAPI 应用，端口 8000 |
| `pg` | `postgres:16` | PostgreSQL 数据库，端口 5432 |

- `app` 通过 `depends_on` + `pg` 的 `healthcheck` 确保数据库就绪后才启动
- 两个服务通过 compose 内部网络通信（`pg:5432`），PG 端口可不对外暴露

### 环境变量

通过 `.env` 文件或 `docker-compose.yml` 中 `environment` 注入：

| 变量 | 必填 | 说明 |
|------|------|------|
| `DATABASE_URL` | 否 | 默认指向 compose 内 pg 服务 |
| `TUSHARE_TOKEN` | 否 | Tushare 数据源，不配则只走 AkShare |
| `LLM_API_KEY` | 是 | DeepSeek API Key，启动时校验，必填 |
| `LLM_BASE_URL` | 否 | 默认 `https://api.deepseek.com` |
| `LLM_MODEL` | 否 | 默认 `deepseek-chat` |

### 运维命令

```bash
docker-compose logs -f app          # 实时查看应用日志
docker-compose restart app          # 重启应用
docker-compose up -d --build        # 代码更新后重新构建
docker-compose down                 # 停止所有服务
docker-compose down -v              # 停止并删除数据卷（⚠️ 会清空数据库）

# 数据备份
docker exec pg pg_dump -U postgres lianghua > backup.sql
# 数据恢复
docker exec -i pg psql -U postgres lianghua < backup.sql
```

### 数据持久化

- `pgdata` named volume 挂载到 PG 容器的 `/var/lib/postgresql/data`
- `./data` 和 `./logs` 目录 bind mount 到 app 容器，宿主机可直接查看
