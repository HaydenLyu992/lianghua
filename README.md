# 良华 — A股多因子分析系统

对 A 股个股从 **6 个维度** 进行综合分析和涨跌趋势研判，同时提供实时资金流向排行榜和在线回测功能。

> ⚠️ 本项目仅做数据分析和趋势研判，不涉及自动交易。

## 功能概览

| 功能 | 说明 |
|------|------|
| **个股多因子分析** | 输入股票代码，从 6 个维度综合研判涨跌趋势 |
| **涨跌停专项分析** | 涨停/跌停池、封板强度、连板高度、炸板率 |
| **资金流向排行榜** | 实时主力资金净流入/流出 TOP 50 |
| **在线回测** | 自定义策略 + 历史数据回测，输出收益曲线和绩效指标 |
| **LLM 新闻分析** | DeepSeek 对个股相关新闻做利好/利空/中性分类 + 影响评级（强制启用）|

## 6 维分析体系

```
基本面 ───  财报、公告、增减持、限售解禁、机构调研
行业  ───  行业政策、上下游价格、景气度、竞争格局
宏观  ───  货币政策、财政政策、PMI/CPI/GDP、汇率
资金  ───  北向资金、融资融券、主力流向、龙虎榜
情绪  ───  涨停池、股吧舆情、技术指标（MACD/RSI/布林带）
地缘  ───  国际冲突、制裁、全球大宗商品、美股传导
```

## 技术栈

- **后端**: Python 3.11+ / FastAPI
- **数据源**: AkShare（主力）+ Tushare（辅助）
- **存储**: PostgreSQL 16（已有 Docker 容器 `postgres`，数据库 `lianghua`）
- **ORM**: SQLAlchemy 2.0 async + asyncpg
- **调度**: APScheduler
- **回测**: Backtrader
- **技术指标**: TA-Lib
- **LLM**: DeepSeek（强制启用，需配置 API Key）

## 快速开始

### 开发模式

```bash
pip install -r requirements.txt
python run.py                 # 确保已有 PG 容器在运行
# 访问 http://localhost:3456
```

### 一键部署

```bash
docker-compose up -d          # 构建并启动 app + PostgreSQL
# 访问 http://localhost:3456
```

## 项目结构

```
lianghua/
├── Dockerfile            # 应用镜像
├── docker-compose.yml    # 一键部署（app + PG）
├── app/                  # Web 应用（FastAPI + Jinja2）
├── core/                 # 数据采集层
├── factors/              # 6 大因子模块
├── analysis/             # 综合分析引擎 + LLM
├── backtest/             # 回测引擎
├── ranking/              # 资金流向排行榜
├── config.py             # 全局配置
└── run.py                # 开发启动入口
```

## 部署到生产环境

目标机器只需要 **Docker + Docker Compose**，无需安装 Python、PostgreSQL 或其他依赖。

```bash
# 1. 将项目拷贝到目标机器
scp -r lianghua/ user@目标IP:/opt/lianghua

# 2. 在目标机器上进入目录
cd /opt/lianghua

# 3. 一键构建并启动所有服务（app + PostgreSQL）
docker-compose up -d

# 4. 查看运行状态
docker-compose ps

# 5. 访问 http://目标IP:3456
```

**常用运维命令：**

```bash
docker-compose logs -f app     # 查看应用日志
docker-compose restart app     # 重启应用
docker-compose down            # 停止所有服务
docker-compose up -d --build   # 重新构建并启动（代码更新后）
```

**数据持久化：** PostgreSQL 数据存储在 Docker named volume `pgdata` 中，容器删除后数据不丢失。如需备份：

```bash
docker exec pg pg_dump -U postgres lianghua > backup.sql
```

## 免责声明

本系统仅提供数据分析参考，不构成任何投资建议。股市有风险，投资需谨慎。
