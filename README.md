# A股主板多风格半自动量化系统 V1

这是一个本地收盘后量化信号系统，默认扫描沪深 A股主板，并保留人工白名单作为备用。系统只生成 Markdown 与 CSV 交易计划，不连接证券账户、不保存账号密码、不自动下单。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python -m src.main daily-signal
```

运行后会生成：

- `data/reports/YYYY-MM-DD_daily_signal.md`
- `data/reports/YYYY-MM-DD_daily_signal.csv`

## 报告怎么看

报告每天回答四个问题：

- 市场状态是否允许交易：绿色可交易、黄色轻仓观察、红色空仓观望。
- 当前股票池里哪些股票可以买。
- 每只候选股的触发价、建议金额、止损价、止盈减仓价。
- 哪些股票因为追高、放量异常、趋势破坏或风险标签禁止买入。
- 候选股是否满足 A股 100股一手的资金约束；买不起一手的股票会降级到观察名单。
- 候选股所属风格、风格状态、账户单笔风险、分层止损动作是否清晰。

## 股票池模式

`config/settings.yaml` 中的 `stock_pool_mode` 默认为 `main_board_all`，系统会：

- 拉取 A股实时行情列表。
- 保留沪深主板代码：`600/601/603/605/000/001/002`。
- 剔除 ST、退市风险名称。
- 预过滤成交额不足和一手金额超过账户资金的股票。
- 最多扫描 `max_scan_symbols` 只股票，避免运行过慢。

如需回到手工白名单，把 `stock_pool_mode` 改成 `manual`。

## 持仓文件

`data/holdings.csv` 用来记录已有持仓，字段为：

```text
symbol,name,quantity,cost_price,buy_date,stop_loss_price,take_profit_price,note
```

文件不存在时系统会自动创建空文件。

## 风险声明

本系统仅提供量化研究信号，不构成投资建议或收益承诺。系统不会替用户登录证券账户、提交委托、撤单或自动交易。所有实盘操作必须由用户本人确认并承担结果。

## V2 Candidate Gate

正式买入候选必须同时满足以下准入条件：

- `portfolio_mode != cash`
- `market_regime != cash`
- `timing_decision == BUY`
- `final_action == BUY`
- `candidate_data_source == fresh`
- `candidate_latest_date == expected_trade_date`
- `is_expected_trade_date == true`
- `volume > 0` 且 `amount > 0`
- `account_risk_pass == true`
- `suggested_lots >= 1`

任一条件不满足，股票只会进入观察名单或规避名单，并保留 Candidate Gate 的失败原因和已检查字段，便于复核。

本系统不会自动下单，不连接证券账户；正式候选只是量化研究信号，不构成投资建议或收益承诺。

## V3 PR2.1 规则冻结

V3 PR2.1 完成后进入30个交易日观察期。观察期内冻结V3综合评分公式与权重、BUY/WATCH阈值、Timing分类、Market Permission和Risk模型，仅允许修复程序错误、数据源稳定性问题和增强报告审计。评分分布、风险组件、Timing reason codes与Provider并发指标均为诊断字段，不参与score、ranking、action或Candidate Gate。

## V3 简化决策引擎

V3 是与 V2 并行的新增开仓评分引擎，不删除或改写 V2：

```bash
python -m src.main daily-signal --engine v2
python -m src.main daily-signal --engine v3
python -m src.main daily-signal --engine v3 --debug
```

## V3.1 自选股诊断

V3.1 与V3交易引擎、现有持仓管理并列运行，只生成 `diagnosis_action` 供人工研究参考，不修改候选、条件单或 `holding_action`。

```powershell
python -m src.main watchlist-diagnosis
python -m src.main watchlist-diagnosis --debug
python -m src.main watchlist-diagnosis --symbol 002241
```

输入按 `data/holdings.csv > data/watchlist.yaml > --symbol` 合并。诊断报告写入 `data/reports/watchlist/`，历史变化原子保存到 `data/logs/watchlist_history.json`。本模块不会自动下单，诊断结果不构成投资建议或收益承诺。

V3 使用个股因子、Timing、市场分和新增开仓风险扣分进行综合排序。`extreme_risk` 与
`breakdown` 只限制新增开仓，不会改变已有持仓动作、止损价或触发卖出。

V3 使用独立的市场权限 `BLOCKED/LIMITED/DEFENSIVE/BALANCED/ATTACK`。权限升级需连续
2个不同交易日确认，风险降级立即生效；V2 `portfolio_mode` 仅用于V2，不拦截V3。
行业数据缺失超过50%时，系统关闭行业覆盖加分。该降级只影响分析池排序解释，不影响
Timing、风险或交易资格。

每日 V2/V3 对比记录保存在 `data/logs/daily_decision/YYYY-MM-DD.json`。V3 规则版本
`v3.1.0` 在PR1验收后冻结，用于连续30个交易日的数据收集；观察期内不再新增交易规则。
