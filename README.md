# run.analyzer

🏃 跑步数据分析与训练计划生成器 —— 基于 Streamlit 的轻量 Web 应用：
上传跑步记录 CSV，自动生成里程 / 配速 / 心率趋势图与汇总统计，并调用大模型（OpenAI 兼容接口）
给出一份面向学生跑者的运动评估与下周训练建议。

## 功能

- CSV 上传与原始数据预览
- 里程、配速、平均心率三张趋势图（Plotly 交互式）
- 汇总统计：总跑步次数、累计里程、平均配速、平均心率
- AI 运动评估与训练计划生成（侧边栏填写 API Key / Base URL / 模型名称）

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

浏览器会自动打开应用页面（默认 http://localhost:8501）。

## 使用

1. 侧边栏填入 API Key（以及可选的 Base URL 和模型名称，默认为 OpenAI 兼容的 `https://api.doubao.com/v1` + `doubao-lite-4k`）。
2. 上传符合格式的 CSV，或直接使用仓库内的示例文件 `run_data.csv`。
3. 点击「生成分析与训练计划」获取 AI 评估。

## 数据格式

CSV 必须包含以下列：

| 列名 | 含义 | 示例 |
| --- | --- | --- |
| 日期 | 跑步日期 | 2026-10-01 |
| 里程(km) | 本次里程 | 3.0 |
| 时长(分钟) | 本次时长 | 21 |
| 配速(min/km) | 平均配速 | 7.0 |
| 平均心率 | 平均心率 | 132 |

示例见 `run_data.csv`。

## 安全说明

API Key 只在运行时由侧边栏输入，保存在当前会话内存中，不写入任何文件，也不包含在本仓库内。
请勿把真实 Key 提交到仓库。

## 技术栈

Streamlit · pandas · Plotly · OpenAI Python SDK
