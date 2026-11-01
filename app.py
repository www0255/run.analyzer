import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

st.set_page_config(page_title="跑步数据分析器", layout="wide")
st.title("🏃 跑步数据分析与训练计划生成器")

# ---------------------- 侧边栏：AI 设置 ----------------------
with st.sidebar:
    st.header("⚙️ AI 设置")
    api_key = st.text_input("API Key", type="password")
    base_url = st.text_input("Base URL", value="https://api.doubao.com/v1")
    model_name = st.text_input("模型名称", value="doubao-lite-4k")

# ---------------------- 文件上传 ----------------------
uploaded_file = st.file_uploader("上传跑步记录 CSV", type="csv")
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 原始数据预览")
    st.dataframe(df, use_container_width=True)

    # 校验必要列
    required_cols = ["日期", "里程(km)", "时长(分钟)", "配速(min/km)", "平均心率"]
    if not all(c in df.columns for c in required_cols):
        st.error(f"CSV 需要包含列：{required_cols}")
    else:
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期")

        # ---------------------- 绘图 ----------------------
        st.subheader("📈 可视化图表")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_dist = px.line(df, x="日期", y="里程(km)", title="每次跑步里程趋势")
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            fig_pace = px.line(df, x="日期", y="配速(min/km)", title="配速变化（数值越小越快）")
            st.plotly_chart(fig_pace, use_container_width=True)
        with col3:
            fig_hr = px.line(df, x="日期", y="平均心率", title="平均心率变化")
            st.plotly_chart(fig_hr, use_container_width=True)

        # ---------------------- 统计指标 ----------------------
        total_km = df["里程(km)"].sum()
        avg_pace = df["配速(min/km)"].mean()
        avg_hr = df["平均心率"].mean()
        run_count = len(df)

        stat_text = f"""
        总跑步次数：{run_count}
        累计里程：{total_km:.2f} km
        平均配速：{avg_pace:.2f} min/km
        平均心率：{avg_hr:.0f}
        """
        st.subheader("📊 汇总统计")
        st.text(stat_text)

        # ---------------------- AI 生成训练方案 ----------------------
        st.subheader("🤖 AI 运动评估与训练计划")
        if st.button("生成分析与训练计划") and api_key:
            client = OpenAI(api_key=api_key, base_url=base_url)
            prompt = f"""
你是一位路跑教练，面向学生跑者。
下面是用户近期跑步统计：
{stat_text}
原始记录：
{df.to_csv(index=False)}

请输出：
1. 简要评估近期运动状态，注意心率风险提示；
2. 一份下周简易训练计划，适合学生，兼顾恢复，不要强度过高；
3. 文字简洁，适合展示在网页上。
            """
            with st.spinner("AI 分析中..."):
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                res = resp.choices[0].message.content
                st.markdown(res)
        elif not api_key:
            st.info("请在侧边栏填入 API Key 后再生成")
else:
    st.info("上传 run_data.csv 开始分析")
