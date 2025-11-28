"""
시공간적 불만 트렌드 분석 중 3.3.1 시계열 분석 모듈.

df_final.csv 기반 다양한 관점의 시계열 분석을 Streamlit 앱으로 제공.

전제:
- 입력 CSV: `df_final.csv`
- 필수 컬럼: year, month, region_code, Main_Topic, SubTopic_name

실행 방법:
    streamlit run 3.3.1_시계열데이터분석.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from streamlit_lottie import st_lottie
    import json
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "df_final.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "timeseries"

# 색상 팔레트
COLOR_PALETTE = px.colors.qualitative.Set3
COLOR_SCALE = px.colors.sequential.Viridis


# ========== 고급 시계열 분석 방법론 ==========

def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """단순 이동평균 (SMA)."""
    return series.rolling(window=window, min_periods=1).mean()


def exponential_moving_average(series: pd.Series, alpha: float = None, span: int = None) -> pd.Series:
    """지수 이동평균 (EMA)."""
    if span is not None:
        alpha = 2 / (span + 1)
    elif alpha is None:
        alpha = 0.3
    return series.ewm(alpha=alpha, adjust=False).mean()


def weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    """가중 이동평균 (WMA)."""
    weights = np.arange(1, window + 1)
    return series.rolling(window=window, min_periods=1).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def holt_winters_smoothing(series: pd.Series, seasonal_periods: int = 12) -> pd.Series:
    """Holt-Winters 지수평활법."""
    try:
        if len(series) < seasonal_periods * 2:
            return simple_moving_average(series, min(6, len(series)))
        
        model = ExponentialSmoothing(
            series,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        )
        fitted = model.fit()
        return pd.Series(fitted.fittedvalues, index=series.index)
    except Exception:
        return simple_moving_average(series, min(6, len(series)))


def loess_smoothing(series: pd.Series, frac: float = 0.3) -> pd.Series:
    """LOESS (Local Regression) 평활."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(series.values, np.arange(len(series)), frac=frac)
        return pd.Series(smoothed[:, 1], index=series.index)
    except ImportError:
        return simple_moving_average(series, min(6, len(series)))


def spline_smoothing(series: pd.Series, s: float = None) -> pd.Series:
    """스플라인 보간."""
    try:
        x = np.arange(len(series))
        spline = UnivariateSpline(x, series.values, s=s)
        return pd.Series(spline(x), index=series.index)
    except Exception:
        return simple_moving_average(series, min(6, len(series)))


def savitzky_golay_filter(series: pd.Series, window_length: int = 11, polyorder: int = 3) -> pd.Series:
    """Savitzky-Golay 필터."""
    try:
        if len(series) < window_length:
            return simple_moving_average(series, len(series))
        if window_length % 2 == 0:
            window_length += 1
        smoothed = savgol_filter(series.values, window_length, polyorder)
        return pd.Series(smoothed, index=series.index)
    except Exception:
        return simple_moving_average(series, min(6, len(series)))


def kalman_filter_smoothing(series: pd.Series) -> pd.Series:
    """칼만 필터 평활."""
    try:
        from pykalman import KalmanFilter
        kf = KalmanFilter(transition_matrices=[[1, 1], [0, 1]], observation_matrices=[[1, 0]])
        state_means, _ = kf.em(series.values).smooth(series.values)
        return pd.Series(state_means[:, 0], index=series.index)
    except ImportError:
        return simple_moving_average(series, min(6, len(series)))


def prophet_forecast(series: pd.Series, periods: int = 0) -> pd.Series:
    """Prophet 예측 및 평활."""
    if not PROPHET_AVAILABLE:
        return simple_moving_average(series, min(6, len(series)))
    
    try:
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
        return pd.Series(forecast['yhat'].values[:len(series)], index=series.index)
    except Exception:
        return simple_moving_average(series, min(6, len(series)))


def detect_changepoints(series: pd.Series, method: str = "pelt") -> list:
    """변화점 탐지."""
    try:
        from ruptures import detect
        if len(series) < 10:
            return []
        
        signal = series.values.reshape(-1, 1)
        algo = detect.Pelt(model="rbf").fit(signal)
        changepoints = algo.predict(pen=10)
        return [series.index[cp - 1] for cp in changepoints[:-1]]
    except ImportError:
        return []


def apply_smoothing_method(series: pd.Series, method: str, **kwargs) -> pd.Series:
    """선택한 평활 방법 적용."""
    series = series.dropna()
    
    if method == "단순 이동평균 (SMA)":
        window = kwargs.get('window', 6)
        return simple_moving_average(series, window)
    
    elif method == "지수 이동평균 (EMA)":
        span = kwargs.get('span', 6)
        return exponential_moving_average(series, span=span)
    
    elif method == "가중 이동평균 (WMA)":
        window = kwargs.get('window', 6)
        return weighted_moving_average(series, window)
    
    elif method == "Holt-Winters":
        seasonal = kwargs.get('seasonal_periods', 12)
        return holt_winters_smoothing(series, seasonal)
    
    elif method == "LOESS":
        frac = kwargs.get('frac', 0.3)
        return loess_smoothing(series, frac)
    
    elif method == "스플라인":
        s = kwargs.get('s', None)
        return spline_smoothing(series, s)
    
    elif method == "Savitzky-Golay":
        window = kwargs.get('window', 11)
        poly = kwargs.get('polyorder', 3)
        return savitzky_golay_filter(series, window, poly)
    
    elif method == "칼만 필터":
        return kalman_filter_smoothing(series)
    
    elif method == "Prophet":
        periods = kwargs.get('periods', 0)
        return prophet_forecast(series, periods)
    
    else:
        return series


@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """df_final.csv 로드 및 날짜 생성."""
    df = pd.read_csv(path)
    required_cols = {"year", "month", "region_code", "Main_Topic"}
    missing = required_cols.difference(df.columns)
    if missing:
        st.error(f"누락된 컬럼: {missing}")
        st.stop()
    
    # 날짜 생성
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df["topic"] = df["Main_Topic"]
    df["region"] = df["region_code"]
    
    return df


@st.cache_data
def compute_monthly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """월별 집계 및 증감률."""
    monthly = (
        df.groupby(["topic", "date"])
        .size()
        .reset_index(name="count")
        .sort_values(["topic", "date"])
    )
    
    # 증감률 계산
    monthly["yoy_rate"] = (
        monthly.groupby("topic")["count"].pct_change(periods=12).round(4)
    )
    monthly["mom_rate"] = (
        monthly.groupby("topic")["count"].pct_change(periods=1).round(4)
    )
    
    # 이동평균
    monthly["moving_avg_6m"] = (
        monthly.groupby("topic")["count"]
        .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
        .round(2)
    )
    monthly["moving_avg_12m"] = (
        monthly.groupby("topic")["count"]
        .transform(lambda s: s.rolling(window=12, min_periods=1).mean())
        .round(2)
    )
    
    # 정규화: 전체 대비 비율
    monthly_total = monthly.groupby("date")["count"].sum().reset_index(name="total")
    monthly = monthly.merge(monthly_total, on="date")
    monthly["pct_of_total"] = (monthly["count"] / monthly["total"] * 100).round(2)
    
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly["quarter"] = monthly["date"].dt.quarter
    
    return monthly


@st.cache_data
def compute_region_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """지역별 월별 집계."""
    region_monthly = (
        df.groupby(["region", "date"])
        .size()
        .reset_index(name="count")
        .sort_values(["region", "date"])
    )
    
    region_monthly["yoy_rate"] = (
        region_monthly.groupby("region")["count"].pct_change(periods=12).round(4)
    )
    region_monthly["mom_rate"] = (
        region_monthly.groupby("region")["count"].pct_change(periods=1).round(4)
    )
    
    # 정규화
    monthly_total = region_monthly.groupby("date")["count"].sum().reset_index(name="total")
    region_monthly = region_monthly.merge(monthly_total, on="date")
    region_monthly["pct_of_total"] = (region_monthly["count"] / region_monthly["total"] * 100).round(2)
    
    return region_monthly


@st.cache_data
def compute_topic_region_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """주제+지역 조합 월별 집계."""
    topic_region = (
        df.groupby(["topic", "region", "date"])
        .size()
        .reset_index(name="count")
        .sort_values(["topic", "region", "date"])
    )
    
    # 주제별 정규화
    topic_total = topic_region.groupby(["topic", "date"])["count"].sum().reset_index(name="topic_total")
    topic_region = topic_region.merge(topic_total, on=["topic", "date"])
    topic_region["pct_of_topic"] = (topic_region["count"] / topic_region["topic_total"] * 100).round(2)
    
    return topic_region


@st.cache_data
def compute_quarterly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """분기별 집계."""
    df["quarter"] = pd.PeriodIndex(df["date"], freq="Q").astype(str)
    quarterly = (
        df.groupby(["topic", "year", "quarter"])
        .size()
        .reset_index(name="count")
        .sort_values(["topic", "year", "quarter"])
    )
    quarterly["yoy_rate"] = (
        quarterly.groupby("topic")["count"].pct_change(periods=4).round(4)
    )
    quarterly["qoq_rate"] = (
        quarterly.groupby("topic")["count"].pct_change(periods=1).round(4)
    )
    return quarterly


@st.cache_data
def compute_yearly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 집계."""
    yearly = (
        df.groupby(["topic", "year"])
        .size()
        .reset_index(name="count")
        .sort_values(["topic", "year"])
    )
    yearly["yoy_rate"] = (
        yearly.groupby("topic")["count"].pct_change(periods=1).round(4)
    )
    yearly["cumulative"] = yearly.groupby("topic")["count"].cumsum()
    
    # 정규화
    yearly_total = yearly.groupby("year")["count"].sum().reset_index(name="total")
    yearly = yearly.merge(yearly_total, on="year")
    yearly["pct_of_total"] = (yearly["count"] / yearly["total"] * 100).round(2)
    
    return yearly


@st.cache_data
def compute_growth_analysis(monthly: pd.DataFrame) -> pd.DataFrame:
    """주제별 성장률 분석."""
    growth = []
    for topic in monthly["topic"].unique():
        topic_data = monthly[monthly["topic"] == topic].sort_values("date")
        if len(topic_data) < 2:
            continue
        
        first_count = topic_data.iloc[0]["count"]
        last_count = topic_data.iloc[-1]["count"]
        total_growth = ((last_count - first_count) / first_count * 100) if first_count > 0 else 0
        
        avg_monthly_growth = topic_data["mom_rate"].mean() * 100 if not topic_data["mom_rate"].isna().all() else 0
        avg_yoy_growth = topic_data["yoy_rate"].mean() * 100 if not topic_data["yoy_rate"].isna().all() else 0
        
        growth.append({
            "topic": topic,
            "first_count": first_count,
            "last_count": last_count,
            "total_growth_pct": round(total_growth, 2),
            "avg_monthly_growth_pct": round(avg_monthly_growth, 2),
            "avg_yoy_growth_pct": round(avg_yoy_growth, 2),
            "total_count": topic_data["count"].sum(),
            "avg_monthly_count": round(topic_data["count"].mean(), 2),
        })
    
    return pd.DataFrame(growth).sort_values("total_growth_pct", ascending=False)


@st.cache_data
def compute_market_share(monthly: pd.DataFrame) -> pd.DataFrame:
    """주제별 점유율 변화."""
    share = monthly[["topic", "date", "count", "pct_of_total"]].copy()
    share["share_change"] = share.groupby("topic")["pct_of_total"].pct_change().round(4)
    return share


@st.cache_data
def compute_cumulative(monthly: pd.DataFrame) -> pd.DataFrame:
    """누적 건수 추이."""
    cumulative = monthly.copy()
    cumulative["cumulative_count"] = cumulative.groupby("topic")["count"].cumsum()
    cumulative["cumulative_pct"] = (
        cumulative.groupby("date")["cumulative_count"].transform(lambda x: x / x.sum() * 100)
    ).round(2)
    return cumulative


@st.cache_data
def detect_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    """이상치 탐지 (Z-score 기반)."""
    anomalies = []
    for topic in monthly["topic"].unique():
        topic_data = monthly[monthly["topic"] == topic].copy()
        if len(topic_data) < 3:
            continue
        
        mean = topic_data["count"].mean()
        std = topic_data["count"].std()
        if std == 0:
            continue
        
        topic_data["z_score"] = (topic_data["count"] - mean) / std
        topic_data["is_anomaly"] = abs(topic_data["z_score"]) > 2
        
        anomaly_rows = topic_data[topic_data["is_anomaly"]]
        if not anomaly_rows.empty:
            anomalies.append(anomaly_rows[["topic", "date", "count", "z_score"]])
    
    if anomalies:
        return pd.concat(anomalies, ignore_index=True)
    return pd.DataFrame(columns=["topic", "date", "count", "z_score"])


def apply_custom_style(fig, title: str, height: int = 500):
    """차트에 커스텀 스타일 적용."""
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=60, r=100, t=80, b=60),
    )
    return fig


def show_loading_animation(message: str = "로딩 중..."):
    """로딩 애니메이션 표시."""
    loading_html = f"""
    <div class="loading-container">
        <div class="loading-spinner">
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
        </div>
        <p class="loading-text">{message}</p>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)


def show_progress_loader(message: str = "처리 중...", progress: float = 0.0):
    """프로그레스 바와 함께 로딩 표시."""
    progress_bar = st.progress(progress)
    status_text = st.empty()
    status_text.text(message)
    return progress_bar, status_text


def main():
    """Streamlit 메인 앱."""
    st.set_page_config(
        page_title="시계열 불만 트렌드 분석",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # 고급 커스텀 CSS
    st.markdown("""
    <style>
    /* 메인 헤더 */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 12px 24px;
        background-color: white;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* 입력 필드 스타일 */
    .stSelectbox, .stMultiselect, .stSlider {
        background-color: white;
        border-radius: 8px;
    }
    
    /* 확장 가능한 섹션 */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 600;
    }
    
    /* 메인 컨테이너 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 카드 스타일 */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: box-shadow 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* 로딩 애니메이션 */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        min-height: 300px;
    }
    
    .loading-spinner {
        position: relative;
        width: 80px;
        height: 80px;
        margin-bottom: 1.5rem;
    }
    
    .spinner-ring {
        position: absolute;
        width: 100%;
        height: 100%;
        border: 4px solid transparent;
        border-top-color: #667eea;
        border-radius: 50%;
        animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
    }
    
    .spinner-ring:nth-child(1) {
        animation-delay: -0.45s;
        border-top-color: #667eea;
    }
    
    .spinner-ring:nth-child(2) {
        animation-delay: -0.3s;
        border-top-color: #764ba2;
        width: 70%;
        height: 70%;
        top: 15%;
        left: 15%;
    }
    
    .spinner-ring:nth-child(3) {
        animation-delay: -0.15s;
        border-top-color: #f093fb;
        width: 50%;
        height: 50%;
        top: 25%;
        left: 25%;
    }
    
    .spinner-ring:nth-child(4) {
        border-top-color: #4facfe;
        width: 30%;
        height: 30%;
        top: 35%;
        left: 35%;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        margin-top: 1rem;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* 간단한 로딩 스피너 (작은 버전) */
    .mini-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-top-color: #667eea;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    /* 프로그레스 바 스타일 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 메인 헤더
    st.markdown('<h1 class="main-header fade-in">📊 시공간적 불만 트렌드 분석</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">고급 시계열 분석 대시보드</p>', unsafe_allow_html=True)
    
    # 데이터 로딩
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        show_loading_animation("데이터 로딩 중...")
        
        df = load_data()
        progress_bar, status_text = show_progress_loader("데이터 처리 중...", 0.1)
        
        monthly = compute_monthly_timeseries(df)
        progress_bar.progress(0.3)
        status_text.text("월별 시계열 계산 중...")
        
        region_monthly = compute_region_timeseries(df)
        progress_bar.progress(0.4)
        status_text.text("지역별 시계열 계산 중...")
        
        topic_region = compute_topic_region_timeseries(df)
        progress_bar.progress(0.5)
        status_text.text("주제+지역 조합 계산 중...")
        
        quarterly = compute_quarterly_timeseries(df)
        progress_bar.progress(0.6)
        status_text.text("분기별 시계열 계산 중...")
        
        yearly = compute_yearly_timeseries(df)
        progress_bar.progress(0.7)
        status_text.text("연도별 시계열 계산 중...")
        
        growth = compute_growth_analysis(monthly)
        progress_bar.progress(0.8)
        status_text.text("성장률 분석 중...")
        
        market_share = compute_market_share(monthly)
        progress_bar.progress(0.9)
        status_text.text("점유율 분석 중...")
        
        cumulative = compute_cumulative(monthly)
        anomalies = detect_anomalies(monthly)
        
        progress_bar.progress(1.0)
        status_text.text("완료!")
        
        import time
        time.sleep(0.3)  # 완료 메시지 표시를 위한 짧은 지연
    
    loading_placeholder.empty()
    
    # 사이드바 필터
    with st.sidebar:
        st.header("🎛️ 필터 옵션")
        
        # 주제 선택
        all_topics = sorted(monthly["topic"].unique())
        selected_topics = st.multiselect(
            "📌 주제 선택",
            all_topics,
            default=all_topics[:5] if len(all_topics) > 5 else all_topics,
            help="분석할 주제를 선택하세요"
        )
        
        # 지역 선택
        all_regions = sorted(df["region"].unique())
        selected_regions = st.multiselect(
            "📍 지역 선택",
            all_regions,
            default=all_regions,
            help="분석할 지역을 선택하세요"
        )
        
        # 날짜 범위
        min_date = monthly["date"].min()
        max_date = monthly["date"].max()
        date_range = st.date_input(
            "📅 날짜 범위",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="분석할 기간을 선택하세요"
        )
        
        st.markdown("---")
        st.markdown("### 📊 통계 요약")
        total_count = df.shape[0]
        topic_count = len(all_topics)
        region_count = len(all_regions)
        
        # 메트릭 카드 스타일 적용
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem; color: white;">
            <div style="font-size: 0.9rem; opacity: 0.9;">총 민원 건수</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{:,}</div>
        </div>
        """.format(total_count), unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📌 주제 수", topic_count, delta=None)
            st.metric("📍 지역 수", region_count, delta=None)
        with col2:
            st.metric("📅 시작", min_date.strftime('%Y-%m'))
            st.metric("📅 종료", max_date.strftime('%Y-%m'))
    
    # 필터 적용 (필터 변경 시 로딩 표시)
    filter_changed = st.session_state.get('prev_topics', None) != selected_topics or \
                     st.session_state.get('prev_date_range', None) != date_range
    
    if filter_changed:
        filter_loading = st.empty()
        with filter_loading.container():
            st.markdown('<div class="mini-spinner"></div><span style="color: #667eea;">필터 적용 중...</span>', unsafe_allow_html=True)
    
    if selected_topics:
        monthly_filtered = monthly[monthly["topic"].isin(selected_topics)].copy()
    else:
        monthly_filtered = monthly.copy()
    
    if len(date_range) == 2:
        monthly_filtered = monthly_filtered[
            (monthly_filtered["date"] >= pd.Timestamp(date_range[0])) &
            (monthly_filtered["date"] <= pd.Timestamp(date_range[1]))
        ]
    
    # 세션 상태 업데이트
    st.session_state['prev_topics'] = selected_topics
    st.session_state['prev_date_range'] = date_range
    
    if filter_changed:
        filter_loading.empty()
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 기본 시계열", "📊 정규화 분석", "📉 성장률 분석",
        "🗺️ 지역별 분석", "📅 시간 단위 분석", "📦 분포 분석",
        "🔍 고급 분석", "📋 데이터 테이블"
    ])
    
    # 탭 1: 기본 시계열
    with tab1:
        st.markdown("### 📈 기본 시계열 분석")
        st.markdown("---")
        
        # 차트 옵션
        with st.expander("⚙️ 차트 설정", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                show_markers = st.checkbox("마커 표시", value=True)
                show_smoothing = st.checkbox("평활 기법 적용", value=False)
                smoothing_method = st.selectbox(
                    "평활 방법",
                    [
                        "단순 이동평균 (SMA)",
                        "지수 이동평균 (EMA)",
                        "가중 이동평균 (WMA)",
                        "Holt-Winters",
                        "LOESS",
                        "스플라인",
                        "Savitzky-Golay",
                        "칼만 필터",
                        "Prophet"
                    ],
                    index=0,
                    disabled=not show_smoothing
                )
            with col2:
                chart_type = st.selectbox("차트 타입", ["라인", "영역", "바"], index=0)
                show_legend = st.checkbox("범례 표시", value=True)
                smoothing_params = {}
                if show_smoothing:
                    if "이동평균" in smoothing_method or "WMA" in smoothing_method:
                        smoothing_params['window'] = st.slider("윈도우 크기", 3, 24, 6)
                    elif "EMA" in smoothing_method:
                        smoothing_params['span'] = st.slider("Span", 3, 24, 6)
                    elif smoothing_method == "LOESS":
                        smoothing_params['frac'] = st.slider("Fraction", 0.1, 0.9, 0.3, 0.1)
                    elif smoothing_method == "Savitzky-Golay":
                        smoothing_params['window'] = st.slider("윈도우 크기", 5, 21, 11, 2)
                        smoothing_params['polyorder'] = st.slider("다항식 차수", 1, 5, 3)
            with col3:
                y_axis_type = st.selectbox("Y축 타입", ["선형", "로그"], index=0)
                show_grid = st.checkbox("그리드 표시", value=True)
                show_changepoints = st.checkbox("변화점 탐지", value=False) if show_smoothing else False
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("주제별 월간 민원 트렌드")
            
            # 원본 데이터로 차트 생성
            if chart_type == "라인":
                fig = px.line(
                    monthly_filtered,
                    x="date",
                    y="count",
                    color="topic",
                    markers=show_markers,
                    title="주제별 월간 민원 건수",
                    labels={"count": "건수", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
            elif chart_type == "영역":
                fig = px.area(
                    monthly_filtered,
                    x="date",
                    y="count",
                    color="topic",
                    title="주제별 월간 민원 건수",
                    labels={"count": "건수", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
            else:
                fig = px.bar(
                    monthly_filtered,
                    x="date",
                    y="count",
                    color="topic",
                    title="주제별 월간 민원 건수",
                    labels={"count": "건수", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
            
            # 평활 기법 적용
            if show_smoothing:
                for topic in monthly_filtered["topic"].unique():
                    topic_data = monthly_filtered[monthly_filtered["topic"] == topic].sort_values("date")
                    if len(topic_data) > 0:
                        series = topic_data.set_index("date")["count"]
                        smoothed = apply_smoothing_method(series, smoothing_method, **smoothing_params)
                        
                        fig.add_trace(go.Scatter(
                            x=smoothed.index,
                            y=smoothed.values,
                            name=f"{topic} ({smoothing_method})",
                            mode="lines",
                            line=dict(width=3, dash="dash"),
                            opacity=0.8,
                        ))
                        
                        # 변화점 탐지
                        if show_changepoints:
                            changepoints = detect_changepoints(series)
                            if changepoints:
                                cp_values = [series.loc[cp] for cp in changepoints if cp in series.index]
                                fig.add_trace(go.Scatter(
                                    x=changepoints,
                                    y=cp_values,
                                    mode="markers",
                                    marker=dict(size=12, color="red", symbol="diamond"),
                                    name=f"{topic} 변화점",
                                    showlegend=(topic == monthly_filtered["topic"].unique()[0]),
                                ))
            
            fig.update_layout(
                yaxis_type="log" if y_axis_type == "로그" else "linear",
                showlegend=show_legend,
                xaxis_showgrid=show_grid,
                yaxis_showgrid=show_grid,
            )
            fig = apply_custom_style(fig, "주제별 월간 민원 건수")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with col2:
            st.subheader("누적 건수 추이")
            
            cumulative_filtered = cumulative[cumulative["topic"].isin(selected_topics) if selected_topics else cumulative["topic"].isin(all_topics)]
            if len(date_range) == 2:
                cumulative_filtered = cumulative_filtered[
                    (cumulative_filtered["date"] >= pd.Timestamp(date_range[0])) &
                    (cumulative_filtered["date"] <= pd.Timestamp(date_range[1]))
                ]
            
            fig = px.line(
                cumulative_filtered,
                x="date",
                y="cumulative_count",
                color="topic",
                title="주제별 누적 민원 건수",
                labels={"cumulative_count": "누적 건수", "date": "월"},
                color_discrete_sequence=COLOR_PALETTE,
            )
            fig = apply_custom_style(fig, "주제별 누적 민원 건수")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        # 비교 모드
        st.subheader("📊 비교 모드")
        compare_mode = st.radio("비교 방식", ["단일 주제 상세", "주제 간 비교", "전체 요약", "평활 기법 비교"], horizontal=True)
        
        if compare_mode == "단일 주제 상세":
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_topic_detail = st.selectbox("상세 분석할 주제", all_topics)
            with col2:
                detail_smoothing = st.selectbox(
                    "평활 기법",
                    ["없음", "단순 이동평균 (SMA)", "지수 이동평균 (EMA)", "Holt-Winters", "LOESS", "Prophet"],
                    index=0
                )
            
            topic_data = monthly[monthly["topic"] == selected_topic_detail].sort_values("date")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("월별 건수", "증감률"),
                vertical_spacing=0.15,
            )
            
            fig.add_trace(
                go.Scatter(x=topic_data["date"], y=topic_data["count"], name="원본", mode="lines+markers", line=dict(color="blue")),
                row=1, col=1
            )
            
            # 평활 기법 적용
            if detail_smoothing != "없음":
                series = topic_data.set_index("date")["count"]
                if detail_smoothing == "단순 이동평균 (SMA)":
                    smoothed = apply_smoothing_method(series, "단순 이동평균 (SMA)", window=6)
                elif detail_smoothing == "지수 이동평균 (EMA)":
                    smoothed = apply_smoothing_method(series, "지수 이동평균 (EMA)", span=6)
                elif detail_smoothing == "Holt-Winters":
                    smoothed = apply_smoothing_method(series, "Holt-Winters", seasonal_periods=12)
                elif detail_smoothing == "LOESS":
                    smoothed = apply_smoothing_method(series, "LOESS", frac=0.3)
                elif detail_smoothing == "Prophet":
                    smoothed = apply_smoothing_method(series, "Prophet", periods=0)
                
                fig.add_trace(
                    go.Scatter(x=smoothed.index, y=smoothed.values, name=f"{detail_smoothing}", mode="lines", line=dict(color="red", dash="dash", width=2)),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Scatter(x=topic_data["date"], y=topic_data["yoy_rate"]*100, name="전년 대비 (%)", mode="lines+markers", line=dict(color="green")),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=topic_data["date"], y=topic_data["mom_rate"]*100, name="전월 대비 (%)", mode="lines+markers", line=dict(color="orange")),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="월", row=2, col=1)
            fig.update_yaxes(title_text="건수", row=1, col=1)
            fig.update_yaxes(title_text="증감률 (%)", row=2, col=1)
            fig.update_layout(height=600, title_text=f"{selected_topic_detail} 상세 분석")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        elif compare_mode == "평활 기법 비교":
            selected_topic_compare = st.selectbox("비교할 주제", all_topics, key="smoothing_compare")
            topic_data = monthly[monthly["topic"] == selected_topic_compare].sort_values("date")
            series = topic_data.set_index("date")["count"]
            
            methods_to_compare = st.multiselect(
                "비교할 평활 기법",
                ["단순 이동평균 (SMA)", "지수 이동평균 (EMA)", "가중 이동평균 (WMA)", "Holt-Winters", "LOESS", "스플라인", "Savitzky-Golay"],
                default=["단순 이동평균 (SMA)", "지수 이동평균 (EMA)", "Holt-Winters"]
            )
            
            if methods_to_compare:
                with st.spinner("평활 기법 계산 중..."):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        name="원본",
                        mode="lines+markers",
                        line=dict(width=2, color="black"),
                    ))
                    
                    colors = px.colors.qualitative.Set2
                    for i, method in enumerate(methods_to_compare):
                        try:
                            if method == "단순 이동평균 (SMA)":
                                smoothed = apply_smoothing_method(series, method, window=6)
                            elif method == "지수 이동평균 (EMA)":
                                smoothed = apply_smoothing_method(series, method, span=6)
                            elif method == "가중 이동평균 (WMA)":
                                smoothed = apply_smoothing_method(series, method, window=6)
                            elif method == "Holt-Winters":
                                smoothed = apply_smoothing_method(series, method, seasonal_periods=12)
                            elif method == "LOESS":
                                smoothed = apply_smoothing_method(series, method, frac=0.3)
                            elif method == "스플라인":
                                smoothed = apply_smoothing_method(series, method, s=None)
                            elif method == "Savitzky-Golay":
                                smoothed = apply_smoothing_method(series, method, window=11, polyorder=3)
                            
                            fig.add_trace(go.Scatter(
                                x=smoothed.index,
                                y=smoothed.values,
                                name=method,
                                mode="lines",
                                line=dict(width=2, dash="dash", color=colors[i % len(colors)]),
                            ))
                        except Exception as e:
                            st.warning(f"{method} 적용 실패: {e}")
                    
                    fig.update_layout(
                        title=f"{selected_topic_compare} - 평활 기법 비교",
                        xaxis_title="월",
                        yaxis_title="건수",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        elif compare_mode == "주제 간 비교":
            compare_topics = st.multiselect("비교할 주제 (최대 5개)", all_topics, default=all_topics[:3])
            if compare_topics:
                compare_data = monthly[monthly["topic"].isin(compare_topics)]
                fig = px.line(
                    compare_data,
                    x="date",
                    y="count",
                    color="topic",
                    facet_row="topic" if len(compare_topics) <= 3 else None,
                    title="주제 간 비교",
                    labels={"count": "건수", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig = apply_custom_style(fig, "주제 간 비교", height=300*len(compare_topics))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    
    # 탭 2: 정규화 분석
    with tab2:
        st.header("📊 정규화 분석 (전체 대비 비율)")
        
        with st.expander("⚙️ 차트 설정", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                normalize_type = st.selectbox("정규화 타입", ["전체 대비 비율", "주제별 정규화", "누적 비율"], index=0)
                stack_mode = st.checkbox("스택 모드", value=True)
            with col2:
                show_percentage = st.checkbox("퍼센트 표시", value=True)
                show_trend_line = st.checkbox("추세선 표시", value=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("주제별 점유율 변화")
            
            if normalize_type == "전체 대비 비율":
                y_col = "pct_of_total"
            elif normalize_type == "주제별 정규화":
                topic_region_filtered = topic_region[
                    topic_region["topic"].isin(selected_topics) if selected_topics else True
                ]
                y_col = "pct_of_topic"
                data = topic_region_filtered
            else:
                y_col = "cumulative_pct"
                data = cumulative[cumulative["topic"].isin(selected_topics) if selected_topics else cumulative["topic"].isin(all_topics)]
            
            if normalize_type == "전체 대비 비율":
                data = monthly_filtered
            
            if stack_mode:
                fig = px.area(
                    data,
                    x="date",
                    y=y_col,
                    color="topic",
                    title="주제별 민원 점유율 (스택 영역)",
                    labels={y_col: "점유율 (%)", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
            else:
                fig = px.line(
                    data,
                    x="date",
                    y=y_col,
                    color="topic",
                    title="주제별 민원 점유율",
                    labels={y_col: "점유율 (%)", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
            
            if show_trend_line:
                for topic in data["topic"].unique():
                    topic_data = data[data["topic"] == topic].sort_values("date")
                    z = np.polyfit(range(len(topic_data)), topic_data[y_col], 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=topic_data["date"],
                        y=p(range(len(topic_data))),
                        name=f"{topic} (추세)",
                        mode="lines",
                        line=dict(dash="dot", width=1),
                        opacity=0.5,
                    ))
            
            fig.update_layout(yaxis_range=[0, 100])
            fig = apply_custom_style(fig, "주제별 민원 점유율")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with col2:
            st.subheader("연도별 정규화 비교")
            yearly_filtered = yearly[yearly["topic"].isin(selected_topics) if selected_topics else yearly["topic"].isin(all_topics)]
            
            fig = px.bar(
                yearly_filtered,
                x="year",
                y="pct_of_total",
                color="topic",
                title="연도별 주제별 민원 점유율",
                labels={"pct_of_total": "점유율 (%)", "year": "연도"},
                barmode="stack" if stack_mode else "group",
                color_discrete_sequence=COLOR_PALETTE,
            )
            fig.update_layout(yaxis_range=[0, 100])
            fig = apply_custom_style(fig, "연도별 주제별 민원 점유율")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        # 점유율 변화율
        st.subheader("점유율 변화율 분석")
        share_filtered = market_share[market_share["topic"].isin(selected_topics) if selected_topics else market_share["topic"].isin(all_topics)]
        
        fig = px.bar(
            share_filtered.groupby("topic")["share_change"].mean().reset_index(),
            x="topic",
            y="share_change",
            title="주제별 평균 점유율 변화율",
            labels={"share_change": "변화율", "topic": "주제"},
            color="share_change",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig = apply_custom_style(fig, "주제별 평균 점유율 변화율")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    
    # 탭 3: 성장률 분석
    with tab3:
        st.header("📉 성장률 분석")
        
        with st.expander("⚙️ 차트 설정", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                growth_metric = st.selectbox("성장률 지표", ["전체 성장률", "월평균 성장률", "전년 대비 평균 성장률"], index=0)
                sort_by = st.selectbox("정렬 기준", ["성장률", "총 건수", "평균 건수"], index=0)
            with col2:
                show_top_n = st.slider("상위 N개 표시", 5, len(all_topics), min(10, len(all_topics)))
                show_negative = st.checkbox("음수 성장률 표시", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("주제별 성장률 비교")
            
            growth_filtered = growth[growth["topic"].isin(selected_topics) if selected_topics else growth["topic"].isin(all_topics)]
            
            if growth_metric == "전체 성장률":
                y_col = "total_growth_pct"
            elif growth_metric == "월평균 성장률":
                y_col = "avg_monthly_growth_pct"
            else:
                y_col = "avg_yoy_growth_pct"
            
            if sort_by == "성장률":
                growth_filtered = growth_filtered.sort_values(y_col, ascending=False)
            elif sort_by == "총 건수":
                growth_filtered = growth_filtered.sort_values("total_count", ascending=False)
            else:
                growth_filtered = growth_filtered.sort_values("avg_monthly_count", ascending=False)
            
            growth_filtered = growth_filtered.head(show_top_n)
            
            if not show_negative:
                growth_filtered = growth_filtered[growth_filtered[y_col] >= 0]
            
            fig = px.bar(
                growth_filtered,
                x="topic",
                y=y_col,
                title=f"주제별 {growth_metric}",
                labels={y_col: "성장률 (%)", "topic": "주제"},
                color=y_col,
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig = apply_custom_style(fig, f"주제별 {growth_metric}")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with col2:
            st.subheader("전년 대비 증감률 히트맵")
            
            heatmap_topics = st.multiselect("히트맵 주제 선택", all_topics, default=selected_topics[:10] if selected_topics else all_topics[:10])
            
            if heatmap_topics:
                with st.spinner("히트맵 생성 중..."):
                    monthly_heatmap = monthly_filtered[monthly_filtered["topic"].isin(heatmap_topics)]
                    pivot = monthly_heatmap.pivot_table(
                        index="topic",
                        columns=monthly_heatmap["date"].dt.to_period("M").astype(str),
                        values="yoy_rate",
                        aggfunc="mean",
                    )
                    if not pivot.empty:
                        fig = px.imshow(
                            pivot,
                            labels=dict(x="월", y="주제", color="전년 대비 증감률"),
                            title="주제별 전년 대비 증감률 히트맵",
                            color_continuous_scale="RdYlGn",
                            aspect="auto",
                        )
                        fig = apply_custom_style(fig, "주제별 전년 대비 증감률 히트맵", height=400)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        # 성장률 상세 테이블
        st.subheader("성장률 상세 테이블")
        st.dataframe(growth_filtered, use_container_width=True)
    
    # 탭 4: 지역별 분석
    with tab4:
        st.header("🗺️ 지역별 분석")
        
        with st.expander("⚙️ 차트 설정", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                region_chart_type = st.selectbox("차트 타입", ["라인", "영역", "히트맵"], index=0)
                show_region_normalized = st.checkbox("정규화 표시", value=False)
            with col2:
                region_aggregation = st.selectbox("집계 방식", ["합계", "평균", "최대값"], index=0)
        
        region_filtered = region_monthly[
            region_monthly["region"].isin(selected_regions) if selected_regions else region_monthly["region"].isin(all_regions)
        ]
        
        if len(date_range) == 2:
            region_filtered = region_filtered[
                (region_filtered["date"] >= pd.Timestamp(date_range[0])) &
                (region_filtered["date"] <= pd.Timestamp(date_range[1]))
            ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("지역별 월간 민원 트렌드")
            
            y_col = "pct_of_total" if show_region_normalized else "count"
            
            if region_chart_type == "라인":
                fig = px.line(
                    region_filtered,
                    x="date",
                    y=y_col,
                    color="region",
                    title="지역별 월간 민원",
                    labels={y_col: "건수" if not show_region_normalized else "점유율 (%)", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
            elif region_chart_type == "영역":
                fig = px.area(
                    region_filtered,
                    x="date",
                    y=y_col,
                    color="region",
                    title="지역별 월간 민원",
                    labels={y_col: "건수" if not show_region_normalized else "점유율 (%)", "date": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
            else:
                pivot = region_filtered.pivot_table(
                    index="region",
                    columns=region_filtered["date"].dt.to_period("M").astype(str),
                    values=y_col,
                    aggfunc=region_aggregation.lower(),
                )
                fig = px.imshow(
                    pivot,
                    labels=dict(x="월", y="지역", color="건수" if not show_region_normalized else "점유율 (%)"),
                    title="지역별 민원 히트맵",
                    color_continuous_scale=COLOR_SCALE,
                    aspect="auto",
                )
            
            fig = apply_custom_style(fig, "지역별 월간 민원")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with col2:
            st.subheader("주제별 지역별 히트맵")
            
            topic_region_filtered = topic_region[
                (topic_region["topic"].isin(selected_topics) if selected_topics else True) &
                (topic_region["region"].isin(selected_regions) if selected_regions else True)
            ]
            
            pivot = topic_region_filtered.pivot_table(
                index="topic",
                columns="region",
                values="count",
                aggfunc="sum",
            )
            if not pivot.empty:
                fig = px.imshow(
                    pivot,
                    labels=dict(x="지역", y="주제", color="건수"),
                    title="주제별 지역별 민원 건수 히트맵",
                    color_continuous_scale=COLOR_SCALE,
                    aspect="auto",
                )
                fig = apply_custom_style(fig, "주제별 지역별 민원 건수 히트맵", height=400)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        # 지역별 증감률
        st.subheader("지역별 증감률 히트맵")
        region_pivot = region_filtered.pivot_table(
            index="region",
            columns=region_filtered["date"].dt.to_period("M").astype(str),
            values="yoy_rate",
            aggfunc="mean",
        )
        if not region_pivot.empty:
            fig = px.imshow(
                region_pivot,
                labels=dict(x="월", y="지역", color="전년 대비 증감률"),
                title="지역별 전년 대비 증감률 히트맵",
                color_continuous_scale="RdYlGn",
                aspect="auto",
            )
            fig = apply_custom_style(fig, "지역별 전년 대비 증감률 히트맵", height=400)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    
    # 탭 5: 시간 단위 분석
    with tab5:
        st.header("📅 시간 단위별 분석")
        
        with st.expander("⚙️ 차트 설정", expanded=False):
            time_unit = st.selectbox("시간 단위", ["월별", "분기별", "연도별"], index=0)
            show_seasonal = st.checkbox("계절 패턴 표시", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if time_unit == "분기별":
                st.subheader("분기별 비교")
                quarterly_filtered = quarterly[quarterly["topic"].isin(selected_topics) if selected_topics else quarterly["topic"].isin(all_topics)]
                fig = px.bar(
                    quarterly_filtered,
                    x="quarter",
                    y="count",
                    color="topic",
                    facet_col="year",
                    title="연도별 분기별 민원 건수",
                    labels={"count": "건수", "quarter": "분기"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig = apply_custom_style(fig, "연도별 분기별 민원 건수", height=400)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
            else:
                st.subheader("연도별 비교")
                yearly_filtered = yearly[yearly["topic"].isin(selected_topics) if selected_topics else yearly["topic"].isin(all_topics)]
                fig = px.bar(
                    yearly_filtered,
                    x="year",
                    y="count",
                    color="topic",
                    title="연도별 주제별 민원 건수",
                    labels={"count": "건수", "year": "연도"},
                    barmode="group",
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig = apply_custom_style(fig, "연도별 주제별 민원 건수")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with col2:
            if show_seasonal:
                st.subheader("계절 패턴 (월별 평균)")
                monthly_filtered["month_name"] = monthly_filtered["date"].dt.month_name()
                seasonal = monthly_filtered.groupby(["topic", "month"])["count"].mean().reset_index()
                fig = px.line(
                    seasonal,
                    x="month",
                    y="count",
                    color="topic",
                    title="주제별 계절 패턴 (월별 평균 건수)",
                    labels={"count": "평균 건수", "month": "월"},
                    color_discrete_sequence=COLOR_PALETTE,
                    markers=True,
                )
                fig = apply_custom_style(fig, "주제별 계절 패턴")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    
    # 탭 6: 분포 분석
    with tab6:
        st.header("📦 분포 분석")
        
        with st.expander("⚙️ 차트 설정", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                dist_type = st.selectbox("분포 차트 타입", ["박스플롯", "바이올린", "히스토그램"], index=0)
                show_outliers = st.checkbox("이상치 표시", value=True)
            with col2:
                group_by = st.selectbox("그룹화 기준", ["주제", "지역", "연도"], index=0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{group_by}별 분포")
            
            if group_by == "주제":
                data = monthly_filtered
                group_col = "topic"
            elif group_by == "지역":
                data = region_filtered
                group_col = "region"
            else:
                data = monthly_filtered
                group_col = "year"
            
            if dist_type == "박스플롯":
                fig = px.box(
                    data,
                    x=group_col,
                    y="count",
                    title=f"{group_by}별 민원 건수 분포",
                    labels={"count": "건수", group_col: group_by},
                    color=group_col,
                    color_discrete_sequence=COLOR_PALETTE,
                )
            elif dist_type == "바이올린":
                fig = px.violin(
                    data,
                    x=group_col,
                    y="count",
                    title=f"{group_by}별 민원 건수 분포 (바이올린)",
                    labels={"count": "건수", group_col: group_by},
                    color=group_col,
                    color_discrete_sequence=COLOR_PALETTE,
                )
            else:
                fig = px.histogram(
                    data,
                    x="count",
                    color=group_col,
                    title=f"{group_by}별 민원 건수 분포",
                    labels={"count": "건수"},
                    color_discrete_sequence=COLOR_PALETTE,
                    nbins=30,
                )
            
            fig.update_layout(xaxis_tickangle=-45 if group_by != "연도" else 0)
            fig = apply_custom_style(fig, f"{group_by}별 민원 건수 분포")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with col2:
            st.subheader("증감률 분포")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("전월 대비 증감률", "전년 대비 증감률"),
            )
            
            for i, topic in enumerate(monthly_filtered["topic"].unique()[:5]):
                topic_data = monthly_filtered[monthly_filtered["topic"] == topic]
                fig.add_trace(
                    go.Histogram(x=topic_data["mom_rate"].dropna(), name=topic, opacity=0.7, nbinsx=20),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Histogram(x=topic_data["yoy_rate"].dropna(), name=topic, opacity=0.7, nbinsx=20),
                    row=1, col=2
                )
            
            fig.update_layout(title="주제별 증감률 분포", showlegend=True, height=400)
            fig.update_xaxes(title_text="증감률", row=1, col=1)
            fig.update_xaxes(title_text="증감률", row=1, col=2)
            fig.update_yaxes(title_text="빈도", row=1, col=1)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    
    # 탭 7: 고급 분석
    with tab7:
        st.header("🔍 고급 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("상위/하위 주제 비교")
            n = st.slider("상위/하위 개수", 3, 10, 5, key="top_bottom")
            topic_totals = monthly.groupby("topic")["count"].sum().sort_values(ascending=False)
            top_topics = topic_totals.head(n).index.tolist()
            bottom_topics = topic_totals.tail(n).index.tolist()
            
            top_data = monthly[monthly["topic"].isin(top_topics)]
            bottom_data = monthly[monthly["topic"].isin(bottom_topics)]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f"상위 {n}개 주제", f"하위 {n}개 주제"),
                vertical_spacing=0.1,
            )
            
            for topic in top_topics:
                topic_data = top_data[top_data["topic"] == topic].sort_values("date")
                fig.add_trace(
                    go.Scatter(x=topic_data["date"], y=topic_data["count"], name=topic, mode="lines+markers"),
                    row=1, col=1
                )
            
            for topic in bottom_topics:
                topic_data = bottom_data[bottom_data["topic"] == topic].sort_values("date")
                fig.add_trace(
                    go.Scatter(x=topic_data["date"], y=topic_data["count"], name=topic, mode="lines+markers"),
                    row=2, col=1
                )
            
            fig.update_layout(title="상위/하위 주제 비교", height=600, showlegend=True)
            fig.update_xaxes(title_text="월", row=2, col=1)
            fig.update_yaxes(title_text="건수", row=1, col=1)
            fig.update_yaxes(title_text="건수", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with col2:
            st.subheader("주제 간 상관관계")
            with st.spinner("상관관계 계산 중..."):
                pivot = monthly.pivot(index="date", columns="topic", values="count").fillna(0)
                corr = pivot.corr()
                
                fig = px.imshow(
                    corr,
                    labels=dict(x="주제", y="주제", color="상관계수"),
                    title="주제 간 상관관계 매트릭스",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                )
                fig = apply_custom_style(fig, "주제 간 상관관계 매트릭스", height=500)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        # 이상치 탐지
        st.subheader("이상치 탐지")
        if not anomalies.empty:
            fig = go.Figure()
            
            for topic in monthly_filtered["topic"].unique():
                topic_data = monthly_filtered[monthly_filtered["topic"] == topic].sort_values("date")
                fig.add_trace(go.Scatter(
                    x=topic_data["date"],
                    y=topic_data["count"],
                    name=topic,
                    mode="lines",
                    line=dict(width=1),
                ))
            
            anomalies_filtered = anomalies[anomalies["topic"].isin(selected_topics) if selected_topics else True]
            if not anomalies_filtered.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies_filtered["date"],
                    y=anomalies_filtered["count"],
                    mode="markers",
                    marker=dict(size=15, color="red", symbol="x", line=dict(width=2, color="darkred")),
                    name="이상치",
                ))
            
            fig.update_layout(
                title="주제별 시계열 및 이상치 탐지",
                xaxis_title="월",
                yaxis_title="건수",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
            
            st.dataframe(anomalies_filtered, use_container_width=True)
        else:
            st.info("이상치가 발견되지 않았습니다.")
        
        # 계절 분해
        st.subheader("계절 분해")
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_topic_decomp = st.selectbox(
                "분해할 주제 선택",
                monthly.groupby("topic")["count"].sum().nlargest(10).index.tolist(),
                key="decomp_topic"
            )
        with col2:
            decomp_model = st.selectbox("분해 모델", ["additive", "multiplicative"], index=0, key="decomp_model")
            decomp_period = st.slider("계절 주기", 6, 24, 12, key="decomp_period")
        
        topic_data = monthly[monthly["topic"] == selected_topic_decomp].sort_values("date")
        if len(topic_data) >= decomp_period * 2:
            ts = topic_data.set_index("date")["count"]
            try:
                with st.spinner("계절 분해 계산 중..."):
                    decomposition = seasonal_decompose(ts, model=decomp_model, period=decomp_period)
                
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=("원본", "추세", "계절성", "잔차"),
                    vertical_spacing=0.08,
                )
                
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="원본", line=dict(color="blue")), row=1, col=1)
                fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name="추세", line=dict(color="green")), row=2, col=1)
                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name="계절성", line=dict(color="orange")), row=3, col=1)
                fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name="잔차", line=dict(color="red")), row=4, col=1)
                
                fig.update_layout(
                    title=f"{selected_topic_decomp} - 계절 분해 ({decomp_model}, period={decomp_period})",
                    height=800,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
                
                # 통계 요약
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("추세 변동성", f"{decomposition.trend.std():.2f}")
                with col2:
                    st.metric("계절성 변동성", f"{decomposition.seasonal.std():.2f}")
                with col3:
                    st.metric("잔차 변동성", f"{decomposition.resid.std():.2f}")
                with col4:
                    st.metric("계절성 강도", f"{(decomposition.seasonal.std() / decomposition.trend.std()):.2f}" if decomposition.trend.std() > 0 else "N/A")
            except Exception as e:
                st.error(f"계절 분해 실패: {e}")
        else:
            st.warning(f"계절 분해를 위해서는 최소 {decomp_period * 2}개월 데이터가 필요합니다.")
        
        # 시계열 예측
        st.subheader("시계열 예측")
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_topic_forecast = st.selectbox(
                "예측할 주제",
                monthly.groupby("topic")["count"].sum().nlargest(10).index.tolist(),
                key="forecast_topic"
            )
        with col2:
            forecast_periods = st.slider("예측 기간 (월)", 1, 12, 6, key="forecast_periods")
            forecast_method = st.selectbox(
                "예측 방법",
                ["Prophet", "Holt-Winters", "지수 이동평균 (EMA)"],
                index=0 if PROPHET_AVAILABLE else 2,
                key="forecast_method"
            )
        
        forecast_data = monthly[monthly["topic"] == selected_topic_forecast].sort_values("date")
        if len(forecast_data) >= 12:
            ts = forecast_data.set_index("date")["count"]
            
            try:
                with st.spinner(f"{forecast_method} 예측 계산 중..."):
                    if forecast_method == "Prophet" and PROPHET_AVAILABLE:
                        forecast_series = apply_smoothing_method(ts, "Prophet", periods=forecast_periods)
                        # Prophet은 예측값도 포함하므로 원본 길이만큼만 사용
                        forecast_series = forecast_series.iloc[:len(ts)]
                    elif forecast_method == "Holt-Winters":
                        forecast_series = apply_smoothing_method(ts, "Holt-Winters", seasonal_periods=12)
                    else:
                        forecast_series = apply_smoothing_method(ts, "지수 이동평균 (EMA)", span=6)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts.index,
                    y=ts.values,
                    name="실제값",
                    mode="lines+markers",
                    line=dict(color="blue", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_series.index,
                    y=forecast_series.values,
                    name=f"{forecast_method} 평활",
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                ))
                
                fig.update_layout(
                    title=f"{selected_topic_forecast} - {forecast_method} 예측",
                    xaxis_title="월",
                    yaxis_title="건수",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
            except Exception as e:
                st.error(f"예측 실패: {e}")
        else:
            st.warning("예측을 위해서는 최소 12개월 데이터가 필요합니다.")
    
    # 탭 8: 데이터 테이블
    with tab8:
        st.header("📋 데이터 테이블")
        
        table_option = st.selectbox(
            "테이블 선택",
            [
                "월별 시계열",
                "지역별 시계열",
                "주제+지역 조합",
                "분기별 시계열",
                "연도별 시계열",
                "성장률 분석",
                "점유율 분석",
                "누적 건수",
                "이상치",
            ],
            key="table_select"
        )
        
        with st.expander("📥 데이터 다운로드", expanded=False):
            download_format = st.radio("다운로드 형식", ["CSV", "Excel"], horizontal=True)
        
        if table_option == "월별 시계열":
            display_df = monthly_filtered
        elif table_option == "지역별 시계열":
            display_df = region_filtered
        elif table_option == "주제+지역 조합":
            display_df = topic_region_filtered
        elif table_option == "분기별 시계열":
            display_df = quarterly[quarterly["topic"].isin(selected_topics) if selected_topics else quarterly["topic"].isin(all_topics)]
        elif table_option == "연도별 시계열":
            display_df = yearly[yearly["topic"].isin(selected_topics) if selected_topics else yearly["topic"].isin(all_topics)]
        elif table_option == "성장률 분석":
            display_df = growth[growth["topic"].isin(selected_topics) if selected_topics else growth["topic"].isin(all_topics)]
        elif table_option == "점유율 분석":
            display_df = market_share[market_share["topic"].isin(selected_topics) if selected_topics else market_share["topic"].isin(all_topics)]
        elif table_option == "누적 건수":
            display_df = cumulative[cumulative["topic"].isin(selected_topics) if selected_topics else cumulative["topic"].isin(all_topics)]
        elif table_option == "이상치":
            display_df = anomalies[anomalies["topic"].isin(selected_topics) if selected_topics else True] if not anomalies.empty else pd.DataFrame()
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        if download_format == "CSV":
            csv = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"{table_option}.csv",
                mime="text/csv",
            )
        else:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Excel 다운로드",
                data=excel_data,
                file_name=f"{table_option}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
