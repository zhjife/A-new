import akshare as ak
import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import os
import time
import sys
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
import concurrent.futures
import random

# --- 1. 环境与配置 ---
current_dir = os.getcwd()
sys.path.append(current_dir)

CONFIG = {
    "MIN_AMOUNT": 20000000,   # 最低成交额
    "MIN_PRICE": 2.5,         # 最低股价
    "MAX_WORKERS": 12,        # 线程数
    "DAYS_LOOKBACK": 150      # 回溯天数
}

HOT_CONCEPTS = [] 
HISTORY_FILE = "history_log.csv"

# --- 2. 基础数据获取 ---
def get_market_hot_spots():
    global HOT_CONCEPTS
    try:
        df = ak.stock_board_concept_name_em()
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"🔥 今日风口: {HOT_CONCEPTS}")
    except:
        HOT_CONCEPTS = []

def get_targets_robust():
    print(">>> [1/4] 获取全市场股票并预过滤...")
    try:
        df = ak.stock_zh_a_spot_em()
        col_map = {"最新价": "price", "最新价格": "price", "成交额": "amount", "成交金额": "amount", "代码": "code", "名称": "name"}
        df.rename(columns=col_map, inplace=True)
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df.dropna(subset=["price", "amount"], inplace=True)
        
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]]
        
        targets = df[["code", "name"]]
        print(f"✅ 有效标的: {len(targets)} 只")
        return targets, "在线API"
    except Exception as e:
        print(f"⚠️ 异常: {e}")
        return pd.DataFrame(), "无结果"

def get_data_with_retry(code, start_date):
    time.sleep(random.uniform(0.01, 0.05))
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except: time.sleep(0.2)
    return None

def get_stock_catalysts(code):
    try:
        news_df = ak.stock_news_em(symbol=code)
        if not news_df.empty:
            return news_df.iloc[0]['新闻标题']
    except: pass
    return ""

# --- 3. 核心逻辑 (金叉必选版) ---
def process_stock_logic(df, code, name):
    if len(df) < 100: return None
    
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    df["vwap"] = df["amount"] / volume if "amount" in df.columns else (high + low + close) / 3

    # 计算指标
    df["pct_chg"] = close.pct_change() * 100
    today_pct = df["pct_chg"].iloc[-1]
    pct_3day = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100 if len(close) > 4 else 0
    
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100

    bb_ind = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_ind.bollinger_hband()
    df["BB_Lower"] = bb_ind.bollinger_lband()
    df["BB_Width"] = bb_ind.bollinger_wband()
    df["BB_PctB"] = bb_ind.bollinger_pband()

    # MACD
    macd = MACD(close)
    df["DIF"] = macd.macd()
    df["DEA"] = macd.macd_signal()
    df["MACD_Bar"] = macd.macd_diff()
    
    # KDJ (你需要K和D来判断金叉)
    kdj = StochasticOscillator(high, low, close)
    df["K"] = kdj.stoch()
    df["D"] = kdj.stoch_signal()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    
    # RSI & OBV & CMF
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    df["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["OBV_MA10"] = df["OBV"].rolling(10).mean()
    df["CMF"] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    df["ADX"] = ADXIndicator(high, low, close, window=14).adx()

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # ================================
    # 🔥 1. 金叉熔断机制 (Gold Cross Check)
    # ================================
    
    # MACD 金叉: 昨天死叉(DIF<DEA) -> 今天金叉(DIF>DEA)
    is_macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    
    # KDJ 金叉: 昨天 J<K -> 今天 J>K (且位置不能太高, J<80)
    is_kdj_gold = (prev["J"] < prev["K"]) and (curr["J"] > curr["K"]) and (curr["J"] < 80)
    
    # 【核心过滤】：如果一个金叉都没有，直接淘汰！
    if not (is_macd_gold or is_kdj_gold):
        return None

    # 标记金叉类型
    cross_status = ""
    if is_macd_gold and is_kdj_gold: cross_status = "⚡双金叉共振"
    elif is_macd_gold: cross_status = "🔥MACD金叉"
    elif is_kdj_gold: cross_status = "📈KDJ金叉"

    # ================================
    # 2. 趋势与资金过滤器 (Filters)
    # ================================
    
    # 2.1 J值熔断 (防止高位假金叉)
    if curr["J"] > 100: return None

    # 2.2 资金流向必杀 (OBV流出剔除)
    if curr["OBV"] <= curr["OBV_MA10"]: return None

    # 2.3 CMF 加速门槛 (正值且递增)
    if curr["CMF"] < 0.05: return None
    if curr["CMF"] <= prev["CMF"]: return None # 必须加速流入

    # ================================
    # 3. 筹码与形态 (共振逻辑)
    # ================================
    
    chip_signal = ""
    # 筹码: 股价低位 + 波动率低
    high_120 = df["high"].tail(120).max()
    low_120 = df["low"].tail(120).min()
    current_pos = (curr["close"] - low_120) / (high_120 - low_120 + 0.001)
    
    if current_pos < 0.4:
        volatility = df["close"].tail(60).std() / df["close"].tail(60).mean()
        if volatility < 0.13: chip_signal = "🏆筹码密集" 

    patterns = []
    # 红肥绿瘦 (资金强推)
    recent_20 = df.tail(20)
    vol_up = recent_20[recent_20['close'] > recent_20['open']]['volume'].sum()
    vol_down = recent_20[recent_20['close'] < recent_20['open']]['volume'].sum()
    if vol_up > vol_down * 2.0 and curr["close"] > curr["MA20"]:
        patterns.append("🟥红肥绿瘦")
    # N字反包
    if (prev['close'] < prev['open']) and (curr['close'] > curr['open']) and (curr['close'] > prev['open']):
        patterns.append("⚡N字反包")
    
    pattern_str = " ".join(patterns)

    # ================================
    # 4. 策略判定
    # ================================
    signal_type = ""
    suggest_buy = curr["close"]
    stop_loss = curr["MA20"]
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    
    # 策略: 龙回头
    if has_zt and curr["close"] > curr["MA60"] and curr["volume"] < df["volume"].tail(30).max() * 0.5:
        if -2.0 < curr["BIAS20"] < 8.0: signal_type = "🐉龙回头"; stop_loss = round(curr["BB_Lower"], 2)
    # 策略: 机构控盘
    if not signal_type and curr["close"] > curr["MA60"] and curr["ADX"] > 25:
        signal_type = "🏦机构控盘"; suggest_buy = round(curr["vwap"], 2)
    # 策略: 极度超跌
    div_signal = "💪底背离" if (curr["low"] == df["low"].tail(20).min() and curr["MACD_Bar"] > prev["MACD_Bar"]) else ""
    if not signal_type and ((curr["RSI"] < 20) or div_signal):
        signal_type = "📉极度超跌"; stop_loss = round(curr["low"] * 0.96, 2)
    # 策略: 底部变盘
    if not signal_type and curr["close"] < curr["MA60"] * 1.15 and curr["BB_Width"] < 12:
        signal_type = "⚡底部变盘"

    # ================================
    # 5. 最终筛选 (至少满足一项核心优势)
    # ================================
    
    # 逻辑: 已经满足了[金叉] + [CMF加速] + [OBV流入] + [J<100]
    # 现在只要有 [策略] 或者 [筹码+形态共振] 即可入选
    
    has_strategy = bool(signal_type)
    has_resonance = bool(chip_signal and pattern_str) # 筹码和形态必须同时具备
    
    if not (has_strategy or has_resonance):
        return None

    # 热点
    news_title = get_stock_catalysts(code)
    hot_matched = ""
    for hot in HOT_CONCEPTS:
        if hot in news_title: 
            hot_matched = hot; break
    display_concept = f"🔥{hot_matched}" if hot_matched else ""

    reasons = []
    if signal_type: reasons.append("策略")
    if has_resonance: reasons.append("筹/形共振")
    if hot_matched: reasons.append("热点")
    if cross_status == "⚡双金叉共振": reasons.append("双金叉")
    
    resonance_str = "+".join(reasons)
    bb_state = "🚀突破上轨" if curr["BB_PctB"] > 1.0 else ("↔️极度收口" if curr["BB_Width"] < 12 else "")

    return {
        "代码": code,
        "名称": name,
        "现价": curr["close"],
        "今日涨跌": f"{today_pct:+.2f}%",
        "3日涨跌": f"{pct_3day:+.2f}%",
        "连续": "",
        "金叉信号": cross_status, # 新增列
        "共振因子": resonance_str,
        "信号类型": signal_type,
        "热门概念": display_concept,
        "筹码分布": chip_signal,
        "形态特征": pattern_str,
        "今日CMF": round(curr["CMF"], 3),
        "昨日CMF": round(prev["CMF"], 3),
        "RSI指标": round(curr["RSI"], 1),
        "J值": round(curr["J"], 1),
        "建议挂单": suggest_buy,
        "止损价": stop_loss
    }

# --- 历史与输出模块 ---
def update_history(current_results):
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)
            hist_df['date'] = hist_df['date'].astype(str)
        else: hist_df = pd.DataFrame(columns=["date", "code"])
    except: hist_df = pd.DataFrame(columns=["date", "code"])

    hist_df = hist_df[hist_df['date'] != today_str]
    sorted_dates = sorted(hist_df['date'].unique(), reverse=True)
    processed_results = []
    new_rows = []
    
    for res in current_results:
        code = res['code'] if 'code' in res else res['代码']
        streak = 1
        for d in sorted_dates:
            if not hist_df[(hist_df['date'] == d) & (hist_df['code'] == str(code))].empty: streak += 1
            else: break
        
        res['连续'] = f"🔥{streak}连" if streak >= 2 else "首榜"
        processed_results.append(res)
        new_rows.append({"date": today_str, "code": str(code)})

    if new_rows: hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
    try: hist_df.to_csv(HISTORY_FILE, index=False)
    except: pass
    return processed_results

def save_and_beautify(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"金叉严选_{dt_str}.xlsx"
    
    if not data_list:
        pd.DataFrame([["条件极严(需金叉+CMF递增+共振)，今日无股入选"]]).to_excel(filename)
        return filename

    df = pd.DataFrame(data_list)
    # 调整列顺序
    cols = ["代码", "名称", "现价", "今日涨跌", "3日涨跌", "金叉信号", "连续", "共振因子", 
            "信号类型", "热门概念", "今日CMF", "昨日CMF", "筹码分布", "形态特征", 
            "RSI指标", "J值", "建议挂单", "止损价"]
    
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols]
    
    df.sort_values(by=["连续", "今日CMF"], ascending=[False, False], inplace=True)
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_red = Font(color="FF0000", bold=True)
    font_green = Font(color="008000", bold=True)
    font_gold = Font(color="FF8C00", bold=True)
    fill_yellow = PatternFill("solid", fgColor="FFF2CC")
    
    for cell in ws[1]:
        cell.fill = fill_blue
        cell.font = header_font
    
    for row in ws.iter_rows(min_row=2):
        # 涨跌幅
        for idx in [3, 4]:
            val = str(row[idx].value)
            if "+" in val: row[idx].font = font_red
            elif "-" in val: row[idx].font = font_green
        
        # 🔥 金叉信号 (F列)
        cross_val = str(row[5].value)
        if "双金叉" in cross_val: 
            row[5].font = font_red
            row[5].fill = fill_yellow
        elif "MACD" in cross_val or "KDJ" in cross_val:
            row[5].font = font_gold

        # 连板
        if "连" in str(row[6].value): row[6].font = font_red; row[6].fill = fill_yellow
        # CMF对比
        if isinstance(row[10].value, (int, float)): row[10].font = font_red

    ws.column_dimensions['F'].width = 15 # 金叉列
    
    # 指南
    start_row = ws.max_row + 3
    ws.cell(row=start_row, column=1, value="📘 金叉严选逻辑").font = Font(bold=True, color="0000FF")
    guides = [
        ("【必须金叉】", "要求：今日必须发生 MACD金叉 或 KDJ金叉。拒绝只有趋势但无买点的股票。"),
        ("【CMF加速】", "要求：今日CMF > 昨日CMF，且 > 0.05。资金必须加速入场。"),
        ("【形态共振】", "如果依靠形态入选，必须[筹码]与[形态]同时具备。"),
        ("【J值熔断】", "J > 100 不选，防止追高被套。")
    ]
    for i, (t, d) in enumerate(guides):
        ws.cell(row=start_row+1+i, column=1, value=t).font = Font(bold=True)
        ws.cell(row=start_row+1+i, column=2, value=d)

    wb.save(filename)
    print(f"✅ 结果已保存: {filename}")
    return filename

def analyze_one_stock(code, name, start_dt):
    try:
        df = get_data_with_retry(code, start_dt)
        if df is None: return None
        return process_stock_logic(df, code, name)
    except: return None

def main():
    print("=== A股严选 (双金叉+CMF加速版) ===")
    get_market_hot_spots()
    start_time = time.time()
    targets, source_name = get_targets_robust()
    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    
    print(f"[{source_name}] 待扫描: {len(targets)} 只 | 启动 {CONFIG['MAX_WORKERS']} 线程...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(analyze_one_stock, r['code'], r['name'], start_dt): r['code'] for _, r in targets.iterrows()}
        count = 0
        total = len(targets)
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1
            if count % 100 == 0: print(f"进度: {count}/{total} ...")
            try:
                res = future.result()
                if res:
                    print(f"  ★ 严选: {res['名称']} [{res['金叉信号']}] CMF↑")
                    results.append(res)
            except: pass

    if results: results = update_history(results)
    
    print(f"\n耗时: {int(time.time() - start_time)}秒 | 选中 {len(results)} 只")
    save_and_beautify(results)

if __name__ == "__main__":
    main()
