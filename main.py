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
    "MIN_AMOUNT": 20000000,   # 最低成交额 2000万
    "MIN_PRICE": 2.5,         # 最低股价
    "MAX_WORKERS": 12,        # 12线程并发
    "DAYS_LOOKBACK": 150      # 数据回溯
}

HOT_CONCEPTS = [] 
HISTORY_FILE = "history_log.csv"

# --- 2. 获取市场热门板块 ---
def get_market_hot_spots():
    global HOT_CONCEPTS
    try:
        # 获取概念板块涨幅榜
        df = ak.stock_board_concept_name_em()
        # 取涨幅前 15 的板块作为今日热点
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"🔥 今日资金风口: {HOT_CONCEPTS}")
    except:
        HOT_CONCEPTS = []
        print("⚠️ 热点获取失败，将跳过热点匹配。")

def get_targets_robust():
    print(">>> [1/4] 获取A股全市场股票列表并预过滤...")
    try:
        df = ak.stock_zh_a_spot_em()
        
        # 兼容性处理
        col_map = {
            "最新价": "price", "最新价格": "price", 
            "成交额": "amount", "成交金额": "amount",
            "代码": "code", "名称": "name"
        }
        df.rename(columns=col_map, inplace=True)
        
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df.dropna(subset=["price", "amount"], inplace=True)
        
        # 基础过滤
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        
        # 门槛过滤
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]]
        
        targets = df[["code", "name"]]
        print(f"✅ 有效标的: {len(targets)} 只")
        return targets, "在线API"
        
    except Exception as e:
        print(f"⚠️ 数据获取异常: {e}")
        return pd.DataFrame(), "无结果"

def get_data_with_retry(code, start_date):
    time.sleep(random.uniform(0.01, 0.05)) # 轻微延迟防止封IP
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except: time.sleep(0.2)
    return None

def get_stock_catalysts(code):
    """获取个股最新题材/新闻"""
    try:
        # 这里依然调用新闻接口，因为这是获取个股当前最热属性最快的方法
        news_df = ak.stock_news_em(symbol=code)
        if not news_df.empty:
            title = news_df.iloc[0]['新闻标题']
            # 只返回标题，便于后续匹配热点
            return title
    except: pass
    return ""

# --- 3. 核心逻辑 (优化版) ---
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
    
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100

    bb_ind = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_ind.bollinger_hband()
    df["BB_Lower"] = bb_ind.bollinger_lband()
    df["BB_Width"] = bb_ind.bollinger_wband()
    df["BB_PctB"] = bb_ind.bollinger_pband()

    macd = MACD(close)
    df["DIF"] = macd.macd()
    df["DEA"] = macd.macd_signal()
    df["MACD_Bar"] = macd.macd_diff()
    
    kdj = StochasticOscillator(high, low, close)
    df["J"] = kdj.stoch() * 3 - kdj.stoch_signal() * 2
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["OBV"] = obv
    df["OBV_MA10"] = obv.rolling(10).mean()
    
    cmf = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    df["CMF"] = cmf
    adx_ind = ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx_ind.adx()

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # === 🔥 核心过滤 1: 资金流出直接删除 ===
    # 这是减少数量最直接的方法
    obv_txt = "流入" if curr["OBV"] > curr["OBV_MA10"] else "流出"
    if obv_txt == "流出": return None 

    # === 形态与筹码 ===
    chip_signal = ""
    high_120 = df["high"].tail(120).max()
    low_120 = df["low"].tail(120).min()
    current_pos = (curr["close"] - low_120) / (high_120 - low_120 + 0.001)
    
    # 筹码要求稍严：波动率低且位置低
    if current_pos < 0.35:
        volatility = df["close"].tail(60).std() / df["close"].tail(60).mean()
        if volatility < 0.12: chip_signal = "🏆筹码密集" 

    patterns = []
    
    # === 🔥 核心过滤 2: 优化“红肥绿瘦” ===
    # 之前是 1.3倍，现在改为 2.0倍，且要求 CMF>0 (真金白银) 且 股价站稳MA20 (趋势不坏)
    recent_20 = df.tail(20)
    vol_up = recent_20[recent_20['close'] > recent_20['open']]['volume'].sum()
    vol_down = recent_20[recent_20['close'] < recent_20['open']]['volume'].sum()
    
    if vol_up > vol_down * 2.0:  # 买盘量是卖盘量2倍以上
        if curr["close"] > curr["MA20"]: # 趋势未破位
            if curr["CMF"] > 0: # 资金确实是正向的
                patterns.append("🟥红肥绿瘦(优)")
    
    # N字反包
    if (prev['close'] < prev['open']) and (curr['close'] > curr['open']) and (curr['close'] > prev['open']):
        patterns.append("⚡N字反包")
    
    pattern_str = " ".join(patterns)

    # 背离与MACD
    div_signal = ""
    if curr["low"] == df["low"].tail(20).min():
        if curr["MACD_Bar"] > prev["MACD_Bar"] and curr["MACD_Bar"] < 0: div_signal = "💪MACD底背离"

    macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    macd_status = ""
    if macd_gold: macd_status = "🔥确认金叉"
    elif curr["DIF"] > curr["DEA"] and curr["DIF"] > 0 and curr["MACD_Bar"] > prev["MACD_Bar"]: macd_status = "⛽空中加油"
    elif curr["DIF"] < curr["DEA"] and (curr["DEA"] - curr["DIF"]) < 0.05 and curr["MACD_Bar"] > prev["MACD_Bar"]: macd_status = "🔔即将金叉"

    bb_state = ""
    if curr["BB_PctB"] > 1.0: bb_state = "🚀突破上轨"
    elif curr["BB_Width"] < 12: bb_state = "↔️极度收口" # 收口标准收紧

    # === 选股策略 (标准版) ===
    signal_type = ""
    suggest_buy = curr["close"]
    stop_loss = curr["MA20"]
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    
    # 1. 龙回头
    if has_zt and curr["close"] > curr["MA60"]:
        if curr["volume"] < df["volume"].tail(30).max() * 0.5:
            if -2.0 < curr["BIAS20"] < 8.0: 
                signal_type = "🐉龙回头"
                stop_loss = round(curr["BB_Lower"], 2)
    # 2. 机构控盘
    if not signal_type and curr["close"] > curr["MA60"]:
        if curr["CMF"] > 0.08 and curr["ADX"] > 22 and curr["BIAS20"] < 15.0:
            signal_type = "🏦机构控盘"
            suggest_buy = round(curr["vwap"], 2)
    # 3. 极度超跌
    if not signal_type and ((curr["RSI"] < 22) or div_signal):
        signal_type = "📉极度超跌"
        stop_loss = round(curr["low"] * 0.96, 2)
    # 4. 底部变盘
    if not signal_type and curr["close"] < curr["MA60"] * 1.15 and curr["BB_Width"] < 12:
         if macd_gold or curr["CMF"] > 0.12:
            signal_type = "⚡底部变盘"

    # === 评分系统 ===
    score = 0
    reasons = []
    if signal_type: score += 1; reasons.append("策略")
    if chip_signal: score += 1; reasons.append("筹码")
    if pattern_str: score += 1; reasons.append("形态")
    if div_signal: score += 1; reasons.append("背离")
    if "金叉" in macd_status or "加油" in macd_status: score += 1; reasons.append("MACD")
    if "突破" in bb_state or "收口" in bb_state: score += 1; reasons.append("布林")
    
    # 热点匹配
    news_title = get_stock_catalysts(code)
    hot_matched = ""
    for hot in HOT_CONCEPTS:
        if hot in news_title: 
            hot_matched = hot # 记录匹配到的热点
            score += 1
            reasons.append("热点")
            break
    
    # 如果没有匹配到热点，这里只保留标题作为参考，不加分
    display_concept = f"🔥{hot_matched}" if hot_matched else ""

    # === 🔥 最终过滤 ===
    # 1. 必须资金流入 (前面已 check)
    # 2. 至少满足 1 项硬性条件 (策略/筹码/形态/背离/热点)
    if score < 1: return None
    
    # 3. 如果只是普通的形态(如只是红肥绿瘦)但没有策略信号，要求更严
    if not signal_type and score < 2: return None

    resonance_str = "+".join(reasons)
    pct_str = f"{today_pct:+.2f}%"

    return {
        "代码": code,
        "名称": name,
        "现价": curr["close"],
        "今日涨跌": pct_str,
        "共振因子": resonance_str,
        "信号类型": signal_type,
        "热门概念": display_concept, # 只显示匹配到的热点
        "筹码分布": chip_signal,
        "形态特征": pattern_str,
        "MACD预警": macd_status,
        "底背离": div_signal,
        "布林状态": bb_state,
        "CMF指标": round(curr["CMF"], 3),
        "RSI指标": round(curr["RSI"], 1),
        "J值": round(curr["J"], 1),
        "建议挂单": suggest_buy,
        "止损价": stop_loss
    }

# --- 历史记录与去重 ---
def update_history(current_results):
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)
            hist_df['date'] = hist_df['date'].astype(str)
        else:
            hist_df = pd.DataFrame(columns=["date", "code"])
    except:
        hist_df = pd.DataFrame(columns=["date", "code"])

    hist_df = hist_df[hist_df['date'] != today_str]
    sorted_dates = sorted(hist_df['date'].unique(), reverse=True)
    processed_results = []
    new_rows = []
    
    for res in current_results:
        code = res['code'] if 'code' in res else res['代码']
        streak = 1
        for d in sorted_dates:
            if not hist_df[(hist_df['date'] == d) & (hist_df['code'] == str(code))].empty:
                streak += 1
            else: break
        
        streak_str = "首榜"
        if streak == 2: streak_str = "🔥2连"
        elif streak >= 3: streak_str = f"🚀{streak}连"
        res['连续'] = streak_str
        processed_results.append(res)
        new_rows.append({"date": today_str, "code": str(code)})

    if new_rows:
        hist_df = pd.concat([hist_df, pd.DataFrame(new_rows)], ignore_index=True)
    try:
        hist_df.to_csv(HISTORY_FILE, index=False)
    except: pass
    return processed_results

# --- Excel 输出 ---
def save_and_beautify(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"精简优化版_{dt_str}.xlsx"
    
    if not data_list:
        pd.DataFrame([["无结果"]]).to_excel(filename)
        print("筛选结果为空。")
        return filename

    df = pd.DataFrame(data_list)
    cols = ["代码", "名称", "现价", "今日涨跌", "连续", "共振因子", "信号类型", "热门概念", 
            "筹码分布", "形态特征", "MACD预警", "底背离", 
            "布林状态", "CMF指标", "RSI指标", "J值", "建议挂单", "止损价"]
    
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols]
    
    df.sort_values(by=["连续", "共振因子"], ascending=[False, False], inplace=True)
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_red = Font(color="FF0000", bold=True)
    font_green = Font(color="008000", bold=True)
    font_orange = Font(color="FF8C00", bold=True)
    fill_yellow = PatternFill("solid", fgColor="FFF2CC")
    fill_magenta = PatternFill("solid", fgColor="FFC7CE") 
    
    for cell in ws[1]:
        cell.fill = fill_blue
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')
        
    for row in ws.iter_rows(min_row=2):
        chg_val = str(row[3].value)
        if "+" in chg_val: row[3].font = font_red
        elif "-" in chg_val: row[3].font = font_green
            
        streak_val = str(row[4].value)
        row[4].font = Font(bold=True)
        if "连" in streak_val:
            row[4].fill = fill_magenta; row[4].font = font_red
        
        row[5].font = Font(color="0000FF", bold=True)
        
        # 热门概念：只有匹配到的才显示，且标红
        hot_val = str(row[7].value)
        if "🔥" in hot_val:
            row[7].font = font_red
            row[7].fill = fill_yellow
        
        if "低位密集" in str(row[8].value): row[8].font = font_red; row[8].fill = fill_yellow
        if "红肥" in str(row[9].value): row[9].font = font_red
        
        macd_val = str(row[10].value)
        if "金叉" in macd_val or "加油" in macd_val: row[10].font = font_red
        if row[11].value: row[11].font = font_red
        
        cmf_val = row[13].value
        if isinstance(cmf_val, (int, float)) and cmf_val > 0.1: row[13].font = font_red

    ws.column_dimensions['H'].width = 25 
    
    start_row = ws.max_row + 3
    title_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    ws.cell(row=start_row, column=1, value="📘 精简版指南").font = title_font
    start_row += 1
    
    dictionary = [
        ("【筛选标准】", "已自动剔除资金流出(OBV)股票。"),
        ("【红肥绿瘦(优)】", "优化算法：买盘量是卖盘量2倍以上 + 站稳MA20 + 资金流入。"),
        ("【热门概念】", "仅显示命中今日涨幅前15板块的股票，未命中留空。"),
        ("【连续】", "2连/3连代表该股连续多日入选，确定性更高。")
    ]
    for title, desc in dictionary:
        ws.cell(row=start_row, column=1, value=title).font = Font(bold=True)
        ws.cell(row=start_row, column=2, value=desc).font = Font(size=10)
        start_row += 1

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
    print("=== A股严选 (精简优化版) ===")
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
                    # 只有当有结果时才打印，减少刷屏
                    print(f"  ★ 选中: {res['名称']} {res['今日涨跌']}")
                    results.append(res)
            except: pass

    if results:
        results = update_history(results)
    
    print(f"\n耗时: {int(time.time() - start_time)}秒 | 选中 {len(results)} 只")
    save_and_beautify(results)

if __name__ == "__main__":
    main()
