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
import traceback
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
import concurrent.futures
import random

# --- 1. 环境与配置 ---
current_dir = os.getcwd()
sys.path.append(current_dir)

CONFIG = {
    "MIN_AMOUNT": 10000000,   # 🔥 降级：门槛降回1000万，防止误杀
    "MIN_PRICE": 2.0,         # 🔥 降级：门槛降回2元
    "MAX_WORKERS": 8,         # 8线程比较稳
    "DAYS_LOOKBACK": 150
}

HOT_CONCEPTS = [] 
HISTORY_FILE = "history_log.csv"

# --- 2. 宏观与基础数据 ---
def get_market_hot_spots():
    print(">>> [0/4] 扫描今日热门题材...")
    global HOT_CONCEPTS
    try:
        df = ak.stock_board_concept_name_em()
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"🔥 今日热点: {HOT_CONCEPTS}")
    except:
        HOT_CONCEPTS = []

def get_targets_robust():
    print(">>> [1/4] 获取股票列表并进行[强力清洗]...")
    try:
        df = ak.stock_zh_a_spot_em()
        
        # 🔥 1. 列名兼容性处理 (防止接口变动)
        # 很多时候是这里出了问题导致一列找不到
        col_map = {
            "最新价": "price", "最新价格": "price", 
            "成交额": "amount", "成交金额": "amount",
            "代码": "code", "名称": "name"
        }
        df.rename(columns=col_map, inplace=True)
        
        # 🔥 2. 强制类型转换 (防止全是字符串无法比较)
        df["price"] = pd.to_numeric(df["price"], errors='coerce')
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        
        # 3. 剔除无效数据
        df.dropna(subset=["price", "amount"], inplace=True)
        
        # 4. 基础过滤
        df = df[df["code"].str.startswith(("60", "00"))]
        df = df[~df['name'].str.contains('ST|退')]
        
        # 记录过滤前数量
        raw_len = len(df)
        
        # 5. 门槛过滤
        df = df[df["price"] >= CONFIG["MIN_PRICE"]]
        # 这里特别注意：如果是在早盘刚开盘，成交额可能很小，不要设太高
        df = df[df["amount"] > CONFIG["MIN_AMOUNT"]] 
        
        targets = df[["code", "name"]]
        print(f"✅ 数据清洗完成: 原始 {raw_len} -> 有效 {len(targets)} 只")
        
        if len(targets) == 0:
            print("❌ 警告：预过滤后数量为0！可能是akshare接口数据异常。")
            # 启动保底逻辑
            raise ValueError("Filtered to zero")
            
        return targets, "在线API"
        
    except Exception as e:
        print(f"⚠️ API数据异常: {e}，启动保底测试列表...")
        manual_list = [
            ["600519", "贵州茅台"], ["002594", "比亚迪"], ["601138", "工业富联"],
            ["000063", "中兴通讯"], ["600460", "士兰微"], ["300750", "宁德时代"]
        ]
        return pd.DataFrame(manual_list, columns=["code", "name"]), "保底列表"

def get_data_with_retry(code, start_date):
    time.sleep(random.uniform(0.05, 0.15))
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except: time.sleep(0.5)
    return None

def get_stock_catalysts(code):
    try:
        news_df = ak.stock_news_em(symbol=code)
        if not news_df.empty:
            title = news_df.iloc[0]['新闻标题']
            date = news_df.iloc[0]['发布时间']
            return f"[{date[5:10]}] {title}"
    except: pass
    return "无近期新闻"

# --- 3. 核心逻辑 (逻辑放宽，保证出结果) ---
def process_stock_logic(df, code, name):
    if len(df) < 60: return None # 放宽K线长度限制
    
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    df["vwap"] = df["amount"] / volume if "amount" in df.columns else (high + low + close) / 3

    # 指标计算
    df["pct_chg"] = close.pct_change() * 100
    today_pct = df["pct_chg"].iloc[-1]
    
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100
    ma60_slope = df["MA60"].iloc[-1] > df["MA60"].iloc[-5]

    bb_ind = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_ind.bollinger_hband()
    df["BB_Lower"] = bb_ind.bollinger_lband()
    df["BB_Mid"] = bb_ind.bollinger_mavg()
    df["BB_Width"] = bb_ind.bollinger_wband()
    df["BB_PctB"] = bb_ind.bollinger_pband()

    macd = MACD(close)
    df["DIF"] = macd.macd()
    df["DEA"] = macd.macd_signal()
    df["MACD_Bar"] = macd.macd_diff()
    
    kdj = StochasticOscillator(high, low, close)
    df["K"] = kdj.stoch()
    df["D"] = kdj.stoch_signal()
    df["J"] = 3 * df["K"] - 2 * df["D"]
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
    
    # 筹码与形态
    chip_signal = ""
    high_120 = df["high"].tail(120).max()
    low_120 = df["low"].tail(120).min()
    current_pos = (curr["close"] - low_120) / (high_120 - low_120 + 0.001)
    if current_pos < 0.4: # 放宽位置限制
        volatility = df["close"].tail(60).std() / df["close"].tail(60).mean()
        if volatility < 0.15: chip_signal = "🏆筹码密集" 

    patterns = []
    # 红肥绿瘦 (放宽比例)
    recent_20 = df.tail(20)
    vol_up = recent_20[recent_20['close'] > recent_20['open']]['volume'].sum()
    vol_down = recent_20[recent_20['close'] < recent_20['open']]['volume'].sum()
    if vol_up > vol_down * 1.3: patterns.append("🟥红肥绿瘦")
    
    # N字反包
    if (prev['close'] < prev['open']) and (curr['close'] > curr['open']) and (curr['close'] > prev['open']):
        patterns.append("⚡N字反包")
    
    pattern_str = " ".join(patterns)

    # 背离与状态
    div_signal = ""
    if curr["low"] == df["low"].tail(20).min():
        if curr["MACD_Bar"] > prev["MACD_Bar"]: div_signal = "💪MACD底背离"

    macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    macd_status = ""
    if macd_gold: macd_status = "🔥确认金叉"
    elif curr["DIF"] > curr["DEA"] and curr["DIF"] > 0: macd_status = "⛽多头趋势"
    elif curr["DIF"] < curr["DEA"] and (curr["DEA"] - curr["DIF"]) < 0.05 and curr["MACD_Bar"] > prev["MACD_Bar"]: macd_status = "🔔即将金叉"

    bb_state = ""
    if curr["BB_PctB"] > 1.0: bb_state = "🚀突破上轨"
    elif curr["BB_PctB"] < 0.0: bb_state = "📉跌破下轨"
    elif curr["BB_Width"] < 15: bb_state = "↔️极度收口" # 放宽带宽
    elif abs(curr["close"] - curr["BB_Mid"])/curr["BB_Mid"] < 0.02: bb_state = "🛡️中轨支撑"

    # === 选股策略 (恢复到宽容模式) ===
    signal_type = ""
    suggest_buy = curr["close"]
    stop_loss = curr["MA20"]
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    
    # 策略1: 龙回头
    if has_zt and curr["close"] > curr["MA60"]:
        if -5.0 < curr["BIAS20"] < 10.0: 
            signal_type = "🐉龙回头"
            stop_loss = round(curr["BB_Lower"], 2)
    # 策略2: 机构控盘 (放宽)
    if not signal_type and curr["close"] > curr["MA60"]:
        if curr["CMF"] > 0.05 and curr["ADX"] > 20: # 门槛降回正常值
            signal_type = "🏦机构控盘"
            suggest_buy = round(curr["vwap"], 2)
    # 策略3: 极度超跌
    if not signal_type and ((curr["RSI"] < 25) or div_signal): # 门槛降回25
        signal_type = "📉极度超跌"
        stop_loss = round(curr["low"] * 0.96, 2)
    # 策略4: 底部变盘
    if not signal_type and curr["close"] < curr["MA60"] * 1.2:
         if curr["BB_Width"] < 15:
            signal_type = "⚡底部变盘"

    # === 🔥 核心修改：评分机制降级 ===
    obv_txt = "流入" if curr["OBV"] > curr["OBV_MA10"] else "流出"
    # 🔥 此次不再因为资金流出直接 return None，而是仅仅作为扣分项
    
    score = 0
    reasons = []
    if signal_type: score += 1; reasons.append("策略")
    if chip_signal: score += 1; reasons.append("筹码")
    if pattern_str: score += 1; reasons.append("形态")
    if div_signal: score += 1; reasons.append("背离")
    if "金叉" in macd_status or "加油" in macd_status: score += 1; reasons.append("MACD")
    if "突破" in bb_state or "收口" in bb_state: score += 1; reasons.append("布林")
    
    news_info = get_stock_catalysts(code)
    is_hot = False
    for hot in HOT_CONCEPTS:
        if hot in news_info: is_hot = True; break
    if is_hot: score += 1; reasons.append("热点")

    # 🔥 门槛降级：只要有1分就入选，保证有结果
    if score < 1: return None
    
    resonance_str = "+".join(reasons)
    vol_ma5 = df["volume"].rolling(5).mean().iloc[-1]
    vol_ratio = round(curr["volume"] / vol_ma5, 2) if vol_ma5 > 0 else 0
    pct_str = f"{today_pct:+.2f}%"

    return {
        "代码": code,
        "名称": name,
        "现价": curr["close"],
        "今日涨跌": pct_str,
        "连续上榜": "",
        "共振因子": resonance_str,
        "信号类型": signal_type,
        "题材与利好": news_info,
        "筹码分布": chip_signal,
        "形态特征": pattern_str,
        "MACD预警": macd_status,
        "底背离": div_signal,
        "布林状态": bb_state,
        "BIAS%": round(curr["BIAS20"], 1),
        "CMF指标": round(curr["CMF"], 3),
        "RSI指标": round(curr["RSI"], 1),
        "J值": round(curr["J"], 1),
        "资金流向": obv_txt,
        "建议挂单": suggest_buy,
        "止损价": stop_loss,
        "量比": vol_ratio
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
        if streak == 2: streak_str = "🔥2连板"
        elif streak >= 3: streak_str = f"🚀{streak}连板"
        res['连续上榜'] = streak_str
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
    filename = f"严选_平衡版_{dt_str}.xlsx"
    
    if not data_list:
        pd.DataFrame([["无结果"]]).to_excel(filename)
        return filename

    df = pd.DataFrame(data_list)
    cols = ["代码", "名称", "现价", "今日涨跌", "连续上榜", "共振因子", "信号类型", "题材与利好", 
            "筹码分布", "形态特征", "MACD预警", "底背离", 
            "布林状态", "BIAS%", "CMF指标", "RSI指标", "J值",
            "资金流向", "建议挂单", "止损价", "量比"]
    
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols]
    
    df.sort_values(by=["连续上榜", "筹码分布"], ascending=[False, False], inplace=True)
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_red = Font(color="FF0000", bold=True)
    font_green = Font(color="008000", bold=True)
    font_purple = Font(color="800080", bold=True)
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
        if "连板" in streak_val:
            row[4].fill = fill_magenta; row[4].font = font_red
        
        row[5].font = Font(color="0000FF", bold=True)
        
        news_cell = row[7]
        news_cell.alignment = Alignment(wrap_text=True, vertical='center')
        news_cell.font = Font(size=9)
        for hot in HOT_CONCEPTS:
            if hot in str(news_cell.value):
                news_cell.font = Font(size=9, color="FF0000", bold=True)
                break
        
        if "低位密集" in str(row[8].value): row[8].font = font_red; row[8].fill = fill_yellow
        if "红肥" in str(row[9].value) or "N字" in str(row[9].value): row[9].font = font_red
        
        macd_val = str(row[10].value)
        if "即将" in macd_val: row[10].font = font_orange
        if "金叉" in macd_val or "加油" in macd_val: row[10].font = font_red; row[10].fill = fill_yellow
        if row[11].value: row[11].font = font_red
        
        bb_val = str(row[12].value)
        if "突破" in bb_val: row[12].font = font_red
        if "收口" in bb_val: row[12].font = font_orange

        cmf_val = row[14].value
        if isinstance(cmf_val, (int, float)):
            if cmf_val > 0.1: row[14].font = Font(color="FF0000", bold=True)
            elif cmf_val > 0: row[14].font = font_red

        rsi_val = row[15].value
        if isinstance(rsi_val, (int, float)):
            if rsi_val < 20: row[15].font = font_green; row[15].fill = fill_yellow
            elif rsi_val > 80: row[15].font = font_red

        # 🔥 资金流向高亮修正
        flow_cell = row[17]
        if "流入" in str(flow_cell.value):
            flow_cell.font = font_red
        else:
            flow_cell.font = font_green

    ws.column_dimensions['H'].width = 45 
    
    start_row = ws.max_row + 3
    title_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    sub_title_font = Font(name='微软雅黑', size=11, bold=True, color="000000")
    text_font = Font(name='微软雅黑', size=10)
    
    ws.cell(row=start_row, column=1, value="📘 平衡版选股指南").font = title_font
    start_row += 1
    ws.cell(row=start_row, column=1, value="💡 说明：").font = Font(bold=True)
    ws.cell(row=start_row, column=2, value="此版本为【保证结果版】。条件已放宽，资金流出不直接过滤，请人工参考[资金流向]列。").font = text_font
    start_row += 1
    
    strategies = [
        ("【🔥 连续上榜】", "含义：该股连续多日入选。2连板=确认走强；3连板=妖股气质。重点关注！"),
        ("【🐉 龙回头】", "逻辑：前期妖股+生命线支撑+极致缩量。操作：低吸博反抽。"),
        ("【🏦 机构控盘】", "逻辑：趋势向上+强资金(CMF>0.1)。操作：沿5日/10日线持有。"),
        ("【📉 极度超跌】", "逻辑：RSI<20 或 底背离。操作：左侧博反弹，快进快出。"),
        ("【⚡ 底部变盘】", "逻辑：布林带宽<10+资金异动。操作：往往是大行情起点。")
    ]
    for title, desc in strategies:
        ws.cell(row=start_row, column=1, value=title).font = Font(bold=True)
        ws.cell(row=start_row, column=2, value=desc).font = text_font
        start_row += 1

    wb.save(filename)
    print(f"✅ 结果已保存: {filename}")
    return filename

# --- 5. 主程序 ---
def analyze_one_stock(code, name, start_dt):
    try:
        df = get_data_with_retry(code, start_dt)
        if df is None: return None
        return process_stock_logic(df, code, name)
    except: return None

def main():
    print("=== A股共振严选 (数据修复+平衡版) ===")
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
                    print(f"  ★ 选中: {res['名称']} {res['今日涨跌']} [{res['共振因子']}]")
                    results.append(res)
            except: pass

    if results:
        results = update_history(results)
    
    print(f"\n耗时: {int(time.time() - start_time)}秒 | 选中 {len(results)} 只")
    save_and_beautify(results)

if __name__ == "__main__":
    main()
