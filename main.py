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
import random  # <--- 新增随机库

# --- 1. 环境与配置 ---
current_dir = os.getcwd()
sys.path.append(current_dir)

CONFIG = {
    "MIN_AMOUNT": 20000000,   # 最低成交额
    "MIN_PRICE": 2.5,         # 最低股价
    "MAX_WORKERS": 12,        # 🔥 [安全修正] 改为12线程 (兼顾速度与防封)
    "DAYS_LOOKBACK": 150      # 数据回溯
}

HOT_CONCEPTS = [] 
HISTORY_FILE = "history_log.csv"

# --- 2. 宏观与基础数据 ---
def get_market_hot_spots():
    print(">>> [0/4] 扫描今日热门题材与政策导向...")
    global HOT_CONCEPTS
    try:
        df = ak.stock_board_concept_name_em()
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"🔥 今日热点: {HOT_CONCEPTS}")
    except:
        HOT_CONCEPTS = []

def get_targets_robust():
    """
    🔥 [提速核心]：利用实时数据[强力预过滤]，减少50%以上的无效HTTP请求
    """
    print(">>> [1/4] 获取A股全市场股票列表并进行[预过滤]...")
    try:
        # 获取实时行情
        df = ak.stock_zh_a_spot_em()
        
        # 1. 基础过滤
        df = df[df["代码"].str.startswith(("60", "00"))]
        df = df[~df['名称'].str.contains('ST|退')]
        
        # 2. 🔥 预过滤：价格 (直接剔除低价股，不再请求历史数据)
        df = df[df["最新价"] >= CONFIG["MIN_PRICE"]]
        
        # 3. 🔥 预过滤：成交额 (剔除僵尸股)
        # 此时如果是盘中，成交额是动态的；如果是盘后，是全天的。
        # 只要成交额极低(例如<500万)，说明今天全天没戏，直接剔除。
        df = df[df["成交额"] > 5000000] 

        targets = df[["代码", "名称"]]
        targets.columns = ["code", "name"]
        
        print(f"✅ 预过滤完成：有效活跃标的 {len(targets)} 只 (已剔除 {5000-len(targets)} 只垃圾股)")
        return targets, "在线API"
    except Exception as e:
        print(f"⚠️ 预过滤失败，使用保底列表: {e}")
        manual_list = [["600519", "贵州茅台"], ["002594", "比亚迪"]]
        return pd.DataFrame(manual_list, columns=["code", "name"]), "保底列表"

def get_data_with_retry(code, start_date):
    # 🔥 [防封核心]：每次请求前随机休眠 0.05~0.2秒
    # 这会稍微降低速度，但能极大降低 IP 被封概率
    time.sleep(random.uniform(0.05, 0.2)) 
    
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=5)
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except: 
            time.sleep(0.5) # 出错多睡一会
    return None

def get_stock_catalysts(code):
    try:
        # 新闻接口请求量小，且只针对入选股，风险较低
        news_df = ak.stock_news_em(symbol=code)
        if not news_df.empty:
            title = news_df.iloc[0]['新闻标题']
            date = news_df.iloc[0]['发布时间']
            return f"[{date[5:10]}] {title}"
    except: pass
    return "无近期新闻"

# --- 3. 核心逻辑 (逻辑不变) ---
def process_stock_logic(df, code, name):
    if len(df) < 120: return None
    
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    df["vwap"] = df["amount"] / volume if "amount" in df.columns else (high + low + close) / 3

    # 二次门槛确认
    curr_amount = df["amount"].iloc[-1] if "amount" in df.columns else (close.iloc[-1] * volume.iloc[-1])
    if curr_amount < CONFIG["MIN_AMOUNT"]: return None
    
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
    if current_pos < 0.35:
        volatility = df["close"].tail(60).std() / df["close"].tail(60).mean()
        if volatility < 0.1: chip_signal = "🏆低位单峰密集" 

    patterns = []
    recent_20 = df.tail(20)
    vol_up = recent_20[recent_20['close'] > recent_20['open']]['volume'].sum()
    vol_down = recent_20[recent_20['close'] < recent_20['open']]['volume'].sum()
    if vol_up > vol_down * 1.8 and curr["CMF"] > 0: patterns.append("🟥红肥绿瘦")
    if curr["volume"] < df["volume"].tail(100).min() * 1.2 and current_pos < 0.2: patterns.append("💤地量见地价")
    if (prev['close'] < prev['open']) and (curr['close'] > curr['open']) and (curr['close'] > prev['open']) and (curr['volume'] > prev['volume']):
        patterns.append("⚡N字反包")
    recent_5 = df.tail(5)
    if (recent_5['close'] > recent_5['MA5']).all() and (recent_5['close'].iloc[-1] > recent_5['close'].iloc[0]):
        patterns.append("🐜蚂蚁上树")
    pattern_str = " ".join(patterns)

    # 背离与状态
    div_signal = ""
    if curr["low"] == df["low"].tail(20).min():
        if curr["MACD_Bar"] > prev["MACD_Bar"] and curr["MACD_Bar"] < 0: div_signal = "💪MACD底背离"

    macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    macd_status = ""
    if macd_gold: macd_status = "🔥确认金叉"
    elif curr["DIF"] > curr["DEA"] and curr["DIF"] > 0 and curr["MACD_Bar"] > prev["MACD_Bar"]: macd_status = "⛽空中加油"
    elif curr["DIF"] < curr["DEA"] and (curr["DEA"] - curr["DIF"]) < 0.05 and curr["MACD_Bar"] > prev["MACD_Bar"]: macd_status = "🔔即将金叉"
    else: macd_status = "多头" if curr["DIF"] > curr["DEA"] else "空头"

    bb_state = ""
    if curr["BB_PctB"] > 1.0: bb_state = "🚀突破上轨"
    elif curr["BB_PctB"] < 0.0: bb_state = "📉跌破下轨"
    elif curr["BB_Width"] < 10: bb_state = "↔️极度收口"
    elif abs(curr["close"] - curr["BB_Mid"])/curr["BB_Mid"] < 0.015: bb_state = "🛡️中轨支撑"

    # === 选股策略 ===
    signal_type = ""
    suggest_buy = curr["close"]
    stop_loss = curr["MA20"]
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    
    # 策略1: 龙回头
    if has_zt and curr["close"] > curr["MA60"]:
        if curr["volume"] < df["volume"].tail(30).max() * 0.5:
            if -2.0 < curr["BIAS20"] < 6.0: 
                signal_type = "🐉龙回头"
                stop_loss = round(curr["BB_Lower"], 2)
    # 策略2: 机构控盘
    if not signal_type and curr["close"] > curr["MA60"] and ma60_slope:
        if curr["CMF"] > 0.1 and curr["ADX"] > 25 and curr["BIAS20"] < 12.0:
            signal_type = "🏦机构控盘"
            suggest_buy = round(curr["vwap"], 2)
    # 策略3: 极度超跌
    if not signal_type and ((curr["RSI"] < 20) or div_signal):
        signal_type = "📉极度超跌"
        stop_loss = round(curr["low"] * 0.96, 2)
    # 策略4: 底部变盘
    if not signal_type and curr["close"] < curr["MA60"] * 1.15 and curr["BB_Width"] < 10:
         if macd_gold or curr["CMF"] > 0.15:
            signal_type = "⚡底部变盘"

    # === 风控与评分 ===
    obv_txt = "流入" if curr["OBV"] > curr["OBV_MA10"] else "流出"
    if obv_txt == "流出": return None 

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

    if score < 2: return None
    
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
            else:
                break
        
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
        print(f"✅ 历史记录已更新 (已自动去重): {HISTORY_FILE}")
    except: pass

    return processed_results

# --- 4. Excel 美化 ---
def save_and_beautify(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"严选_安全极速版_{dt_str}.xlsx"
    
    if not data_list:
        pd.DataFrame([["无结果"]]).to_excel(filename)
        return filename

    df = pd.DataFrame(data_list)
    cols = ["代码", "名称", "现价", "今日涨跌", "连续上榜", "共振因子", "信号类型", "题材与利好", 
            "筹码分布", "形态特征", "MACD预警", "底背离", 
            "布林状态", "BIAS%", 
            "CMF指标", "RSI指标", "J值",
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

        j_val = row[16].value
        if isinstance(j_val, (int, float)):
            if j_val < 0: row[16].font = font_green; row[16].fill = fill_yellow
            elif j_val > 100: row[16].font = font_red

    ws.column_dimensions['H'].width = 45 
    
    start_row = ws.max_row + 3
    title_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    sub_title_font = Font(name='微软雅黑', size=11, bold=True, color="000000")
    text_font = Font(name='微软雅黑', size=10)
    
    ws.cell(row=start_row, column=1, value="📘 严选策略实战指南").font = title_font
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

    start_row += 1
    ws.cell(row=start_row, column=1, value="📖 核心释义字典").font = title_font
    start_row += 1
    dictionary = [
        ("【CMF指标】", "资金流量。>0代表资金流入，>0.1代表主力强控盘(红粗)。负值代表流出。"),
        ("【RSI指标】", "相对强弱。20以下为超卖(底)，80以上为超买(顶)。"),
        ("【J值】", "KDJ灵敏线。小于0代表极度超跌，随时可能反弹。"),
        ("【今日涨跌】", "红色代表今日上涨，绿色代表下跌。"),
        ("【共振因子】", "显示该股满足的核心条件。满足条件越多，确定性越高。"),
        ("【筹码分布】", "🏆低位单峰密集：主力吸筹完成，极度稀缺的牛股形态。"),
        ("【形态特征】", "🟥红肥绿瘦：资金运作；⚡N字反包：强势洗盘。"),
        ("【MACD预警】", "🔔即将金叉：鸭子张嘴(左侧)；⛽空中加油：上涨中继。"),
        ("【布林状态】", "↔️极度收口：变盘前兆；🚀突破上轨：主升浪。"),
        ("【止损价】", "⛔ 跌破此价格必须无条件卖出！")
    ]
    for title, desc in dictionary:
        ws.cell(row=start_row, column=1, value=title).font = sub_title_font
        ws.cell(row=start_row, column=2, value=desc).font = text_font
        ws.cell(row=start_row, column=2).alignment = Alignment(wrap_text=True)
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
    print("=== A股共振严选 (安全极速版) ===")
    get_market_hot_spots()
    start_time = time.time()
    targets, source_name = get_targets_robust()
    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    
    print(f"[{source_name}] 待扫描: {len(targets)} 只 | 启动 {CONFIG['MAX_WORKERS']} 线程...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {executor.submit(analyze_one_stock, r['code'], r['name'], start_dt): r['code'] for _, r in targets.iterrows()}
        count = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1
            if count % 100 == 0: print(f"进度: {count}/{len(targets)} ...")
            try:
                res = future.result()
                if res:
                    print(f"  ★ 严选: {res['名称']} {res['今日涨跌']} [{res['共振因子']}]")
                    results.append(res)
            except: pass

    if results:
        print("\n正在处理历史记录与去重...")
        results = update_history(results)
    
    print(f"\n耗时: {int(time.time() - start_time)}秒 | 严选出 {len(results)} 只精品")
    save_and_beautify(results)
    
    if not any(f.endswith('.xlsx') for f in os.listdir('.')):
        pd.DataFrame([["无"]]).to_excel(f"保底_{datetime.now().strftime('%H%M')}.xlsx")

if __name__ == "__main__":
    main()
