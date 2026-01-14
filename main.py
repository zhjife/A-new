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

# --- 1. 环境与配置 ---
current_dir = os.getcwd()
sys.path.append(current_dir)

CONFIG = {
    "MIN_AMOUNT": 15000000,   # 最低成交额
    "MIN_PRICE": 2.0,         # 最低股价
    "MAX_WORKERS": 4,         # 线程数
    "DAYS_LOOKBACK": 150      # 数据回溯天数
}

HOT_CONCEPTS = [] # 全局热点存储

# --- 2. 宏观与基础数据 ---
def get_market_hot_spots():
    """获取市场热点"""
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
    print(">>> [1/4] 获取A股全市场股票列表...")
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "名称"]]
        df.columns = ["code", "name"]
        targets = df[df["code"].str.startswith(("60", "00"))]
        targets = targets[~targets['name'].str.contains('ST|退')]
        return targets, "在线API"
    except:
        manual_list = [["600519", "贵州茅台"], ["002594", "比亚迪"]]
        return pd.DataFrame(manual_list, columns=["code", "name"]), "保底列表"

def get_data_with_retry(code, start_date):
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=8)
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

# --- 3. 核心逻辑 (增加资金流出过滤) ---
def process_stock_logic(df, code, name):
    # === A. 基础清洗 ===
    if len(df) < 100: return None
    rename_dict = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    df["vwap"] = df["amount"] / volume if "amount" in df.columns else (high + low + close) / 3

    # === B. 硬性门槛 ===
    curr_amount = df["amount"].iloc[-1] if "amount" in df.columns else (close.iloc[-1] * volume.iloc[-1])
    if curr_amount < CONFIG["MIN_AMOUNT"]: return None
    if close.iloc[-1] < CONFIG["MIN_PRICE"]: return None

    # === C. 指标计算 ===
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["BIAS20"] = (close - df["MA20"]) / df["MA20"] * 100
    
    # 布林线
    bb_ind = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_ind.bollinger_hband()
    df["BB_Lower"] = bb_ind.bollinger_lband()
    df["BB_Mid"] = bb_ind.bollinger_mavg()
    df["BB_Width"] = bb_ind.bollinger_wband()
    df["BB_PctB"] = bb_ind.bollinger_pband()

    # MACD
    macd = MACD(close)
    df["DIF"] = macd.macd()
    df["DEA"] = macd.macd_signal()
    df["MACD_Bar"] = macd.macd_diff()
    
    # KDJ & RSI
    kdj = StochasticOscillator(high, low, close)
    df["K"] = kdj.stoch()
    df["D"] = kdj.stoch_signal()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    
    # 资金
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["OBV"] = obv
    df["OBV_MA10"] = obv.rolling(10).mean()
    cmf = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    df["CMF"] = cmf
    
    adx_ind = ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx_ind.adx()

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # === D. 形态与筹码 ===
    
    # 1. 筹码分布
    chip_signal = ""
    high_120 = df["high"].tail(120).max()
    low_120 = df["low"].tail(120).min()
    current_pos = (curr["close"] - low_120) / (high_120 - low_120 + 0.001)
    if current_pos < 0.35:
        volatility = df["close"].tail(60).std() / df["close"].tail(60).mean()
        if volatility < 0.1: chip_signal = "🏆低位单峰密集"
        elif volatility < 0.15: chip_signal = "🔒相对密集"

    # 2. 形态特征
    patterns = []
    # 红肥绿瘦
    recent_20 = df.tail(20)
    vol_up = recent_20[recent_20['close'] > recent_20['open']]['volume'].sum()
    vol_down = recent_20[recent_20['close'] < recent_20['open']]['volume'].sum()
    if vol_up > vol_down * 1.5 and curr["CMF"] > 0: patterns.append("🟥红肥绿瘦")
    # 地量
    if curr["volume"] < df["volume"].tail(100).min() * 1.5 and current_pos < 0.2: patterns.append("💤地量见地价")
    # 蚂蚁上树
    recent_5 = df.tail(5)
    is_small = (abs(recent_5['close'] - recent_5['open']) / recent_5['close'] < 0.02).all()
    is_rising = (recent_5['close'] > recent_5['MA5']).all() and (recent_5['close'].iloc[-1] > recent_5['close'].iloc[0])
    if is_small and is_rising: patterns.append("🐜蚂蚁上树")
    # N字反包
    if (prev['close'] < prev['open']) and (curr['close'] > curr['open']) and (curr['close'] > prev['open']) and (curr['volume'] > prev['volume']):
        patterns.append("⚡N字反包")
        
    pattern_str = " ".join(patterns)

    # 3. 底背离
    div_signal = ""
    if curr["low"] == df["low"].tail(20).min():
        if curr["MACD_Bar"] > prev["MACD_Bar"] and curr["MACD_Bar"] < 0: div_signal = "💪MACD底背离"

    # 4. MACD 状态
    macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    macd_status = ""
    if macd_gold: macd_status = "🔥确认金叉"
    elif curr["DIF"] > curr["DEA"] and curr["DIF"] > 0 and curr["MACD_Bar"] > prev["MACD_Bar"]: macd_status = "⛽空中加油"
    elif curr["DIF"] < curr["DEA"] and (curr["DEA"] - curr["DIF"]) < 0.05 and curr["MACD_Bar"] > prev["MACD_Bar"]: macd_status = "🔔即将金叉"
    else: macd_status = "多头" if curr["DIF"] > curr["DEA"] else "空头"

    # 5. 布林状态
    bb_state = ""
    if curr["BB_PctB"] > 1.0: bb_state = "🚀突破上轨"
    elif curr["BB_PctB"] < 0.0: bb_state = "📉跌破下轨"
    elif curr["BB_Width"] < 12: bb_state = "↔️极度收口"
    elif abs(curr["close"] - curr["BB_Mid"])/curr["BB_Mid"] < 0.015: bb_state = "🛡️中轨支撑"
    else: bb_state = "上通道" if curr["BB_PctB"] > 0.5 else "下通道"

    # === E. 选股策略 ===
    signal_type = ""
    suggest_buy = curr["close"]
    stop_loss = curr["MA20"]
    
    df["pct_chg"] = close.pct_change() * 100
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    
    # 策略1: 龙回头
    if has_zt and -2.0 < curr["BIAS20"] < 8.0 and curr["BB_PctB"] > 0.3:
        if curr["volume"] < df["volume"].tail(30).max() * 0.6:
            signal_type = "🐉龙回头"
            stop_loss = round(curr["BB_Lower"], 2)
            
    # 策略2: 机构趋势
    if not signal_type and curr["close"] > curr["MA60"] and curr["CMF"] > 0.05 and curr["ADX"] > 20 and curr["BIAS20"] < 15.0:
        signal_type = "🏦机构控盘"
        suggest_buy = round(curr["vwap"], 2)

    # 策略3: 极度超跌
    if not signal_type and ((curr["RSI"] < 25) or (curr["BB_PctB"] < 0.05) or div_signal):
        signal_type = "📉极度超跌"
        stop_loss = round(curr["low"] * 0.96, 2)

    # 策略4: 底部变盘
    if not signal_type and curr["close"] < curr["MA60"] * 1.1 and (curr["BB_Width"] < 15 or "蚂蚁" in pattern_str):
         if macd_gold or curr["CMF"] > 0.1:
            signal_type = "⚡底部变盘"

    # === F. 评分过滤 (至少2项) ===
    score = 0
    reasons = []

    if signal_type: score += 1; reasons.append("策略")
    if chip_signal: score += 1; reasons.append("筹码")
    if pattern_str: score += 1; reasons.append("形态")
    if div_signal: score += 1; reasons.append("背离")
    if "金叉" in macd_status or "加油" in macd_status or "即将" in macd_status: score += 1; reasons.append("MACD")
    if "突破" in bb_state or "收口" in bb_state or "支撑" in bb_state: score += 1; reasons.append("布林")
    
    news_info = get_stock_catalysts(code)
    is_hot = False
    for hot in HOT_CONCEPTS:
        if hot in news_info: is_hot = True; break
    if is_hot: score += 1; reasons.append("热点")

    if score < 2: return None
    
    # === 🔥 G. 关键风控：资金流向过滤 ===
    # 如果资金是流出的，哪怕分数再高也不要
    obv_txt = "流入" if curr["OBV"] > curr["OBV_MA10"] else "流出"
    if obv_txt == "流出": 
        return None # <--- 核心修改点：直接返回空，不导出

    resonance_str = "+".join(reasons)
    vol_ma5 = df["volume"].rolling(5).mean().iloc[-1]
    vol_ratio = round(curr["volume"] / vol_ma5, 2) if vol_ma5 > 0 else 0

    return {
        "代码": code,
        "名称": name,
        "现价": curr["close"],
        "共振因子": resonance_str,
        "信号类型": signal_type,
        "题材与利好": news_info,
        "筹码分布": chip_signal,
        "形态特征": pattern_str,
        "MACD预警": macd_status,
        "底背离": div_signal,
        "布林状态": bb_state,
        "BIAS%": round(curr["BIAS20"], 1),
        "资金流向": obv_txt, # 这里只会显示"流入"
        "建议挂单": suggest_buy,
        "止损价": stop_loss,
        "量比": vol_ratio
    }

# --- 4. Excel 美化 (含详细释义字典) ---
def save_and_beautify(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"共振严选_{dt_str}.xlsx"
    
    if not data_list:
        pd.DataFrame([["无结果 (资金流出已被过滤)"]]).to_excel(filename)
        return filename

    df = pd.DataFrame(data_list)
    cols = ["代码", "名称", "现价", "共振因子", "信号类型", "题材与利好", 
            "筹码分布", "形态特征", "MACD预警", "底背离", 
            "布林状态", "BIAS%", "资金流向", "建议挂单", "止损价", "量比"]
    
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols]
    
    # 排序
    df["因子数"] = df["共振因子"].apply(lambda x: len(x.split('+')))
    df = df.sort_values(by=["因子数", "筹码分布"], ascending=[False, False])
    df.drop(columns=["因子数"], inplace=True)
    
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    # 样式
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_red = Font(color="FF0000", bold=True)
    font_purple = Font(color="800080", bold=True)
    font_orange = Font(color="FF8C00", bold=True)
    fill_yellow = PatternFill("solid", fgColor="FFF2CC")
    
    for cell in ws[1]:
        cell.fill = fill_blue
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')
        
    for row in ws.iter_rows(min_row=2):
        res_cell = row[3]
        res_cell.font = Font(color="0000FF", bold=True)
        
        news_cell = row[5]
        news_cell.alignment = Alignment(wrap_text=True, vertical='center')
        news_cell.font = Font(size=9)
        for hot in HOT_CONCEPTS:
            if hot in str(news_cell.value):
                news_cell.font = Font(size=9, color="FF0000", bold=True)
                break
        
        if "低位密集" in str(row[6].value): 
            row[6].font = font_red; row[6].fill = fill_yellow
            
        if "红肥" in str(row[7].value) or "N字" in str(row[7].value): row[7].font = font_red
        
        macd_val = str(row[8].value)
        if "即将" in macd_val: row[8].font = font_orange
        if "金叉" in macd_val or "加油" in macd_val: 
            row[8].font = font_red; row[8].fill = fill_yellow
            
        if row[9].value: row[9].font = font_red
        
        bb_val = str(row[10].value)
        if "突破" in bb_val: row[10].font = font_red
        if "收口" in bb_val: row[10].font = font_orange

    ws.column_dimensions['D'].width = 25
    ws.column_dimensions['F'].width = 45
    
    # ==========================================
    # 📚 底部指南与字典
    # ==========================================
    start_row = ws.max_row + 3
    
    # 样式定义
    title_font = Font(name='微软雅黑', size=12, bold=True, color="0000FF")
    sub_title_font = Font(name='微软雅黑', size=11, bold=True, color="000000")
    text_font = Font(name='微软雅黑', size=10)
    
    # --- Part 1: 实战操作指南 ---
    ws.cell(row=start_row, column=1, value="📘 实战操作指南 (Strategy)").font = title_font
    start_row += 1
    
    strategies = [
        ("【🐉 龙回头】", "含义：前期妖股缩量回踩。操作：在'建议挂单'处分批低吸，博反抽，跌破止损位坚决离场。"),
        ("【🏦 机构控盘】", "含义：趋势良好+资金锁仓。操作：沿5日/10日线持有，只要BIAS不过高可一直拿。"),
        ("【📉 极度超跌】", "含义：指标出现恐慌信号。操作：左侧博反弹，预期收益5-10%即止盈，快进快出。"),
        ("【⚡ 底部变盘】", "含义：布林收口+资金异动。操作：往往是大行情起点，可重仓关注，耐心持有。")
    ]
    for title, desc in strategies:
        ws.cell(row=start_row, column=1, value=title).font = Font(bold=True)
        ws.cell(row=start_row, column=2, value=desc).font = text_font
        start_row += 1

    start_row += 1
    
    # --- Part 2: 列名释义字典 ---
    ws.cell(row=start_row, column=1, value="📖 列名释义字典 (Dictionary)").font = title_font
    start_row += 1
    
    dictionary = [
        ("【共振因子】", "核心指标。显示该股有几项指标同时达标。如'筹码+热点'。共振越多，胜率越高。"),
        ("【题材与利好】", "个股最新新闻。红色字体代表该新闻命中了今日市场的热门板块(如固态电池)。"),
        ("【筹码分布】", "🏆低位单峰密集：主力在底部长期横盘吸筹，成本一致，爆发力最强。🔒相对密集：次优选择。"),
        ("【形态特征】", "🟥红肥绿瘦：主力资金运作痕迹；⚡N字反包：强势洗盘结束；🐜蚂蚁上树：温和建仓。"),
        ("【MACD预警】", "🔔即将金叉：鸭子张嘴，左侧埋伏点；⛽空中加油：上涨中继；🔥确认金叉：右侧启动点。"),
        ("【底背离】", "💪MACD底背离：股价创新低但指标未创新低。这是底部反转的最强技术信号。"),
        ("【布林状态】", "↔️极度收口：变盘前兆；🚀突破上轨：主升浪特征；🛡️中轨支撑：稳健买点。"),
        ("【BIAS%】", "乖离率。>15%代表短线涨幅过大，有回调风险(追高需谨慎)；负值代表超跌。"),
        ("【资金流向】", "基于OBV判断。本表已自动过滤掉'流出'的股票，只保留'流入'的优质标的。"),
        ("【建议挂单】", "系统计算的支撑位。龙回头是MA20/布林下轨；机构票是VWAP/MA5。不建议追高买入。"),
        ("【止损价】", "⛔ 风控铁律！收盘价跌破此价格，说明逻辑破坏，必须无条件卖出避险。")
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
    print("=== A股共振严选 (资金净流入版) ===")
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
                    print(f"  ★ 严选: {res['名称']} [{res['共振因子']}]")
                    results.append(res)
            except: pass

    print(f"\n耗时: {int(time.time() - start_time)}秒 | 严选出 {len(results)} 只精品")
    save_and_beautify(results)
    
    if not any(f.endswith('.xlsx') for f in os.listdir('.')):
        pd.DataFrame([["无"]]).to_excel(f"保底_{datetime.now().strftime('%H%M')}.xlsx")

if __name__ == "__main__":
    main()
