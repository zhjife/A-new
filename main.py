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

# 全局变量：用于存储今日热门概念
HOT_CONCEPTS = []

# --- 2. 宏观数据获取 (新增: 获取市场热点) ---
def get_market_hot_spots():
    """
    获取当前市场涨幅前10的概念板块，模拟'搜索热门利好政策'
    """
    print(">>> [0/4] 正在扫描全市场热门题材与政策导向...")
    global HOT_CONCEPTS
    try:
        # 获取概念板块涨幅榜
        df = ak.stock_board_concept_name_em()
        # 按涨跌幅排序，取前15名
        df = df.sort_values(by="涨跌幅", ascending=False).head(15)
        HOT_CONCEPTS = df["板块名称"].tolist()
        print(f"🔥 今日资金/政策热点: {HOT_CONCEPTS}")
    except:
        print("⚠️ 热点获取超时，跳过热点匹配，仅进行个股新闻检索。")
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
    except Exception as e:
        manual_list = [
            ["600519", "贵州茅台"], ["002594", "比亚迪"], ["601138", "工业富联"]
        ]
        return pd.DataFrame(manual_list, columns=["code", "name"]), "保底列表"

# --- 3. 数据获取 ---
def get_data_with_retry(code, start_date):
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=8)
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except:
            time.sleep(0.5)
    return None

# --- 新增: 个股利好与题材检索函数 ---
def get_stock_catalysts(code):
    """
    只有当股票被选中时才调用此函数，节省时间。
    获取: 所属行业 + 是否热门概念 + 最新一条新闻
    """
    try:
        # 1. 获取个股资料 (行业/概念)
        # 注意：这里使用 info 接口，如果太慢可以简化
        # 为了速度，我们这里只尝试获取新闻，概念匹配留给后续优化，或者简单的行业接口
        
        # 尝试获取个股新闻 (取最新一条)
        news_df = ak.stock_news_em(symbol=code)
        latest_news = ""
        if not news_df.empty:
            # 取第一条新闻标题，并截取前20个字
            title = news_df.iloc[0]['新闻标题']
            date = news_df.iloc[0]['发布时间']
            # 如果是最近2天的新闻，更有价值
            latest_news = f"[{date[5:10]}] {title}"
        
        # 2. 简单的行业/概念获取 (利用 Akshare 的个股信息接口)
        # 这里的接口比较慢，所以我们用一个巧妙的方法：
        # 如果前面 get_market_hot_spots 成功了，我们无法直接在这里反查个股是否属于该概念
        # 除非遍历所有概念。为了效率，我们只显示新闻。
        
        return latest_news
    except:
        return "无近期新闻"

# --- 4. 核心逻辑 (保持不变，末尾增加新闻调用) ---
def process_stock_logic(df, code, name):
    # === A. 基础清洗 ===
    if len(df) < 60: return None
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
    
    bb_ind = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_ind.bollinger_hband()
    df["BB_Lower"] = bb_ind.bollinger_lband()
    df["BB_Mid"] = bb_ind.bollinger_mavg()      
    df["BB_Width"] = bb_ind.bollinger_wband()   
    df["BB_PctB"] = bb_ind.bollinger_pband()    

    macd = MACD(close)
    df["DIF"] = macd.macd()
    df["DEA"] = macd.macd_signal()
    
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
    df["PDI"] = adx_ind.adx_pos()
    df["MDI"] = adx_ind.adx_neg()

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # === D. 状态判断 ===
    macd_gold = (prev["DIF"] < prev["DEA"]) and (curr["DIF"] > curr["DEA"])
    kdj_gold = (prev["J"] < prev["K"]) and (curr["J"] > curr["K"])
    is_dual_gold = macd_gold and kdj_gold
    
    macd_str = "🔥金叉" if macd_gold else ("多头" if curr["DIF"] > curr["DEA"] else "空头")
    kdj_str = "⚡金叉" if kdj_gold else ("多头" if curr["J"] > curr["K"] else "空头")

    bb_state = ""
    if curr["BB_PctB"] > 1.0: bb_state = "🚀突破上轨"
    elif curr["BB_PctB"] < 0.0: bb_state = "📉跌破下轨"
    elif curr["BB_Width"] < 12: bb_state = "↔️极度收口"
    elif abs(curr["close"] - curr["BB_Mid"])/curr["BB_Mid"] < 0.015: bb_state = "🛡️中轨支撑"
    else: bb_state = "上通道" if curr["BB_PctB"] > 0.5 else "下通道"

    # ==========================================
    # 🕵️‍♀️ 策略核心
    # ==========================================
    signal_type = ""
    suggest_buy = 0.0
    stop_loss = 0.0
    
    # 策略1: 龙回头
    df["pct_chg"] = close.pct_change() * 100
    has_zt = (df["pct_chg"].tail(30) > 9.5).sum() >= 1
    if has_zt:
        if -2.0 < curr["BIAS20"] < 8.0 and curr["BB_PctB"] > 0.3:
            max_vol_30 = df["volume"].tail(30).max()
            if curr["volume"] < max_vol_30 * 0.6:
                signal_type = "🐉龙回头(缩量踩线)"
                suggest_buy = round(curr["MA20"], 2)
                stop_loss = round(curr["BB_Lower"], 2)
                if curr["BB_Width"] < 20: signal_type += "+收口"

    # 策略2: 机构趋势
    if not signal_type:
        if curr["close"] > curr["MA60"] and curr["CMF"] > 0.05:
            if curr["BB_PctB"] > 0.5 and curr["ADX"] > 20:
                if curr["BIAS20"] < 15.0: 
                    signal_type = "🏦机构控盘"
                    if "突破" in bb_state: signal_type += "(突破)"
                    elif is_dual_gold: signal_type += "(启动)"
                    suggest_buy = round(curr["vwap"], 2)
                    stop_loss = round(curr["MA20"] * 0.98, 2)

    # 策略3: 极度超跌
    if not signal_type:
        is_panic = (curr["RSI"] < 25) or (curr["BB_PctB"] < 0.05)
        is_j_turn = (prev["J"] < 10 and curr["J"] > prev["J"])
        if is_panic or is_j_turn:
            signal_type = "📉极度超跌"
            if "跌破" in bb_state: signal_type += "(破下轨)"
            suggest_buy = round(curr["close"], 2)
            stop_loss = round(curr["low"] * 0.96, 2)

    # 策略4: 底部变盘
    if not signal_type:
        if curr["close"] < curr["MA60"] * 1.1:
            if curr["BB_Width"] < 15: 
                if macd_gold or (curr["CMF"] > 0.1):
                    signal_type = "⚡底部变盘(收口)"
                    suggest_buy = round(curr["close"], 2)
                    stop_loss = round(curr["BB_Lower"], 2)

    if not signal_type: return None

    # === 🔥 新增: 只有被选中后，才去“搜索”该股的利好消息 ===
    # 注意：这会增加一点点耗时，但很有必要
    news_info = get_stock_catalysts(code)
    
    # 简单的热点匹配逻辑（如果个股名称包含热门概念关键字，或者后续扩展行业数据）
    # 这里我们做一个简单的名称/行业逻辑匹配，或者在 news_info 里标记
    
    vol_ma5 = df["volume"].rolling(5).mean().iloc[-1]
    vol_ratio = round(curr["volume"] / vol_ma5, 2) if vol_ma5 > 0 else 0
    obv_txt = "流入" if curr["OBV"] > curr["OBV_MA10"] else "流出"

    return {
        "代码": code,
        "名称": name,
        "现价": curr["close"],
        "信号类型": signal_type,
        "题材与利好": news_info,      # <--- 新增列
        "布林状态": bb_state,
        "BIAS%": round(curr["BIAS20"], 1),
        "MACD金叉": macd_str,
        "KDJ金叉": kdj_str,
        "资金流向": obv_txt,
        "CMF资金": round(curr["CMF"], 3),
        "建议挂单": suggest_buy,
        "止损价": stop_loss,
        "量比": vol_ratio
    }

# --- 5. 多线程执行 ---
def analyze_one_stock(code, name, start_dt):
    try:
        df = get_data_with_retry(code, start_dt)
        if df is None: return None
        return process_stock_logic(df, code, name)
    except:
        return None

# --- 6. Excel 美化 (含新列) ---
def save_and_beautify(data_list):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"实战精选_{dt_str}.xlsx"
    
    if not data_list:
        pd.DataFrame([["无结果"]]).to_excel(filename)
        return filename

    df = pd.DataFrame(data_list)
    # 调整列顺序，把利好放在比较显眼的位置
    cols = ["代码", "名称", "现价", "信号类型", "题材与利好", "布林状态", "BIAS%", "MACD金叉", "KDJ金叉", "资金流向", "CMF资金", "建议挂单", "止损价", "量比"]
    df = df[cols]
    
    df = df.sort_values(by=["信号类型", "BIAS%"], ascending=[True, True])
    df.to_excel(filename, index=False)
    
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    
    header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
    fill_blue = PatternFill("solid", fgColor="4472C4")
    font_red = Font(color="FF0000", bold=True)
    font_green = Font(color="008000", bold=True)
    fill_yellow = PatternFill("solid", fgColor="FFF2CC")
    
    for cell in ws[1]:
        cell.fill = fill_blue
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')
        
    for row in ws.iter_rows(min_row=2):
        # 信号类型
        signal = str(row[3].value)
        if "龙回头" in signal: row[3].font = Font(color="800080", bold=True)
        
        # 题材与利好 (E列) -> 设置自动换行，字体改小一点
        news_cell = row[4]
        news_cell.alignment = Alignment(wrap_text=True, vertical='center')
        news_cell.font = Font(name='微软雅黑', size=9)
        # 如果包含今日热点，标红 (简单匹配)
        for hot in HOT_CONCEPTS:
            if hot in str(news_cell.value):
                news_cell.font = Font(name='微软雅黑', size=9, color="FF0000", bold=True)
                break
        
        # 布林状态 (F列)
        bb_val = str(row[5].value)
        if "突破" in bb_val: 
            row[5].font = font_red
            row[5].fill = fill_yellow
        elif "收口" in bb_val:
            row[5].font = Font(color="FF8C00", bold=True)
        elif "跌破" in bb_val:
            row[5].font = font_green
            
        # 双金叉 (H, I列)
        macd_val = str(row[7].value)
        kdj_val = str(row[8].value)
        if "金叉" in macd_val and "金叉" in kdj_val:
            row[7].fill = fill_yellow
            row[8].fill = fill_yellow
            row[7].font = font_red
            row[8].font = font_red
            
    # 设置题材列的宽度
    ws.column_dimensions['E'].width = 50 
    
    # 指南
    last_row = ws.max_row
    start_row = last_row + 3
    guides = [
        ("📘 实战操作指南", f"今日市场热点：{' '.join(HOT_CONCEPTS[:5])}..."), # 显示前5个热点
        ("【题材共振】", "如果[题材与利好]列包含上述热点，且技术面金叉，为最强买点。"),
        ("【🐉 龙回头】", "妖股回调 + 利好消息不绝 = 第二波行情启动。"),
        ("【⚡ 底部变盘】", "布林收口 + 突发利好 = 暴力拉升起点。"),
        ("⛔ 风控铁律", "消息只是催化剂，跌破 [止损价] 必须离场！")
    ]
    
    for i, (title, desc) in enumerate(guides):
        r = start_row + i
        ws.cell(row=r, column=1, value=title).font = Font(bold=True)
        ws.cell(row=r, column=2, value=desc)
        if "风控" in title: ws.cell(row=r, column=1).font = font_red

    wb.save(filename)
    print(f"✅ 结果已保存: {filename}")
    return filename

# --- 7. 主程序 ---
def main():
    print("=== A股实战选股 (技术+题材利好共振版) ===")
    
    # 1. 先获取今日热点 (模拟搜索)
    get_market_hot_spots()
    
    start_time = time.time()
    targets, source_name = get_targets_robust()
    start_dt = (datetime.now() - timedelta(days=CONFIG["DAYS_LOOKBACK"])).strftime("%Y%m%d")
    
    print(f"[{source_name}] 待扫描: {len(targets)} 只 | 启动 {CONFIG['MAX_WORKERS']} 线程...")
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_stock = {
            executor.submit(analyze_one_stock, row['code'], row['name'], start_dt): row['code']
            for _, row in targets.iterrows()
        }
        
        count = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1
            if count % 100 == 0: print(f"进度: {count}/{len(targets)} ...")
            try:
                res = future.result()
                if res:
                    print(f"  ★ 选中: {res['名称']} | 新闻: {res['题材与利好'][:15]}...")
                    results.append(res)
            except: pass

    print(f"\n耗时: {int(time.time() - start_time)}秒 | 选中 {len(results)} 只")
    save_and_beautify(results)
    
    if not any(f.endswith('.xlsx') for f in os.listdir('.')):
        pd.DataFrame([["无"]]).to_excel(f"强制保底_{datetime.now().strftime('%H%M')}.xlsx")

if __name__ == "__main__":
    main()
