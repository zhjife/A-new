import akshare as ak
import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import os
import time
import sys
import traceback
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import concurrent.futures  # <--- 新增：多线程库

# --- 1. 环境初始化 ---
current_dir = os.getcwd()
sys.path.append(current_dir)

# --- 2. 获取股票列表 ---
def get_targets_robust():
    print(">>> 开始获取股票列表...")
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "名称"]]
        df.columns = ["code", "name"]
        targets = df[df["code"].str.startswith(("60", "00"))]
        # 剔除 ST
        targets = targets[~targets['name'].str.contains('ST|退')]
        return targets, "方案A-东财"
    except:
        manual_list = [
            ["600519", "贵州茅台"], ["002594", "比亚迪"], ["300750", "宁德时代"],
            ["601138", "工业富联"], ["600460", "士兰微"], ["000063", "中兴通讯"]
        ]
        return pd.DataFrame(manual_list, columns=["code", "name"]), "方案C-保底"

# --- 3. 数据获取 (带简单的超时控制) ---
def get_data_with_retry(code, start_date):
    # 多线程模式下，重试次数不宜过多，否则会阻塞线程池
    for i in range(2):
        try:
            # timeout=5 设定超时，防止卡死
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq", timeout=10)
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except:
            time.sleep(1)
    return None
    
# --- 4. 核心计算 (妖股+机构+底吸 + OBV资金流向) ---
def process_stock_logic(df, code, name):
    # === A. 安全过滤 ===
    if len(df) < 120: return None
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # 估算 VWAP
    if "成交额" in df.columns:
        df["vwap"] = df["成交额"] / df["volume"]
    else:
        df["vwap"] = (high + low + close) / 3

    # 1. 放宽成交额限制：2500万
    amount = df["成交额"].iloc[-1] if "成交额" in df.columns else close.iloc[-1] * volume.iloc[-1]
    if amount < 25000000: return None 
    
    # 2. 价格底线
    if close.iloc[-1] < 3.0: return None
    
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    
    # === B. 指标计算 ===
    kdj = StochasticOscillator(high, low, close, window=9, smooth_window=3)
    df["K"] = kdj.stoch()
    df["D"] = kdj.stoch_signal()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    macd = MACD(close)
    df["DIF"] = macd.macd()
    
    adx_ind = ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx_ind.adx()
    df["PDI"] = adx_ind.adx_pos()
    df["MDI"] = adx_ind.adx_neg()
    
    mfi_ind = MFIIndicator(high, low, close, volume, window=14)
    df["MFI"] = mfi_ind.money_flow_index()

    cci_ind = CCIIndicator(high, low, close, window=14)
    df["CCI"] = cci_ind.cci()
    
    # --- OBV 资金流向计算 ---
    obv_ind = OnBalanceVolumeIndicator(close, volume)
    df["OBV"] = obv_ind.on_balance_volume()
    df["OBV_MA5"] = df["OBV"].rolling(5).mean()
    df["OBV_MA10"] = df["OBV"].rolling(10).mean()

    # 切片
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(curr['ADX']): return None

    # 计算 OBV 状态
    obv_status = "资金流出" # 默认
    if curr["OBV"] > curr["OBV_MA5"] and curr["OBV_MA5"] > curr["OBV_MA10"]:
        obv_status = "🔴资金持续流入"
    elif curr["OBV"] > curr["OBV_MA10"]:
        obv_status = "🟠资金流入"
    else:
        obv_status = "🟢资金流出"

    # ==========================================
    # 🕵️‍♀️ 策略逻辑 (优化版)
    # ==========================================
    signal_type = ""
    suggest_buy = 0.0
    stop_loss = 0.0

    # --- 策略组 1: 🐉 妖股战法 ---
    df["pct_chg"] = close.pct_change() * 100
    recent_30 = df.tail(30)
    has_zt = (recent_30["pct_chg"] > 9.5).sum() >= 1
    
    if has_zt:
        if curr["close"] > curr["MA20"]:
            dist_ma10 = abs(curr["close"] - curr["MA10"]) / curr["MA10"]
            dist_ma20 = abs(curr["close"] - curr["MA20"]) / curr["MA20"]
            
            if dist_ma10 < 0.04 or dist_ma20 < 0.04:
                max_vol = recent_30["volume"].max()
                if curr["volume"] < max_vol * 0.6:
                    if dist_ma10 < 0.04:
                        signal_type = "🐉龙回头(踩10日线)"
                        suggest_buy = round(curr["MA10"], 2)
                        stop_loss = round(curr["MA10"] * 0.95, 2)
                    else:
                        signal_type = "🐉龙回头(踩20日线)"
                        suggest_buy = round(curr["MA20"], 2)
                        stop_loss = round(curr["MA20"] * 0.95, 2)

    # --- 策略组 2: 👑 机构趋势 ---
    if not signal_type:
        # ADX > 20, 趋势向上, 且必须是资金流入状态才算机构票
        if curr["ADX"] > 20 and curr["PDI"] > curr["MDI"] and curr["close"] > curr["MA20"]:
            # 机构票最好要求资金至少是流入状态
            if (curr["ADX"] >= prev["ADX"]) and (curr["CCI"] > 50) and (curr["MFI"] < 85):
                 # 如果是流出状态，可能在出货，过滤掉
                if "流出" not in obv_status: 
                    signal_type = "👑机构主升浪"
                    suggest_buy = round(curr["vwap"], 2)
                    stop_loss = round(curr["MA20"] * 0.98, 2)

    # --- 策略组 3: 🟢 极品底吸 ---
    if not signal_type:
        was_oversold = (prev["J"] < 10) or (df.iloc[-3]["J"] < 10)
        
        if was_oversold and curr["J"] > prev["J"] and curr["close"] > curr["open"]:
            signal_type = "🟢J值超卖反击"
            suggest_buy = round(curr["close"], 2)
            stop_loss = round(curr["low"] * 0.98, 2)
            
        elif (min(curr["open"], curr["close"]) - curr["low"] > abs(curr["open"] - curr["close"]) * 1.5) and (curr["low"] < curr["MA20"]):
            signal_type = "🟢金针探底"
            suggest_buy = round(curr["low"] * 1.01, 2)
            stop_loss = round(curr["low"] * 0.99, 2)
            
        elif abs(curr["low"] - curr["MA60"])/curr["MA60"] < 0.02 and curr["close"] > curr["MA60"]:
            signal_type = "🟢生命线(MA60)回踩"
            suggest_buy = round(curr["MA60"], 2)
            stop_loss = round(curr["MA60"] * 0.98, 2)

    if not signal_type: return None

    return {
        "代码": code,
        "名称": name,
        "现价": curr["close"],
        "信号类型": signal_type,
        "资金流向": obv_status,  # <--- 新增列
        "建议挂单": suggest_buy,
        "止损价": stop_loss,
        "ADX": round(curr["ADX"], 1),
        "J值": round(curr["J"], 1),
        "量比": round(volume.iloc[-1] / df["volume"].rolling(5).mean().iloc[-1], 2) if df["volume"].rolling(5).mean().iloc[-1] != 0 else 0
    }


# --- 5. 多线程包装函数 ---
def analyze_one_stock(code, name, start_dt):
    """
    单个股票的处理入口，包含数据获取和逻辑计算
    """
    try:
        # 获取数据
        df = get_data_with_retry(code, start_dt)
        if df is None: return None

        # 清洗数据
        rename_dict = {
            "日期":"date","开盘":"open","收盘":"close",
            "最高":"high","最低":"low","成交量":"volume",
            "成交额":"amount", "换手率":"turnover"
        }
        # 动态重命名，防止接口列名变化报错
        col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
        df.rename(columns=col_map, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # 调用核心逻辑
        return process_stock_logic(df, code, name)
    except:
        return None

# --- 6. 美化 Excel ---
def add_guide_to_excel(filename, data_len):
    try:
        wb = openpyxl.load_workbook(filename)
        ws = wb.active
        header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
        text_font = Font(name='微软雅黑', size=10)
        red_font = Font(name='微软雅黑', size=10, bold=True, color="FF0000")
        
        start_row = data_len + 4
        ws.cell(row=start_row, column=1, value="📘 增强版操作指南 (多线程极速版)").font = Font(size=12, bold=True, color="0000FF")
        
        guides = [
            ("【🐉 妖股战法】", "缩量回踩10日/20日线。建议挂单低吸，不追高。"),
            ("【👑 机构趋势】", "ADX强趋势。沿成本线买入，适合波段持有。"),
            ("【🟢 极品底吸】", "左侧博反弹。严格按止损价操作，快进快出。"),
            ("【⚠️ 风险提示】", "跌破止损价必须无条件卖出！")
        ]
        
        for i, (title, desc) in enumerate(guides):
            curr_r = start_row + 1 + i
            ws.cell(row=curr_r, column=1, value=title).font = Font(bold=True)
            ws.cell(row=curr_r, column=2, value=desc).font = text_font
            if "风险" in title: ws.cell(row=curr_r, column=2).font = red_font

        wb.save(filename)
    except: pass

# --- 7. 主程序 ---
def main():
    print("=== 全功能多线程极速版启动 ===")
    
    # 加上时间戳防止文件冲突
    ts = datetime.now().strftime("%H%M")
    pd.DataFrame([["Init", "OK"]]).to_excel(f"Init_Check_{ts}.xlsx", index=False)
    
    try:
        targets, source_name = get_targets_robust()
        
        # 为了计算 MA60，至少需要过去 120 天数据
        start_dt = (datetime.now() - timedelta(days=150)).strftime("%Y%m%d")
        result_data = []
        
        total = len(targets)
        print(f"待扫描股票: {total} 只 | 来源: {source_name}")
        print("🚀 启动 4 线程并发扫描 (请耐心等待约 10-15 分钟)...")

        # --- 核心：多线程处理 ---
        # max_workers=4 是 GitHub Actions 的安全并发数，太高容易被封 IP
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(analyze_one_stock, row['code'], row['name'], start_dt): row['code']
                for _, row in targets.iterrows()
            }
            
            # 处理结果 (带进度显示)
            count = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                count += 1
                if count % 100 == 0:
                    print(f"进度: {count}/{total} ...")
                
                try:
                    res = future.result()
                    if res:
                        print(f"  ★ 发现: {res['名称']} [{res['信号类型']}]")
                        result_data.append(res)
                except:
                    pass

        # 保存结果
        dt_str = datetime.now().strftime("%Y%m%d_%H%M") # 精确到分钟
        
        if result_data:
            df_res = pd.DataFrame(result_data)
            # 排序：妖股 -> 机构 -> 底吸
            df_res = df_res.sort_values(by=["信号类型"], ascending=False)
            
            filename = f"极速精选_{dt_str}.xlsx"
            df_res.to_excel(filename, index=False)
            add_guide_to_excel(filename, len(df_res))
            print(f"✅ 完成！结果已保存: {filename}")
        else:
            print("今日无符合条件的股票。")
            pd.DataFrame([["无"]]).to_excel(f"无结果_{dt_str}.xlsx")

    except Exception:
        # 严重错误记录
        err = traceback.format_exc()
        print(f"FATAL ERROR: {err}")
        with open("FATAL_ERROR.txt", "w") as f: f.write(err)

    # 强制保底文件 (防止 Action 找不到文件报错)
    # 检查目录下是否有 xlsx
    has_xlsx = any(f.endswith('.xlsx') for f in os.listdir('.'))
    if not has_xlsx:
        pd.DataFrame([["无结果"]]).to_excel(f"强制保底_{dt_str}.xlsx", index=False)

if __name__ == "__main__":
    main()
