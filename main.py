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
        targets = targets[~targets['name'].str.contains('ST|退')]
        return targets, "方案A-东财"
    except:
        manual_list = [
            ["600519", "贵州茅台"], ["002594", "比亚迪"], ["000858", "五粮液"],
            ["601138", "工业富联"], ["600460", "士兰微"], ["000063", "中兴通讯"]
        ]
        return pd.DataFrame(manual_list, columns=["code", "name"]), "方案C-保底"

# --- 3. 数据获取 ---
def get_data_with_retry(code, start_date):
    for i in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except:
            time.sleep(1)
    return None

# --- 4. 核心计算 (妖股增强版) ---
def process_stock(df):
    # === A. 安全过滤 ===
    if len(df) < 120: return None
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]
    volume = df["volume"]
    
    # 估算 VWAP (成本线)
    if "成交额" in df.columns:
        df["vwap"] = df["成交额"] / df["volume"]
    else:
        df["vwap"] = (high + low + close) / 3

    # 僵尸股 & 低价股
    amount = df["成交额"].iloc[-1] if "成交额" in df.columns else close.iloc[-1] * volume.iloc[-1]
    if amount < 50000000: return None
    if close.iloc[-1] < 3.0: return None
    
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    
    if close.iloc[-1] < df["MA60"].iloc[-1]: return None # 必须在年线之上

    # === B. 指标计算 ===
    # KDJ
    kdj = StochasticOscillator(high, low, close)
    df["K"] = kdj.stoch()
    df["D"] = kdj.stoch_signal()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    # MACD
    macd = MACD(close)
    df["DIF"] = macd.macd()
    df["DEA"] = macd.macd_signal()
    
    # 机构指标
    adx_ind = ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx_ind.adx()
    df["PDI"] = adx_ind.adx_pos()
    df["MDI"] = adx_ind.adx_neg()
    
    mfi_ind = MFIIndicator(high, low, close, volume, window=14)
    df["MFI"] = mfi_ind.money_flow_index()

    cci_ind = CCIIndicator(high, low, close, window=14)
    df["CCI"] = cci_ind.cci()
    
    obv_ind = OnBalanceVolumeIndicator(close, volume)
    df["OBV"] = obv_ind.on_balance_volume()
    df["OBV_MA10"] = df["OBV"].rolling(10).mean()

    # 切片
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(curr['ADX']): return None

    # ==========================================
    # 🕵️‍♀️ 策略逻辑
    # ==========================================
    signal_type = ""
    suggest_buy = 0.0
    stop_loss = 0.0
    risk_warning = "" # 风险提示

    # --- 策略组 1: 🐉 妖股战法 (增强版：防高位接盘) ---
    df["pct_chg"] = close.pct_change() * 100
    recent_30 = df.tail(30)
    
    # 1. 基因检测: 30天内有过涨停 (涨幅>9.5%)
    has_zt = (recent_30["pct_chg"] > 9.5).sum() >= 1
    
    if has_zt:
        # 找到近期最高点的那一天
        peak_idx = recent_30["high"].idxmax()
        peak_date_row = df.loc[peak_idx]
        
        # --- 🔥 过滤器 A: 见顶形态过滤 (Tombstone Check) ---
        # 如果最高点那天是"巨量长上影" (上影线>3% 且 阴线/假阴线)，视为出货，不买
        peak_upper_shadow = (peak_date_row["high"] - max(peak_date_row["open"], peak_date_row["close"])) / peak_date_row["close"]
        is_bad_peak = peak_upper_shadow > 0.03 and peak_date_row["volume"] > recent_30["volume"].mean() * 2
        
        if not is_bad_peak:
            # 2. 回调状态确认
            # 价格在 MA20 之上 (生命线)
            if curr["close"] > curr["MA20"]:
                # 回踩幅度: 距离 MA10 或 MA20 很近 (<2%)
                dist_ma10 = abs(curr["close"] - curr["MA10"]) / curr["MA10"]
                dist_ma20 = abs(curr["close"] - curr["MA20"]) / curr["MA20"]
                
                if dist_ma10 < 0.025 or dist_ma20 < 0.025:
                    
                    # --- 🔥 过滤器 B: 缩量确认 (Volume Shrinkage) ---
                    # 今天的成交量，必须小于近期最大成交量的 60% (缩量才安全)
                    max_vol = recent_30["volume"].max()
                    if curr["volume"] < max_vol * 0.6:
                        
                        # --- 🔥 过滤器 C: 换手率风控 ---
                        # 如果有换手率数据，且今日换手率 > 15%，说明分歧太大，不买
                        safe_turnover = True
                        if "换手率" in df.columns and curr["换手率"] > 15:
                            safe_turnover = False
                        
                        if safe_turnover:
                            # 判定成功
                            if dist_ma10 < 0.025:
                                signal_type = "🐉龙回头(踩10日线)"
                                suggest_buy = round(curr["MA10"], 2)
                                stop_loss = round(curr["MA10"] * 0.95, 2) # 强势股止损要快
                            else:
                                signal_type = "🐉龙回头(踩20日线)"
                                suggest_buy = round(curr["MA20"], 2)
                                stop_loss = round(curr["MA20"] * 0.97, 2)

    # --- 策略组 2: 👑 机构趋势 (保持不变) ---
    if not signal_type:
        if curr["ADX"] > 25 and curr["PDI"] > curr["MDI"] and curr["close"] > curr["vwap"] and curr["MFI"] < 85:
            if (curr["ADX"] > prev["ADX"]) and (curr["CCI"] > 100):
                signal_type = "👑机构主升浪"
                suggest_buy = round(curr["vwap"], 2)
                stop_loss = round(curr["MA20"], 2)

    # --- 策略组 3: 🟢 极品底吸 (保持不变) ---
    if not signal_type:
        # J值反击
        was_oversold = (prev["J"] < 0) or (df.iloc[-3]["J"] < 0)
        if was_oversold and curr["close"] > curr["open"] and curr["J"] > prev["J"]:
            signal_type = "🟢J值超卖反击"
            suggest_buy = round(curr["close"], 2)
            stop_loss = round(curr["low"] * 0.98, 2)
        # 金针探底
        elif (min(curr["open"], curr["close"]) - curr["low"] > abs(curr["open"] - curr["close"]) * 2) and (curr["low"] < curr["MA20"]):
            signal_type = "🟢金针探底"
            suggest_buy = round(curr["low"] + (min(curr["open"], curr["close"]) - curr["low"])*0.5, 2)
            stop_loss = round(curr["low"] * 0.99, 2)
        # 生命线
        elif abs(curr["low"] - curr["MA60"])/curr["MA60"] < 0.015 and curr["close"] > curr["MA60"]:
            signal_type = "🟢生命线(MA60)回踩"
            suggest_buy = round(curr["MA60"], 2)
            stop_loss = round(curr["MA60"] * 0.98, 2)

    if not signal_type: return None
    
    # 全局OBV过滤
    if curr["OBV"] < df["OBV"].tail(20).mean() * 0.9: return None

    return {
        "code": df.name,
        "name": "", 
        "close": curr["close"],
        "signal_type": signal_type,
        "buy_price": suggest_buy,   
        "stop_loss": stop_loss,
        "adx": round(curr["ADX"], 1),
        "j_val": round(curr["J"], 1),
        "vol_ratio": round(volume.iloc[-1] / df["volume"].rolling(5).mean().iloc[-1], 2) if df["volume"].rolling(5).mean().iloc[-1] != 0 else 0
    }

# --- 5. 美化 Excel ---
def add_guide_to_excel(filename, data_len):
    try:
        wb = openpyxl.load_workbook(filename)
        ws = wb.active
        header_font = Font(name='微软雅黑', size=11, bold=True, color="FFFFFF")
        text_font = Font(name='微软雅黑', size=10)
        red_font = Font(name='微软雅黑', size=10, bold=True, color="FF0000")
        
        start_row = data_len + 4
        ws.cell(row=start_row, column=1, value="📘 增强版操作指南 (防站岗)").font = Font(size=12, bold=True, color="0000FF")
        start_row += 1
        
        guides = [
            ("【🐉 妖股战法 - 安全增强】", ""),
            ("1. 策略逻辑", "只做有过涨停基因，且缩量回调到均线支撑的票。"),
            ("2. 防高位站岗", "已自动剔除：高位放量长上影(墓碑线)、换手率>15%的出货盘。"),
            ("3. 挂单技巧", "请严格在'建议挂单价'埋伏，不成交不追高。"),
            ("", ""),
            ("【👑 机构趋势】", "ADX>25 强趋势，沿成本线买入，适合中线。"),
            ("【🟢 极品底吸】", "左侧博反弹，必须设好止损，快进快出。"),
            ("", ""),
            ("【⚠️ 铁律】", "跌破'止损价'，无论理由，坚决卖出！")
        ]
        
        for i, (title, desc) in enumerate(guides):
            ws.cell(row=start_row + i, column=1, value=title).font = Font(bold=True)
            ws.cell(row=start_row + i, column=2, value=desc).font = text_font
            if "防" in title or "铁律" in title: ws.cell(row=start_row + i, column=2).font = red_font

        wb.save(filename)
    except: pass

# --- 6. 主程序 ---
def main():
    print("=== 全功能增强版 (防高位站岗) ===")
    pd.DataFrame([["Init", "OK"]]).to_excel("Init_Check.xlsx", index=False)
    
    try:
        targets, source_name = get_targets_robust()
        
        start_dt = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
        result_data = []
        
        total = len(targets)
        print(f"开始扫描 {total} 只股票...")

        for i, s in targets.iterrows():
            code = s["code"]
            name = s["name"]
            
            if i % 20 == 0: print(f"进度: {i}/{total} ...")

            try:
                df = get_data_with_retry(code, start_dt)
                if df is None: continue

                rename_dict = {
                    "日期":"date","开盘":"open","收盘":"close",
                    "最高":"high","最低":"low","成交量":"volume",
                    "成交额":"amount", "换手率":"turnover"
                }
                col_map = {k:v for k,v in rename_dict.items() if k in df.columns}
                df.rename(columns=col_map, inplace=True)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df.name = code

                res = process_stock(df)
                
                if res:
                    print(f"  ★ {res['signal_type']}: {code} {name}")
                    result_data.append({
                        "代码": code,
                        "名称": name,
                        "现价": res["close"],
                        "信号类型": res["signal_type"], 
                        "建议挂单": res["buy_price"],  
                        "止损价": res["stop_loss"],
                        "ADX": res["adx"],
                        "J值": res["j_val"],
                        "量比": res["vol_ratio"]
                    })
            except: continue
            time.sleep(0.05)

        dt_str = datetime.now().strftime("%Y%m%d")
        if result_data:
            df_res = pd.DataFrame(result_data)
            df_res = df_res.sort_values(by=["信号类型"], ascending=False)
            filename = f"全策略精选_{dt_str}.xlsx"
            df_res.to_excel(filename, index=False)
            add_guide_to_excel(filename, len(df_res))
            print(f"完成！已保存: {filename}")
        else:
            pd.DataFrame([["无"]]).to_excel(f"无结果_{dt_str}.xlsx")

    except Exception:
        # 发生严重错误时，写入 txt，这样 run.yml 也能上传它
        err = traceback.format_exc()
        print(f"FATAL ERROR: {err}")
        with open("FATAL_ERROR.txt", "w") as f:
            f.write(err)

    # 确保无论如何都有 Excel 生成（防止 Release 为空）
    # 检查当前目录下是否有 xlsx 文件
    has_excel = False
    for fname in os.listdir("."):
        if fname.endswith(".xlsx"):
            has_excel = True
            break
    
    if not has_excel:
        # 如果没生成过 Excel，强制生成一个空的
        dt_str = datetime.now().strftime("%Y%m%d")
        pd.DataFrame([["无结果", "可能是没选出股票，也可能是出错了，请看日志"]]).to_excel(f"强制保底_{dt_str}.xlsx", index=False)

if __name__ == "__main__":
    main()
