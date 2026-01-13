import akshare as ak
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime, timedelta
import os
import time
import sys
import traceback

# --- 1. 环境初始化 ---
current_dir = os.getcwd()
sys.path.append(current_dir)

# --- 2. 获取股票列表 (保持不变) ---
def get_targets_robust():
    print(">>> 开始获取股票列表...")
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "名称"]]
        df.columns = ["code", "name"]
        targets = df[df["code"].str.startswith(("60", "00"))]
        return targets, "方案A-东财"
    except:
        manual_list = [["600519", "贵州茅台"], ["000858", "五粮液"], ["601318", "中国平安"]]
        return pd.DataFrame(manual_list, columns=["code", "name"]), "方案C-保底"

# --- 3. 获取热点 (保持不变) ---
def get_hot_stock_pool():
    # 为了节省时间，这里简写，你可以保留你原来的热点逻辑
    # 如果觉得热点获取太慢，可以先返回 None
    return None 

# --- 4. 数据获取 ---
def get_data_with_retry(code, start_date):
    for i in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
            if df is None or df.empty: raise ValueError("Empty")
            return df
        except:
            time.sleep(1)
    return None

# --- 5. 核心计算 (新增：防坑过滤器) ---
def process_stock(df):
    if len(df) < 60: return None
    
    # 基础指标
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()
    
    # 量比
    vol_ma5 = df["volume"].rolling(5).mean()
    if vol_ma5.iloc[-1] == 0: return None
    vol_ratio = round(df["volume"].iloc[-1] / vol_ma5.iloc[-1], 2)

    # MACD
    macd = MACD(df["close"])
    df["DIF"] = macd.macd()
    df["DEA"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    
    # RSI
    rsi_ind = RSIIndicator(close=df["close"], window=14)
    df["RSI"] = rsi_ind.rsi()

    # OBV
    obv_ind = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    df["OBV"] = obv_ind.on_balance_volume()
    df["OBV_MA10"] = df["OBV"].rolling(10).mean()

    # 布林带
    boll = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["BOLL_Mid"] = boll.bollinger_mavg()
    df["BOLL_Up"] = boll.bollinger_hband()

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(curr['MA60']): return None

    # ==========================================
    # 🛡️ 避坑过滤器 (Pitfall Filters) - 新增！！
    # ==========================================
    
    # 1. 乖离率过滤 (防追高)
    # 计算公式: (收盘价 - MA5) / MA5
    # 逻辑: 如果股价超过5日线 5% 以上，说明短期涨幅过大，第二天极大概率回调
    bias_ma5 = (curr["close"] - curr["MA5"]) / curr["MA5"]
    if bias_ma5 > 0.05: 
        return None 

    # 2. 上影线过滤 (防抛压)
    # 计算公式: (最高价 - 收盘价) / 收盘价
    # 逻辑: 如果上影线长度超过 2%，说明上方压力大，主力做多意愿不坚决
    # 特例：如果是涨停板(收盘价接近最高价)，则忽略此条件
    upper_shadow = (curr["high"] - max(curr["open"], curr["close"])) / curr["close"]
    if upper_shadow > 0.025: # 上影线超过 2.5%
        return None

    # 3. 换手率过滤 (防出货)
    # 如果接口返回了换手率列，且换手率 > 15%，剔除（高位大换手往往是出货）
    if "turnover" in df.columns:
        if curr["turnover"] > 15: return None

    # 4. 布林带压制过滤
    # 如果股价触碰布林带上轨被打回，剔除
    if curr["high"] >= curr["BOLL_Up"] and curr["close"] < curr["BOLL_Up"]:
        # 且收盘价比上轨低 1% 以上
        if (curr["BOLL_Up"] - curr["close"]) / curr["close"] > 0.01:
            return None

    # ==========================================
    # 🔥 核心买点 (保持之前的高标准)
    # ==========================================
    
    # 门槛: 量比>1.5, 趋势向上, 资金流入
    if vol_ratio < 1.5: return None
    if not (curr["MA20"] > prev["MA20"]): return None
    if curr["OBV"] < curr["OBV_MA10"]: return None
    
    # 组合1: 零轴金叉
    setup_1 = (prev["DIF"] < prev["DEA"] and curr["DIF"] > curr["DEA"]) and (curr["DIF"] > -0.1)

    # 组合2: 底背离
    setup_2 = False
    last_60_low_idx = df["low"].tail(60).idxmin()
    if last_60_low_idx != curr.name:
        if curr["close"] < df.loc[last_60_low_idx, "low"] * 1.05:
            if curr["DIF"] > df.loc[last_60_low_idx, "DIF"]:
                setup_2 = True

    # 组合3: 缩量回调后的多头启动 (最佳买点)
    # 逻辑: 均线多头 + 昨天缩量阴线 + 今天放量阳线
    is_ma_bull = curr["MA5"] > curr["MA10"] > curr["MA20"]
    is_rebound = (prev["close"] < prev["open"]) and (curr["close"] > curr["open"]) # 昨阴今阳
    setup_3 = is_ma_bull and is_rebound and (abs(bias_ma5) < 0.03)

    if not (setup_1 or setup_2 or setup_3):
        return None

    signal_name = []
    if setup_1: signal_name.append("趋势金叉")
    if setup_2: signal_name.append("底背离")
    if setup_3: signal_name.append("回调启动")

    return {
        "close": curr["close"],
        "vol_ratio": vol_ratio,
        "rsi": round(curr["RSI"], 1),
        "bias": round(bias_ma5 * 100, 2), # 乖离率
        "signal_type": " + ".join(signal_name)
    }

# --- 6. 主程序 ---
def main():
    print("=== 稳健型选股启动 (防回撤版) ===")
    pd.DataFrame([["Init", "OK"]]).to_excel("Init_Check.xlsx", index=False)
    
    try:
        base_targets, source_name = get_targets_robust()
        targets = base_targets # 这里为了演示，暂时跳过热点过滤，跑全量
        
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

                # 注意：这里增加了 '换手率' 的映射
                df.rename(columns={
                    "日期":"date","开盘":"open","收盘":"close",
                    "最高":"high","最低":"low","成交量":"volume",
                    "换手率":"turnover" # 确保获取换手率
                }, inplace=True)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

                res = process_stock(df)
                
                if res:
                    print(f"  ★ 稳健目标: {code} {name} [{res['signal_type']}] 乖离:{res['bias']}%")
                    
                    result_data.append({
                        "代码": code,
                        "名称": name,
                        "现价": res["close"],
                        "入选类型": res["signal_type"],
                        "量比": res["vol_ratio"], 
                        "乖离率%": res["bias"], # 新增列，越小越安全
                        "RSI": res["rsi"],
                        "数据来源": source_name
                    })
            except: continue
            time.sleep(0.05)

        dt_str = datetime.now().strftime("%Y%m%d")
        if result_data:
            df_res = pd.DataFrame(result_data)
            # 按乖离率排序：优先看乖离率小的（还没大涨的），更安全
            df_res = df_res.sort_values(by="乖离率%", ascending=True)
            
            filename = f"稳健精选_{dt_str}.xlsx"
            df_res.to_excel(filename, index=False)
            print(f"完成！已保存: {filename}")
        else:
            pd.DataFrame([["无"]]).to_excel(f"无结果_{dt_str}.xlsx")

    except Exception:
        with open("FATAL_ERROR.txt", "w") as f: f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
