import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict

# ------------------ Indicator helpers ------------------ #
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    mf = tp * df['Volume']
    tp_diff = tp.diff()
    pos_mf = mf.where(tp_diff > 0, 0.0)
    neg_mf = mf.where(tp_diff < 0, 0.0)
    pos_sum = pos_mf.rolling(window=period).sum()
    neg_sum = neg_mf.rolling(window=period).sum().abs()
    mfi = 100 - (100 / (1 + (pos_sum / neg_sum.replace(0, np.nan))))
    return mfi.fillna(0)

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']; low = df['Low']; close = df['Close']
    prev_high = high.shift(1); prev_low = low.shift(1); prev_close = close.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0)

# ------------------ Core function ------------------ #
def get_stock_data(ticker_symbol: str, market: str, start_date: str, end_date: str) -> None:
    """
    Fetches OHLCV for ticker_symbol, computes indicators, saves Excel and metadata JSON.
    """
    try:
        print(f"Fetching data for {ticker_symbol}...")
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(start=start_date, end=end_date, auto_adjust=False)

        if hist.empty:
            print(f"Could not find historical data for {ticker_symbol}. Skipping.")
            return

        hist.index = pd.to_datetime(hist.index)
        hist.sort_index(inplace=True)
        info: Dict = stock.info or {}

        hist.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)

        # Core price features
        hist['CloseYest'] = hist['Close'].shift(1)
        hist['Daily_Return'] = hist['Close'].pct_change().fillna(0)
        hist['Log_Return'] = np.log1p(hist['Daily_Return']).fillna(0)
        hist['Cum_Return'] = (1 + hist['Daily_Return']).cumprod() - 1

        # Rolling volume averages
        hist['Vol_Roll_7'] = hist['Volume'].rolling(7, min_periods=1).mean()
        hist['Vol_Roll_21'] = hist['Volume'].rolling(21, min_periods=1).mean()
        hist['Vol_Roll_63'] = hist['Volume'].rolling(63, min_periods=1).mean()
        hist['Vol_to_21'] = hist['Volume'] / hist['Vol_Roll_21'].replace(0, np.nan)
        hist['Volume_Spike'] = (hist['Volume'] > 2 * hist['Vol_Roll_21']).astype(int)

        # Trend and moving averages
        hist['SMA_20'] = hist['Close'].rolling(20, min_periods=1).mean()
        hist['SMA_50'] = hist['Close'].rolling(50, min_periods=1).mean()
        hist['SMA_200'] = hist['Close'].rolling(200, min_periods=1).mean()
        hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()

        # Crosses
        hist['Golden_Cross'] = ((hist['SMA_50'] > hist['SMA_200']) & (hist['SMA_50'].shift(1) <= hist['SMA_200'].shift(1))).astype(int)
        hist['Death_Cross'] = ((hist['SMA_50'] < hist['SMA_200']) & (hist['SMA_50'].shift(1) >= hist['SMA_200'].shift(1))).astype(int)

        # Momentum indicators
        hist['Momentum_10'] = hist['Close'] - hist['Close'].shift(10)
        macd, macd_signal, macd_hist = compute_macd(hist['Close'])
        hist['MACD'], hist['MACD_Signal'], hist['MACD_Hist'] = macd, macd_signal, macd_hist
        hist['MACD_Bull_Cross'] = ((hist['MACD'] > hist['MACD_Signal']) & (hist['MACD'].shift(1) <= hist['MACD_Signal'].shift(1))).astype(int)
        hist['RSI_14'] = compute_rsi(hist['Close'])
        hist['ATR_14'] = compute_atr(hist)
        hist['Volatility_21d'] = hist['Daily_Return'].rolling(21, min_periods=1).std() * np.sqrt(252)

        # Bollinger Bands
        hist['BB_Mid'] = hist['Close'].rolling(20, min_periods=1).mean()
        hist['BB_STD'] = hist['Close'].rolling(20, min_periods=1).std()
        hist['BB_upper'] = hist['BB_Mid'] + 2 * hist['BB_STD']
        hist['BB_lower'] = hist['BB_Mid'] - 2 * hist['BB_STD']
        hist['BB_bandwidth'] = (hist['BB_upper'] - hist['BB_lower']) / hist['BB_Mid'].replace(0, np.nan)
        hist['BB_pct'] = (hist['Close'] - hist['BB_lower']) / (hist['BB_upper'] - hist['BB_lower']).replace(0, np.nan)

        # Volume-based & trend strength
        hist['OBV'] = (np.sign(hist['Close'].diff().fillna(0)) * hist['Volume']).cumsum()
        hist['MFI_14'] = compute_mfi(hist)
        hist['ADX_14'] = compute_adx(hist)

        # Risk and relative position
        hist['Rolling_Max_Close'] = hist['Close'].cummax()
        hist['Drawdown'] = (hist['Close'] / hist['Rolling_Max_Close']) - 1
        hist['Zscore_20'] = (hist['Close'] - hist['SMA_20']) / hist['BB_STD'].replace(0, np.nan)
        hist['Pct_from_SMA_50'] = (hist['Close'] - hist['SMA_50']) / hist['SMA_50'].replace(0, np.nan)
        hist['Pct_from_SMA_200'] = (hist['Close'] - hist['SMA_200']) / hist['SMA_200'].replace(0, np.nan)
        hist['Gap_Open_Pct'] = (hist['Open'] - hist['CloseYest']) / hist['CloseYest'].replace(0, np.nan)
        hist['Gap_Close_Open_Pct'] = (hist['Close'] - hist['Open']) / hist['Open'].replace(0, np.nan)

        # ---- FINAL CLEAN COLUMN SET ----
        desired_columns = [
            'Open','High','Low','Close','CloseYest','Daily_Return','Log_Return','Cum_Return',
            'Volume','Vol_Roll_7','Vol_Roll_21','Vol_Roll_63','Vol_to_21','Volume_Spike',
            'SMA_20','SMA_50','SMA_200','EMA_12','EMA_26',
            'Pct_from_SMA_50','Pct_from_SMA_200','Momentum_10',
            'MACD','MACD_Signal','MACD_Hist','MACD_Bull_Cross',
            'RSI_14','MFI_14',
            'BB_Mid','BB_upper','BB_lower','BB_bandwidth','BB_pct',
            'ATR_14','Volatility_21d','ADX_14',
            'OBV','Drawdown','Zscore_20',
            'Gap_Open_Pct','Gap_Close_Open_Pct',
            'Golden_Cross','Death_Cross'
        ]

        final_df = hist[desired_columns].copy()

        # Remove timezone from index for Excel compatibility
        if getattr(final_df.index, 'tz', None) is not None:
            try:
                final_df.index = final_df.index.tz_convert(None)
            except Exception:
                final_df.index = final_df.index.tz_localize(None)

        # Save metadata JSON separately
        metadata = {
            'ticker': ticker_symbol,
            'market': market,
            'shortName': info.get('shortName'),
            'longName': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'marketCap': info.get('marketCap'),
            'averageVolume': info.get('averageVolume'),
            'beta': info.get('beta'),
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'dividendYield': info.get('dividendYield'),
            'trailingEps': info.get('trailingEps'),
            'lastUpdated': datetime.now(timezone.utc).isoformat()
        }

        # ---- SAVE FILES ----
        filename_ticker = ticker_symbol.replace('.', '_')
        output_dir = os.path.join('stock_data', market)
        os.makedirs(output_dir, exist_ok=True)
        excel_path = os.path.join(output_dir, f'{filename_ticker}.xlsx')
        json_path = os.path.join(output_dir, f'{filename_ticker}_meta.json')

        final_df.to_excel(excel_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✅ Saved {ticker_symbol} data to {excel_path}")

    except Exception as e:
        print(f"❌ Error fetching {ticker_symbol}: {e}")

# ------------------ Main orchestration ------------------ #
def main():
    # Lists (same as your original lists)
    # 🟩 ~150 Indian + ~150 US stocks (total ≈ 300)
    indian_stocks = [
    # --- NIFTY 50 / Large Caps ---
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','HINDUNILVR.NS','BHARTIARTL.NS','ITC.NS','SBIN.NS',
    'BAJFINANCE.NS','HCLTECH.NS','KOTAKBANK.NS','LT.NS','MARUTI.NS','SUNPHARMA.NS','AXISBANK.NS','TITAN.NS','ONGC.NS',
    'NTPC.NS','TATAMOTORS.NS','WIPRO.NS','ULTRACEMCO.NS','ASIANPAINT.NS','COALINDIA.NS','ADANIPORTS.NS','POWERGRID.NS',
    'NESTLEIND.NS','M&M.NS','IOC.NS','BAJAJ-AUTO.NS','TATASTEEL.NS','JSWSTEEL.NS','INDUSINDBK.NS','GRASIM.NS',
    'PIDILITIND.NS','HINDALCO.NS','TECHM.NS','BRITANNIA.NS','CIPLA.NS','EICHERMOT.NS','DRREDDY.NS','HEROMOTOCO.NS',
    'HDFCLIFE.NS','DIVISLAB.NS','VEDL.NS','GAIL.NS','SIEMENS.NS','INDIGO.NS','AMBUJACEM.NS','TATACONSUM.NS',
    'SHREECEM.NS','BPCL.NS','UPL.NS','APOLLOHOSP.NS',

    # --- Mid Caps & Liquid Nifty Next 50 names ---
    'ICICIPRULI.NS','LTIM.NS','HAVELLS.NS','BOSCHLTD.NS','DLF.NS','TRENT.NS','CHOLAFIN.NS','BANKBARODA.NS','BEL.NS',
    'HAL.NS','PNB.NS','SAIL.NS','MARICO.NS','DABUR.NS','COLPAL.NS','GODREJCP.NS','IRCTC.NS','MUTHOOTFIN.NS',
    'NMDC.NS','BERGEPAINT.NS','CANBK.NS','BANDHANBNK.NS','IDFCFIRSTB.NS','BIOCON.NS','AUROPHARMA.NS','LUPIN.NS',
    'ZYDUSLIFE.NS','PAGEIND.NS','PGHH.NS','MCDOWELL-N.NS','ADANIPOWER.NS','INDHOTEL.NS','POLYCAB.NS','JUBLFOOD.NS',
    'SRF.NS','PIIND.NS','TVSMOTOR.NS','ASHOKLEY.NS','APOLLOTYRE.NS','MFSL.NS','TATAPOWER.NS','MGL.NS','IEX.NS',
    'KPITTECH.NS','PERSISTENT.NS','IRFC.NS','RVNL.NS','NHPC.NS','COFORGE.NS','HONAUT.NS','CROMPTON.NS','LTI.NS',
    'OBEROIRLTY.NS','ESCORTS.NS','GRANULES.NS','NAUKRI.NS','AUBANK.NS','ICRA.NS','ABCAPITAL.NS','HAPPSTMNDS.NS',
    'KPRMILL.NS','KEI.NS','PNCINFRA.NS','CAMS.NS','SYNGENE.NS','ZENSARTECH.NS','CYIENT.NS','DEEPAKNTR.NS','ALKEM.NS',
    'BAJAJHLDNG.NS','FEDERALBNK.NS','CUB.NS','AIAENG.NS','ABB.NS','BHEL.NS','EXIDEIND.NS','GODREJPROP.NS','UBL.NS',
    'RBLBANK.NS','TORNTPHARM.NS','SUNTV.NS','SUPREMEIND.NS','VGUARD.NS','RADICO.NS','MPHASIS.NS','MANAPPURAM.NS',
    'PVRINOX.NS','PRESTIGE.NS','GLENMARK.NS','HINDCOPPER.NS','AJANTPHARM.NS','JINDALSTEL.NS','GSPL.NS','TATACHEM.NS'
]

    us_stocks = [
    # --- Mega Caps / Dow / S&P leaders ---
    'AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','BRK-B','V','JPM','XOM','WMT','UNH','JNJ','MA','PG','HD','CVX',
    'MRK','KO','PEP','AVGO','COST','ADBE','BAC','CRM','MCD','ACN','PFE','TMO','CSCO','ABT','NFLX','LIN','DHR','DIS',
    'AMD','WFC','VZ','INTC','CMCSA','PM','NEE','NKE','UPS','CAT','HON','IBM','LOW','COP','UNP','BA','RTX','GS','INTU',
    'AMGN','MDT','SBUX','DE','BLK','GE','AMT','ISRG','BKNG','GILD','CVS','LMT','SPGI','PLD','SYK','TJX','ELV','ZTS',
    'T','AXP','MO','C','PYPL','SCHW','ADI','CI','DUK','MMC','SO','NOW','CB','TMUS','CL','MMM','FISV','GM','PNC','BSX',
    'ETN','MDLZ','TGT','BDX','ADP','USB','REGN','HUM','NSC','AON','ITW','SLB','ABBV','QCOM','TXN','BK','SPG','ALL',
    'ROST','KMB','EOG','EW','PSA','SHW','FDX','D','MS','CMG','HCA','NOC','EXC','MCO','LRCX','ORLY','LHX','EQIX','GMAB',
    'AFL','TRV','EMR','F','GIS','KR','NEM','PH','PGR','DG','MSI','CME','ADM','MAR','DLR','AEP','OKE','TROW','WBD','EBAY',
    'PAYX','KMI','DHI','PRU','MET','SRE','APD','NUE','HLT','AZO','LEN','FAST','OTIS','CTAS','CLX','HPQ','WELL','HES'
]


    # 15 years by default
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print("--- Starting Stock Data Collection ---")
    print(f"Data period: {start_date_str} to {end_date_str}")

    print("\n--- Processing Indian Market Stocks ---")
    for ticker in indian_stocks:
        get_stock_data(ticker, 'India', start_date_str, end_date_str)

    print("\n--- Processing US Market Stocks ---")
    for ticker in us_stocks:
        get_stock_data(ticker, 'US', start_date_str, end_date_str)

    print("\n--- Data Collection Complete ---")

if __name__ == '__main__':
    main()
