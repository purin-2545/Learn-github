# =====================================
# 🧠 Section 1: Global Config & MT5 Init
# =====================================
import os
import sys
import time
import ctypes
import logging
import random
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import tensorflow as tf
import talib
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
try:
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        CSVLogger,
        EarlyStopping,
        ReduceLROnPlateau,
    )
except ModuleNotFoundError:
    from keras.callbacks import (
        ModelCheckpoint,
        CSVLogger,
        EarlyStopping,
        ReduceLROnPlateau,
    )
try:
    from tensorflow.keras import mixed_precision
    from tensorflow.keras.mixed_precision import LossScaleOptimizer, set_global_policy
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError:
    from keras import mixed_precision
    from keras.mixed_precision import LossScaleOptimizer
    from keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ─── เปิด Memory Growth ไม่ให้จอง VRAM เต็มทีเดียว ─────────
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ─── ตั้ง Mixed-Precision ให้ใช้ FP16 บน Tensor Cores ────────
mixed_precision.set_global_policy('mixed_float16')
base_opt = Adam(learning_rate=1e-4)
opt = LossScaleOptimizer(base_opt)

# -----------------------
# Ensure necessary folders
# -----------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models/ea22", exist_ok=True)
os.makedirs("models/ea27", exist_ok=True)

# ⚙️ Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__) 

# 🔐 Seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 📌 Global Parameters
LOOK_BACK       = 30
MAX_HOLD_PERIOD = 18
RISK_PERCENT    = 0.01
SYMBOL          = "EURUSDm"
MIN_TRADES      = 50

# ✅ Dynamic SL/TP Calculation (แทน SL_FACTOR / TP_FACTOR แบบคงที่)
def calculate_sl_tp_dynamic(entry_price: float, atr: float, position_type: str, rr_ratio: float = 2.0) -> tuple:
    """
    คำนวณ Stop Loss และ Take Profit แบบปรับอัตโนมัติตาม ATR และ Risk-Reward Ratio
    """
    sl_range = np.clip(atr * 3, 0.0005, 0.005)  # ป้องกัน SL สั้นเกินหรือยาวเกิน
    tp_range = sl_range * rr_ratio

    if position_type.lower() == 'long':
        sl = entry_price - sl_range
        tp = entry_price + tp_range
    elif position_type.lower() == 'short':
        sl = entry_price + sl_range
        tp = entry_price - tp_range
    else:
        raise ValueError("position_type must be 'long' or 'short'")

    return sl, tp

# ✅ Optional: ฟังก์ชันใช้ S/R Levels

def calculate_sl_tp_from_sr(df: pd.DataFrame, idx: int, position_type: str) -> tuple:
    candle = df.iloc[idx]
    close = candle['close']
    atr = candle['ATR']

    if position_type == 'long':
        sl = candle.get('support', close - atr)
        tp = candle.get('resistance', close + atr * 2)
    elif position_type == 'short':
        sl = candle.get('resistance', close + atr)
        tp = candle.get('support', close - atr * 2)
    else:
        raise ValueError("Invalid position_type")

    if abs(tp - close) < atr:
        tp = close + atr if position_type == 'long' else close - atr
    if abs(close - sl) < atr:
        sl = close - atr if position_type == 'long' else close + atr

    return sl, tp

def add_confirm_entry_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่มฟีเจอร์ confirm_entry: 1 หากแท่งเทียนได้รับการยืนยัน, 0 หากไม่ผ่าน
    ใช้ logic แท่งเทียน + RSI + S/R แบบ simplified (vectorized)
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    cond_long = (
        (df['close'] > df['open']) &
        (df['RSI'] < 70) &
        (df['close'] > prev_open) &
        (df['open'] < prev_close) &
        ((df['resistance'] - df['close']) / df['close'] >= 0.01)
    )
    cond_short = (
        (df['close'] < df['open']) &
        (df['RSI'] > 30) &
        (df['open'] > prev_close) &
        (df['close'] < prev_open) &
        ((df['close'] - df['support']) / df['close'] >= 0.01)
    )
    df['confirm_entry'] = 0
    df.loc[cond_long | cond_short, 'confirm_entry'] = 1
    return df

# =============================
# 🧩 DLL Load (if needed)
# =============================
def load_dll(path: str):
    try:
        cd = ctypes.CDLL(os.path.abspath(path))
        cd.add.argtypes = (ctypes.c_int, ctypes.c_int)
        cd.add.restype = ctypes.c_int
        logger.info("DLL loaded test: %d", cd.add(10, 20))
        return cd
    except Exception as e:
        logger.error("DLL load error: %s", e)
        sys.exit(1)

DLL_PATH = r"C:\Users\ACE\OneDrive - Khon Kaen University\Desktop\MyLibrary\x64\Debug\MyLibrary.dll"
DLL = load_dll(DLL_PATH)

def get_account_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 100000.0

# =====================================
# 📊 Section 2: Data Loader & Indicators
# =====================================
import talib

# 🔁 Load CSV
def load_csv_data(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    data = {}
    for tf, path in file_paths.items():
        df = load_csv(path)  # load_csv ตั้ง index_col='time'
        data[tf] = df.sort_index()  # เรียง index (DatetimeIndex) ให้เรียงจากน้อย->มาก
    return data

def create_features(df: pd.DataFrame, dfs_other: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    # 2) ถ้ามีข้อมูล M15/H1 ให้ merge ก่อน
    if 'M15' in dfs_other and 'H1' in dfs_other:
        df = merge_timeframes(df, [dfs_other['M15'], dfs_other['H1']])

    # 3) คำนวณ indicator หลัก
    df = calculate_indicators(df)

    # 4) คำนวณ support/resistance
    df = calculate_support_resistance(df, window=20)

    # 5) คำนวณ volatility
    df = add_volatility_features(df)

    # 6) สร้าง lagged features
    df = add_lagged_features(df, cols=['RSI','MACD'], lags=[1,2,3])

    # 7) ฟีเจอร์ confirm entry
    df = add_confirm_entry_feature(df)

    # 8) dropna ครั้งเดียวตอนท้าย
    df = df.dropna()
    return df

def prepare_datasets(data: dict) -> list:
    """
    เตรียมชุดข้อมูลสำหรับ training/validation
    - ใช้เฉพาะ 'M5' เป็นหลัก
    - สร้าง label ตามราคาปิดหลัง MAX_HOLD_PERIOD แท่ง
    - แบ่งชุดด้วย TimeSeriesSplit
    คืนค่า list ของ tuples: (X_train, X_val, y_train, y_val)
    """
    df = create_features(data['M5'])
    # ตัดคอลัมน์ target และสร้าง X, y
    X = df.drop(columns=['close'])
    y = (df['close'].shift(-MAX_HOLD_PERIOD) > df['close']).astype(int)

    # กำจัดแถวท้ายที่ label เป็น NaN
    valid_idx = y.dropna().index
    X, y = X.loc[valid_idx], y.loc[valid_idx]

    # แบ่งชุดข้อมูลตามเวลา
    tscv = TimeSeriesSplit(n_splits=3)
    splits = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        splits.append((X_train, X_val, y_train, y_val))

    return splits

# 🧮 Indicator Calculation
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI']           = talib.RSI(df['close'], timeperiod=14)
    df['EMA_10']        = talib.EMA(df['close'], timeperiod=10)
    df['EMA_50']        = talib.EMA(df['close'], timeperiod=50)
    df['EMA_Crossover'] = df['EMA_10'] - df['EMA_50']
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    df['ATR']           = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['OBV']           = talib.OBV(df['close'], df.get('tick_volume', df.get('volume', np.zeros(len(df)))))
    df['ADX']           = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    bb_u, bb_m, bb_l    = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_upper']      = bb_u
    df['BB_middle']     = bb_m
    df['BB_lower']      = bb_l
    df['BB_width']      = bb_u - bb_l
    df['CCI']           = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

    df['close_change']  = df['close'].pct_change()
    df['hour']          = df.index.hour
    df['day_of_week']   = df.index.dayofweek

    # 🔥 เพิ่ม Volatility Indicators
    df['HistVol']       = df['close'].rolling(10).std()
    df['GK_vol']        = 0.5 * (np.log(df['high'] / df['low']) ** 2) - \
                          (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
    df['Parkinson_vol'] = (1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2)

    return df

# 📈 Support / Resistance
def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['support']    = df['low'].rolling(window=window, min_periods=1).min()
    df['resistance'] = df['high'].rolling(window=window, min_periods=1).max()
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    # อาจรวม Garman-Klass, Parkinson, ATR (ซึ่ง ATR คือตัวเดิม), plus rolling std
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['HistVol_10'] = df['close'].rolling(window=10).std()
    df['GK_vol'] = 0.5 * (np.log(df['high'] / df['low'])**2) - (2*np.log(2)-1)*(np.log(df['close']/df['open'])**2)
    df['Parkinson_vol'] = (1/(4*np.log(2))) * (np.log(df['high']/df['low'])**2)
    return df

def add_lagged_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

# 🔗 Merge Multi-Timeframe
def merge_timeframes(df_main, dfs_other, suffixes=['M15', 'H1']):
    df_merged = df_main.copy()
    for df, suffix in zip(dfs_other, suffixes):
        df = df.copy()
        df.columns = [f"{col}_{suffix}" for col in df.columns]
        df_merged = df_merged.join(df, how='left').fillna(method='ffill')
    return df_merged

# 🔍 Add Regime, Orderbook, Divergence, Momentum
def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    df['regime_trend'] = (df['ADX'] > 25).astype(int)
    df['momentum_3'] = df['close'].pct_change(3)
    df['momentum_5'] = df['close'].pct_change(5)
    df['lag1_close'] = df['close'].shift(1)

    # RSI Divergence & EMA Cross TF
    if 'RSI_H1' in df.columns and 'RSI_M15' in df.columns:
        df['RSI_Divergence_H1_M15'] = df['RSI_H1'] - df['RSI_M15']
    if 'EMA_10_H1' in df.columns and 'EMA_10_M15' in df.columns:
        df['EMA_Cross_H1_M15'] = df['EMA_10_H1'] - df['EMA_10_M15']

    df['volatility_5'] = df['close'].rolling(5).std()
    df['RSI_lag1']     = df['RSI'].shift(1)
    df['MACD_lag1']    = df['MACD'].shift(1)
    return df.dropna()

# 📉 Orderbook Feature
def add_orderbook_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    book = mt5.market_book_get(symbol)
    if not book:
        df['order_imbalance'] = 0.0
        return df
    bids = sum(l.volume for l in book if l.type == mt5.BOOK_TYPE_BUY)
    asks = sum(l.volume for l in book if l.type == mt5.BOOK_TYPE_SELL)
    df['order_imbalance'] = (bids - asks) / max(bids + asks, 1)
    return df

def create_labels_from_price(
        data_pca: np.ndarray,
        prices: np.ndarray,
        look_back: int,
        hold_bars: int,
        thr_short: float,
        thr_long: float
    ) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    n = len(prices)
    assert data_pca.shape[0] == n, "data_pca และ prices ต้องมีความยาวเท่ากัน"
    max_i = n - look_back - hold_bars
    for i in range(max_i):
        seq = data_pca[i : i + look_back]  # shape=(look_back, n_features_pca)
        ret = prices[i + look_back + hold_bars] / prices[i + look_back] - 1.0

        if ret > thr_long:
            lbl = 2
        elif ret < -thr_short:
            lbl = 0
        else:
            lbl = 1

        X_list.append(seq)
        y_list.append(lbl)

    X = np.array(X_list)                 # (n_seq, look_back, n_features_pca)
    y = to_categorical(y_list, num_classes=3)
    return X, y

def create_labels_from_price_nonzero(
        data_pca: np.ndarray,
        prices: np.ndarray,
        look_back: int,
        hold_bars: int,
        thr_short: float,
        thr_long: float
    ) -> tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    max_i = n - look_back - hold_bars

    # 1) คำนวณ forward_returns และ mask_nonzero
    forward_returns = np.empty(max_i, dtype=np.float32)
    for i in range(max_i):
        forward_returns[i] = prices[i + look_back + hold_bars] / prices[i + look_back] - 1.0

    # mask of positions where return != 0
    mask_nonzero = forward_returns != 0.0
    indices = np.nonzero(mask_nonzero)[0]  # array ของ i ที่ ret != 0

    X_list, y_list = [], []
    for i in indices:
        seq = data_pca[i : i + look_back]  # shape = (look_back, n_features_pca)
        ret = forward_returns[i]
        if ret > thr_long:
            lbl = 2
        elif ret < -thr_short:
            lbl = 0
        else:
            lbl = 1
        X_list.append(seq)
        y_list.append(lbl)

    X = np.array(X_list)
    # ถ้าไม่มีโครงสร้าง hold/long/short เลย อย่าขึ้น error แต่ return empty
    if len(y_list) == 0:
        return np.empty((0, look_back, data_pca.shape[1])), np.empty((0, 3))
    y = to_categorical(y_list, num_classes=3)
    return X, y

# =============================================
# 🎯 Section 3: Dataset, Threshold, Augmentation
# =============================================

from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from typing import Tuple, Union, Optional

# 🎯 Threshold สำหรับกำหนด long/short
def calculate_trade_threshold(
    df: pd.DataFrame,
    atr_period: int = 14,
    multiplier: float = 0.5
) -> float:
    """
    คำนวณ threshold ด้วย ATR แบบ rolling mean
    คืนค่า float เดียว: avg_ATR * multiplier
    """
    # 1) คำนวณ ATR ให้ถูกต้อง (high, low, close ต้องส่งเข้าไป)
    atr = talib.ATR(
        df['high'],
        df['low'],
        df['close'],
        timeperiod=atr_period
    )
    # 2) เอา ATR มาหาค่า rolling mean ย้อนหลัง atr_period แท่ง
    avg_atr = atr.rolling(window=atr_period, min_periods=1).mean().iloc[-1]
    # 3) คืนค่า threshold
    return avg_atr * multiplier

# 🎯 Create Classification Dataset
def create_dataset_classification(
    data: np.ndarray,
    look_back: int,
    threshold: Union[float, Tuple[float,float]]
):
    # ถ้าเป็น tuple แจกออกมา
    if isinstance(threshold, (list,tuple,np.ndarray)):
        thr_short, thr_long = threshold
    else:
        thr_short = thr_long = threshold

    X, y = [], []
    for i in range(len(data) - look_back - 1):
        seq = data[i : i+look_back]
        ret = data[i+look_back+1, 0] - data[i+look_back, 0]
        if   ret >  thr_long: lbl = 2
        elif ret < -thr_short: lbl = 0
        else:                  lbl = 1
        X.append(seq)
        y.append(lbl)
    X = np.array(X)
    y = to_categorical(y, num_classes=3)
    return X, y

# 🔁 Data Augmentation (เพิ่ม noise)
def augment_data(X: np.ndarray, noise_level: float = 0.001) -> np.ndarray:
    noise = np.random.normal(0, noise_level, X.shape)
    return np.concatenate([X, X + noise], axis=0)

# ⚖️ Oversample dataset ให้ balance
def oversample_dataset(X: np.ndarray, y: np.ndarray, look_back: int = LOOK_BACK):
    y_labels = np.argmax(y, axis=1)
    if len(np.unique(y_labels)) < 2:
        return X, y  # ไม่มีประโยชน์จะ oversample
    # กำหนด k_neighbors = min(5, smallest_class_count-1)
    counts = np.bincount(y_labels)
    k = max(1, min(5, counts.min() - 1))
    sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=42)
    X_flat = X.reshape(X.shape[0], -1)
    X_res, y_res = sm.fit_resample(X_flat, y_labels)
    X_res = X_res.reshape(-1, look_back, X.shape[2])
    y_res = to_categorical(y_res, num_classes=3)
    return X_res, y_res

# ✅ เช็คว่ามีข้อมูลพอไหม
def check_dataset_sufficiency(df: pd.DataFrame, look_back: int, hold_bars: int, min_trades: int):
    required = (look_back + hold_bars + 1) * min_trades
    avail = len(df)
    logger.info(f"Dataset check: need ≥{required}, have {avail}")
    if avail < required:
        logger.error(f"❌ Insufficient data: {avail} < {required}")
        sys.exit(1)

# ==================================
# 🧠 Section 4: Model Builders
# ==================================
try:
    # หากใช้ TensorFlow ≥2.x
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Layer, Input, LSTM, Dense, Dropout, Bidirectional,
        Conv1D, MaxPooling1D, GlobalAveragePooling1D,
        MultiHeadAttention, LayerNormalization, Add
    )
    from tensorflow.keras.regularizers import l2
except ModuleNotFoundError:
    # fallback ไป standalone Keras
    from keras.models import Sequential, Model
    from keras.layers import (
        Layer, Input, LSTM, Dense, Dropout, Bidirectional,
        Conv1D, MaxPooling1D, GlobalAveragePooling1D,
        MultiHeadAttention, LayerNormalization
    )
    from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tcn import TCN

# ✅ Custom Attention Layer
class AttentionLayer(Layer):
    def build(self, input_shape):
        # input_shape = (batch_size, timesteps, features)
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),   # features × 1
            initializer="normal",
            name="att_weight"
        )
        self.b = self.add_weight(
            shape=(1, 1),                 # bias scalar (broadcast ได้)
            initializer="zeros",
            name="att_bias"
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, timesteps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch, timesteps, 1)
        a = K.softmax(e, axis=1)               # normalize over timesteps
        # sum weighted features ตาม timesteps → shape = (batch, features)
        return K.sum(x * a, axis=1)

    def compute_output_shape(self, input_shape):
        # output shape = (batch, features)
        return (input_shape[0], input_shape[-1])

# ✅ LSTM + Attention Deep Model
def build_model_lstm_att_hp(look_back: int, n_features: int) -> Model:
    inp = Input(shape=(look_back, n_features))
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(inp)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
    x = AttentionLayer()(x)
    out = Dense(3, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    # ถ้าใช้ mixed_precision
    base_opt = Adam(learning_rate=1e-4)
    opt = LossScaleOptimizer(base_opt)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ✅ CNN + LSTM
def build_model_cnn_lstm(look_back: int, n_features: int) -> Model:
    inp = Input(shape=(look_back, n_features))
    x = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.4)(x)
    # สมมติ look_back=30 → หลัง pool เหลือ length=14 (ถ้า valid padding)
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    out = Dense(3, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    base_opt = Adam(learning_rate=1e-4)
    opt = LossScaleOptimizer(base_opt)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_tcn(look_back, n_features, filters=64, kernel_size=3):
    inp = Input(shape=(look_back, n_features), dtype="float16")
    x   = inp
    # ใช้ 3 เลเยอร์ causal conv + residual
    for d in [1,2,4]:
        y = Conv1D(filters, kernel_size,
                   dilation_rate=d,
                   padding="causal",
                   activation="relu")(x)
        y = Dropout(0.4)(y)
        # residual
        if y.shape[-1] != x.shape[-1]:
            x_proj = Dense(filters)(x)
        else:
            x_proj = x
        x = tf.keras.layers.Add()([x_proj, y])

    x   = GlobalAveragePooling1D()(x)
    out = Dense(3, activation="softmax", dtype="float32")(x)  # cast back to float32
    base_opt = Adam(1e-4)
    opt      = LossScaleOptimizer(base_opt)
    model = Model(inp, out)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_model_transformer(
    look_back, n_features,
    head_size=64, num_heads=4,
    ff_dim=128, dropout=0.1, n_blocks=4
):
    inp = Input(shape=(look_back, n_features), dtype="float16")
    x   = inp
    for _ in range(n_blocks):
        # multi-head attention
        y = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
        y = Dropout(dropout)(y)
        # residual + norm
        res = tf.cast(x, y.dtype)
        x   = LayerNormalization(epsilon=1e-4)(res + y)

        # FFN
        z = Dense(ff_dim, activation="relu")(x)
        z = Dropout(dropout)(z)
        z = Dense(n_features)(z)
        res2 = tf.cast(x, z.dtype)
        x    = LayerNormalization(epsilon=1e-4)(res2 + z)

    x   = GlobalAveragePooling1D()(x)
    x   = Dropout(dropout)(x)
    out = Dense(3, activation="softmax", dtype="float32")(x)
    base_opt = Adam(1e-4)
    opt      = LossScaleOptimizer(base_opt)
    model = Model(inp, out)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_model_tft(
    look_back, n_features,
    head_size=64, num_heads=4,
    ff_dim=128, dropout=0.1, n_blocks=3
):
    inp = Input(shape=(look_back, n_features), dtype="float16")
    x   = inp
    for _ in range(n_blocks):
        # attention block
        y = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
        y = Dropout(dropout)(y)
        res = tf.cast(x, y.dtype)
        x   = LayerNormalization(epsilon=1e-4)(res + y)

        # FFN
        z = Dense(ff_dim, activation="relu")(x)
        z = Dropout(dropout)(z)
        z = Dense(n_features)(z)
        res2 = tf.cast(x, z.dtype)
        x    = LayerNormalization(epsilon=1e-4)(res2 + z)

    # ตามด้วย LSTM
    x   = LSTM(64, return_sequences=False)(x)
    x   = Dropout(dropout)(x)
    out = Dense(3, activation="softmax", dtype="float32")(x)
    base_opt = Adam(1e-4)
    opt      = LossScaleOptimizer(base_opt)
    model = Model(inp, out)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ✅ Meta-Model (สำหรับ ensemble)
def build_meta_model(input_dim: int) -> Model:
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ✅ RF & XGB
def build_rf_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def build_xgb_model(n_classes: int = 3) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators      = 100,
        objective         = "multi:softprob",  # บอกให้เป็น multi-class
        num_class         = n_classes,         # กำหนดจำนวนคลาสให้ตรง
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = 42
    )

# ✅ รวมผล prediction จากทุก base model
def get_base_predictions(models, X, rf_model=None, batch_size=64):
    """
    Predict โดยแบ่งเป็น batch ย่อยเพื่อลด footprint ของ GPU memory
    แล้ว clear_session หลังจากแต่ละโมเดล
    """
    preds = []
    # 1) พยากรณ์แต่ละโมเดลทีละ batch
    for idx, m in enumerate(models):
        if hasattr(m, 'predict'):
            # sklearn model
            arr = m.predict(X)
        else:
            # บังคับให้ predict บน CPU เพื่อเลี่ยง TensorDataset error
            with tf.device('/CPU:0'):
                arr = m.predict(X, batch_size=batch_size, verbose=0)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        print(f"model #{idx} ({type(m).__name__}): {arr.shape}")
        preds.append(arr)

    # 2) พยากรณ์ด้วย RF ถ้ามี
    if rf_model:
        flat = X.reshape(X.shape[0], -1)
        if hasattr(rf_model, 'predict_proba'):
            arr = rf_model.predict_proba(flat)
        else:
            # fallback: reshape ผลลัพธ์ 1-D เป็น 2-D
            arr = rf_model.predict(flat).reshape(-1, 1)
        print(f"rf_model: {arr.shape}")
        preds.append(arr)

    # 3) ตัดทุกอาร์เรย์ให้มีจำนวนแถวเท่ากัน (min_n)
    min_n = min(p.shape[0] for p in preds)
    preds = [p if p.ndim==2 else p.reshape(-1,1) for p in preds]

    K.clear_session()
    gc.collect()
    return np.concatenate(preds, axis=1)

# ============================================
# 🚀 Section 5: Tuning, SHAP, Drift, Save
# ============================================
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
import shap

# 🔁 Tuning TCN
def tune_tcn(X, y, look_back: int, pbounds=None):
    if pbounds is None:
        pbounds = {'filters': (16, 128), 'kernel': (2, 8)}

    y_lbl = np.argmax(y, axis=1)

    def cv(filters, kernel):
        K.clear_session()
        gc.collect()

        model = build_model_tcn(
            look_back, X.shape[2],
            filters=int(filters),
            kernel_size=int(kernel)
        )

        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        h = model.fit(
            X, y_lbl,
            validation_split=0.2,
            epochs=10,
            batch_size=64,
            callbacks=[es],
            verbose=0
        )
        return max(h.history['val_accuracy'])

    opt = BayesianOptimization(f=cv, pbounds=pbounds, random_state=42)
    opt.maximize(init_points=5, n_iter=10)
    logger.info("✅ TCN tuning done: %s", opt.max)
    return opt.max['params']

def tune_tft(X, y, look_back: int, pbounds=None):
    if pbounds is None:
        pbounds = {
            'ff_dim':    (64, 256),
            'head_size': (16,  64),
            'dropout':   (0.0,  0.5)
        }

    # 1) integer labels
    y_lbl = np.argmax(y, axis=1)
    # 2) float32 for Keras
    X = np.asarray(X, dtype=np.float32)

    # 3) cv must accept (ff_dim, head_size, dropout)
    def cv(ff_dim, head_size, dropout):
        K.clear_session()
        gc.collect()

        model = build_model_tft(
            look_back  = look_back,
            n_features = X.shape[2],
            head_size  = int(head_size),
            ff_dim     = int(ff_dim),
            dropout    = float(dropout),   # <— now uses the argument
            n_blocks   = 3,
            num_heads  = 4
        )

        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        tscv = TimeSeriesSplit(n_splits=3)
        vals = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y_lbl[tr_idx], y_lbl[va_idx]
            h = model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=10,
                batch_size=64,
                callbacks=[es],
                verbose=0
            )
            vals.append(max(h.history['val_accuracy']))
        return float(np.mean(vals))

    # 4) now BO will pass dropout
    opt = BayesianOptimization(f=cv, pbounds=pbounds, random_state=42)
    opt.maximize(init_points=3, n_iter=5)

    print("✅ TFT tuning done:", opt.max)
    return opt.max['params']

def explain_with_shap(model, X, feat_names, out_path: Optional[str] = None):
    # เลือก Explainer ตามโมเดล
    if hasattr(model, 'feature_importances_'):
        expl = shap.TreeExplainer(model)
    else:
        expl = shap.DeepExplainer(model, X[:50])
    subset = X[50:100]
    sv = expl.shap_values(subset)
    # วาด summary plot
    shap.summary_plot(sv, features=subset, feature_names=feat_names, show=False)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

# 🔄 Concept Drift Detection
def detect_concept_drift(ref: np.ndarray, new: np.ndarray, alpha=0.01):
    """
    ใช้ KS-test เพื่อตรวจ distribution drift ในแต่ละ feature
    คืน True ถ้า severity ≥ 20%
    """
    drift_count = 0
    n_feat = ref.shape[1]
    for i in range(n_feat):
        _, p = ks_2samp(ref[:, i], new[:, i])
        if p < alpha:
            drift_count += 1
    severity = drift_count / n_feat
    logger.info(f"📉 Drift severity: {severity:.2%}")
    return severity > 0.2

def auto_retrain_if_drift(model, refX, newX, newY):
    if detect_concept_drift(refX, newX):
        # สำรองโมเดลเดิม
        model.save('backup_model.h5')
        logger.warning("⚠️ Concept drift detected → retraining")
        history = model.fit(newX, newY,
                            epochs=10, batch_size=64,
                            validation_split=0.2, verbose=1)
        val_acc = max(history.history.get('val_accuracy', [0]))
        logger.info(f"🔄 Retrained, best val_acc={val_acc:.2%}")

# 💾 Save Model + Scaler + PCA + Tree
def save_model_bundle(models: dict, folder: str = 'models'):
    os.makedirs(folder, exist_ok=True)
    for name, obj in models.items():
        base = os.path.join(folder, name)
        if hasattr(obj, 'save'):  # Keras
            obj.save(base + ".h5")
        elif isinstance(obj, sklearn.base.BaseEstimator):
            joblib.dump(obj, base + ".pkl")
        else:
            with open(base + ".bin", "wb") as f:
                pickle.dump(obj, f)
        logger.info("📦 Saved: %s", base)

# ================================================
# 🎯 FINAL: main() — ฝึก EA22 + EA27 และ Save
# ================================================
import pickle
import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd
import joblib
import warnings
import gc
import keras_tuner as kt
import xgboost as xgb
import tensorflow as tf
from hmmlearn.hmm import GaussianHMM
from typing import Optional
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.feature_selection import mutual_info_classif
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from collections import OrderedDict
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import CSVLogger
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tcn import TCN
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict
from tensorflow.keras.callbacks import TensorBoard
from xgboost import XGBClassifier
from sklearn.utils import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.isotonic import IsotonicRegression

# 🧠 Load all model builder functions
# Assumes functions like build_model_lstm_att_hp, build_model_cnn_lstm etc. are defined elsewhere
        
# === Tick-level Volume Profile ===
def get_volume_profile(symbol: str, n_bins: int = 20, duration_sec: int = 300):
    now = time.time()
    ticks = mt5.copy_ticks_from(symbol, now - duration_sec, duration_sec, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return np.zeros(n_bins)
    df = pd.DataFrame(ticks)
    price_bins = np.linspace(df['last'].min(), df['last'].max(), n_bins + 1)
    df['bin'] = np.digitize(df['last'], price_bins)
    vol_profile = df.groupby('bin')['volume'].sum().reindex(range(1, n_bins + 1), fill_value=0).values
    return vol_profile / (vol_profile.sum() + 1e-8)

# === Orderbook Depth Feature ===
def get_orderbook_features(symbol: str, max_levels: int = 10):
    book = mt5.market_book_get(symbol)
    if not book:
        return np.zeros(max_levels + 3)
    bids = [l for l in book if l.type == mt5.BOOK_TYPE_BUY][:max_levels]
    asks = [l for l in book if l.type == mt5.BOOK_TYPE_SELL][:max_levels]
    bid_vols = np.array([l.volume for l in bids] + [0] * (max_levels - len(bids)))
    ask_vols = np.array([l.volume for l in asks] + [0] * (max_levels - len(asks)))
    imbalance_per_level = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-8)
    spread = asks[0].price - bids[0].price if bids and asks else 0
    total_bid = bid_vols.sum()
    total_ask = ask_vols.sum()
    return np.concatenate([imbalance_per_level, [total_bid, total_ask, spread]])

# === Add to df ===
def enrich_df_with_tick_orderbook(df: pd.DataFrame, symbol: str):
    vp = get_volume_profile(symbol)
    ob = get_orderbook_features(symbol)
    for i in range(len(vp)):
        df[f'vp_bin_{i+1}'] = vp[i]
    for i in range(len(ob)-3):
        df[f'imb_{i+1}'] = ob[i]
    df['total_bid'] = ob[-3]
    df['total_ask'] = ob[-2]
    df['spread'] = ob[-1]
    return df

# === Feature Selection ===
def select_features_by_mutual_info(X: np.ndarray, y: np.ndarray, k: int = 30):
    y_labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y
    scores = mutual_info_classif(X, y_labels)
    top_k_idx = np.argsort(scores)[-k:]
    return X[:, top_k_idx], top_k_idx

# === Autoencoder Feature Extractor ===
def build_autoencoder(input_dim: int, encoding_dim: int = 32) -> Model:
    inp = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(1e-4))(inp)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder = Model(inp, encoded)
    return autoencoder, encoder

# === Meta-Model Builder ===
def build_meta_model(input_dim: int):
    model = Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Deep Model (LSTM) ===
def build_lstm_classifier(input_dim):
    model = Sequential([
        Input(shape=(1, input_dim)),
        LSTM(64),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Deep Model (CNN) ===
def build_cnn_classifier(input_dim):
    model = Sequential([
        Input(shape=(1, input_dim)),
        Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Mock save_model_bundle ===
def save_model_bundle(models: dict, folder: str = 'models'):
    os.makedirs(folder, exist_ok=True)
    for name, obj in models.items():
        path = os.path.join(folder, name)
        if hasattr(obj, 'save'):
            # Keras models
            obj.save(path + ".h5")
        elif isinstance(obj, BaseEstimator):
            # sklearn models
            joblib.dump(obj, path + ".pkl")
        else:
            # objects อื่นๆ
            with open(path + ".bin", "wb") as f:
                pickle.dump(obj, f)
        logger.info("📦 Saved artifact: %s", path)

# === Stub utils ===
def load_csv(path: str) -> pd.DataFrame:
    # 1) อ่านไฟล์ พร้อมแปลงคอลัมน์ time เป็น datetime  
    # 2) ตั้งเป็น index เพื่อให้ downstream ฟังก์ชันใช้ df.index.hour, df.index.dayofweek ได้  
    return pd.read_csv(path, parse_dates=['time'], index_col='time')

# === Multi-Horizon Label Generator ===
def generate_multi_horizon_labels(data: np.ndarray, look_back: int, horizons=[1, 3, 5]):
    X, y = [], []
    for i in range(len(data) - look_back - max(horizons)):
        X.append(data[i:i+look_back])
        future = [data[i+look_back+h, 0] - data[i+look_back, 0] for h in horizons]
        y.append(future)
    return np.array(X), np.array(y)

def detect_market_regime(prices: np.ndarray, n_states: int = 3) -> np.ndarray:
    """
    คืน array ของ regime index สำหรับแต่ละ bar
    ถ้า len(prices)=N คืน regimes.shape = N
    """
    # แปลงเป็น log-returns
    returns = np.diff(np.log(prices + 1e-8)).reshape(-1, 1)  # shape=(N-1,1)
    # สร้างและฝึก HMM
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    model.fit(returns)                   # แยก fit ออกมา
    regimes = model.predict(returns)     # แล้ว predict
    # pad ด้านหน้า 1 ค่าให้ยาวเท่ากับ prices
    regimes = np.insert(regimes, 0, regimes[0])
    return regimes  # shape=(N,)

def prepare_base_dataframe(path: str, symbol: str, dfs_other: Optional[dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    # 1) อ่านไฟล์ CSV
    df = load_csv(path)  # index_col='time'
    
    # 2) สร้างฟีเจอร์หลัก
    if dfs_other:
        df = create_features(df, dfs_other)
    else:
        df = create_features(df, {})

    # 3) เติมข้อมูล tick-level & orderbook
    df = enrich_df_with_tick_orderbook(df, symbol)
    
    # 4) Detect regime จากราคาหลัง enrich เสร็จ
    prices = df['close'].values
    df['regime'] = detect_market_regime(prices, n_states=3)

    return df
    
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

def categorical_focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=1)
    return loss

def cross_validate_ea22(
    X: np.ndarray,
    y: np.ndarray,
    build_base_fns: list,
    build_rf_fn,
    build_meta_fn,
    look_back: int,
    batch_size: int = 64,
    epochs: int = 10,
    n_splits: int = 5,
    long_boost: float = 1.5       # สำหรับกระจายน้ำหนักคลาส Long
):
    y_lbl = np.argmax(y, axis=1)
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, f1_hold_list, f1_macro_list = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_lbl), start=1):
        print(f"\n[EA22 CV] Fold {fold}/{n_splits}")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr_lbl   = y_lbl[tr_idx]
        y_va_lbl   = y_lbl[va_idx]

        # 1) Train base deep models
        trained = []
        for build_fn in build_base_fns:
            m = build_fn(look_back, X_tr.shape[2])
            m.fit(X_tr, y_tr_lbl, epochs=epochs, batch_size=batch_size, verbose=0)
            trained.append(m)

        # 2) Train RF
        rf = build_rf_fn()
        rf.fit(X_tr.reshape(len(X_tr), -1), y_tr_lbl)

        # 3) สร้าง meta-inputs
        meta_X_tr = get_base_predictions(trained, X_tr, rf_model=rf, batch_size=batch_size)
        meta_X_va = get_base_predictions(trained, X_va, rf_model=rf, batch_size=batch_size)

        # 4) เตรียม class_weight_meta + boost คลาส Long (2)
        classes_meta = np.unique(y_tr_lbl)
        cw_vals      = compute_class_weight('balanced', classes=classes_meta, y=y_tr_lbl)
        class_weight_meta = {int(c): float(w) for c,w in zip(classes_meta, cw_vals)}
        class_weight_meta.setdefault(2, 1.0)
        class_weight_meta[2] *= long_boost
        print("  class_weight_meta:", class_weight_meta)

        # 5) Train meta-model
        meta = build_meta_fn(input_dim=meta_X_tr.shape[1])
        meta.compile(optimizer=Adam(1e-4),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        meta.fit(meta_X_tr, y_tr_lbl,
                 epochs=epochs,
                 batch_size=batch_size,
                 class_weight=class_weight_meta,
                 verbose=0)

        # ---- Calibration (Isotonic) ----
        prob_tr   = meta.predict(meta_X_tr)       # shape=(n_tr,3)
        P_hold_tr = prob_tr[:, 1]
        y_hold_tr = (y_tr_lbl == 1).astype(int)

        # ถ้ามีทั้ง Hold และ non-Hold ในชุดฝึก จึง calibrate
        if len(np.unique(y_hold_tr)) == 2:
            # 1) กรอง NaN ออกจาก P_hold_tr
            mask     = ~np.isnan(P_hold_tr)
            P_clean  = P_hold_tr[mask]
            y_clean  = y_hold_tr[mask]
            # 2) Fit isotonic
            iso      = IsotonicRegression(out_of_bounds='clip')
            iso.fit(P_clean, y_clean)
            # 3) สร้างฟังก์ชันช่วยกรอง/เติม NaN แล้ว predict
            def calibrate(p):
                # แทน NaN ด้วยค่า 0.5 แล้วค่อย predict
                p2 = np.nan_to_num(p, nan=0.5)
                return iso.predict(p2)
        else:
            # ถ้าไม่มีทั้งสองคลาส ให้ข้าม calibration
            def calibrate(p):
                # แค่เติม NaN เป็น 0.5 แล้วคืน p ดิบ
                return np.nan_to_num(p, nan=0.5)

        # ---- ใช้ calibrate เวลาพยากรณ์ ----
        prob_va    = meta.predict(meta_X_va)
        raw_hold   = prob_va[:, 1]
        P_hold_cal = calibrate(raw_hold)

        # ---- หา optimal threshold ด้วย F1-macro ----
        y_true      = y_va_lbl
        best_thr, best_f1m = 0.0, 0.0
        for thr in np.linspace(0.30, 0.60, 301):
            y_pred_thr = np.where(
                P_hold_cal > thr,
                1,
                np.argmax(prob_va, axis=1)
            )
            f1m = f1_score(y_true, y_pred_thr, average='macro')
            if f1m > best_f1m:
                best_f1m, best_thr = f1m, thr

        print(f"  best_hold_thr={best_thr:.6f}, F1-macro={best_f1m:.4f}")

        # ---- ประเมินด้วย threshold ที่หาได้ ----
        y_pred = np.where(P_hold_cal > best_thr,
                          1,
                          np.argmax(prob_va, axis=1))

        acc      = accuracy_score(y_true, y_pred)
        f1_hold  = f1_score(y_true, y_pred, labels=[1], average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        print(f"  accuracy: {acc:.4f}, F1-Hold: {f1_hold:.4f}, F1-macro: {f1_macro:.4f}")
        print("  Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=[0,1,2]))

        accs.append(acc)
        f1_hold_list.append(f1_hold)
        f1_macro_list.append(f1_macro)

        # เคลียร์ memory
        from tensorflow.keras import backend as K
        import gc
        K.clear_session()
        gc.collect()

    print(f"\n[EA22 CV] Mean accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"[EA22 CV] Mean F1-Hold  : {np.mean(f1_hold_list):.4f} ± {np.std(f1_hold_list):.4f}")
    print(f"[EA22 CV] Mean F1-macro : {np.mean(f1_macro_list):.4f} ± {np.std(f1_macro_list):.4f}")

    return accs, f1_hold_list, f1_macro_list

def find_threshold_by_grid(
        df: pd.DataFrame,
        data_pca: np.ndarray,
        look_back: int,
        hold_bars: int,
        low: float = 0.0001,
        high: float = 0.005,
        step: float = 0.0001,
        target_hold_frac: tuple[float, float] = (0.10, 0.30)
    ) -> tuple[float, float]:
    prices = df['close'].values
    returns = pd.Series(prices).pct_change().dropna().values

    # ลูปหา thr symmetric ที่ Hold frac อยู่ในช่วง target_hold_frac
    for thr in np.arange(low, high + 1e-9, step):
        X_tmp, y_tmp = create_labels_from_price(
            data_pca,
            prices,
            look_back=look_back,
            hold_bars=hold_bars,
            thr_short=thr,
            thr_long=thr
        )
        hold_frac = np.mean(np.argmax(y_tmp, axis=1) == 1)
        if target_hold_frac[0] <= hold_frac <= target_hold_frac[1]:
            print(f"Found thr={thr:.6f} → Hold frac {hold_frac:.2%}")
            return thr, thr

    # ถ้ายังหาไม่เจอใน grid → fallback เป็น percentile
    p25, p75 = np.percentile(returns, [25, 75])
    thr_short, thr_long = abs(p25), abs(p75)
    if thr_short == 0 and thr_long == 0:
        base = np.std(returns)
    else:
        base = max(thr_short, thr_long)

    for factor in [1, 2, 5, 10]:
        tst = base * factor
        X_tmp, y_tmp = create_labels_from_price(
            data_pca,
            prices,
            look_back=look_back,
            hold_bars=hold_bars,
            thr_short=tst,
            thr_long=tst
        )
        hold_frac = np.mean(np.argmax(y_tmp, axis=1) == 1)
        if hold_frac > 0:
            print(f"Fallback std×{factor}: thr={tst:.6f} → Hold frac {hold_frac:.2%}")
            return tst, tst

    print("ยังไม่เจอ Hold เลย → return (0,0)")
    return 0.0, 0.0

def oversample_hold_only(
        X: np.ndarray,
        y: np.ndarray,
        hold_class: int = 1,
        target_frac: float = 0.12
    ) -> tuple[np.ndarray, np.ndarray]:
    labels = np.argmax(y, axis=1)
    counts = np.bincount(labels, minlength=y.shape[1])
    n_total = len(labels)
    n_hold  = counts[hold_class]
    n_desired_hold = int(target_frac * n_total)

    # ถ้า Hold มีเยอะจนเกิน target → รับคืน dataset เดิม
    if n_hold >= n_desired_hold:
        return X, y

    # กรณี Hold ยังน้อยกว่า target → oversample
    from imblearn.over_sampling import RandomOverSampler
    from tensorflow.keras.utils import to_categorical

    ros = RandomOverSampler(
        sampling_strategy={hold_class: n_desired_hold},
        random_state=42
    )
    X_flat = X.reshape(len(X), -1)
    X_res_flat, y_res_lbls = ros.fit_resample(X_flat, labels)
    X_res = X_res_flat.reshape(-1, X.shape[1], X.shape[2])
    y_res = to_categorical(y_res_lbls, num_classes=y.shape[1])
    return X_res, y_res

def get_class_weights(y):
    """
    y: one-hot encoded array shape=(N,3)
    คืน dict mapping class index → weight
    """
    # แปลงกลับเป็น labels 0/1/2
    labels = np.argmax(y, axis=1)
    # หาคลาสที่มีอยู่จริง
    present_classes = np.unique(labels)
    # คำนวณ class weight สำหรับคลาสที่มีจริง
    cw = compute_class_weight(
        class_weight='balanced',
        classes=present_classes,
        y=labels
    )
    # สร้าง dict เต็ม 0–2 โดยถ้าไม่มีคลาสใด ให้ weight=1.0
    class_weights = {i: 1.0 for i in range(y.shape[1])}
    for cls, w in zip(present_classes, cw):
        class_weights[int(cls)] = float(w)
    return class_weights

def weighted_cce_loss(weight_vector):
    """
    weight_vector: 1D array หรือ list ของน้ำหนักแต่ละคลาส [w0, w1, w2]
    คืนฟังก์ชัน loss ที่ scale cross‐entropy ด้วย weight per sample
    """
    # แปลงเป็น tensor ค้างไว้
    weights = tf.constant(weight_vector, dtype=tf.float32)
    def loss(y_true, y_pred):
        # ปกติ categorical_crossentropy จะคืน shape=(batch,)
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        # คำนวณ weight ต่อ sample ด้วย dot product ระหว่าง one-hot y_true กับ weights
        sample_weights = tf.reduce_sum(y_true * weights, axis=1)
        # scale loss
        return cce * sample_weights
    return loss

def sparse_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss สำหรับ sparse labels (y_true เป็น int) 
    คำนวณทั้งหมดใน float32 เพื่อหลีกเลี่ยง dtype mismatch
    """
    def loss_fn(y_true, y_pred):
        # 1) Cast y_pred → float32
        y_pred = tf.cast(y_pred, tf.float32)
        # 2) Flatten y_true → int32 vector
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # 3) Sparse Categorical Crossentropy (float32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # 4) สร้าง one-hot ของ y_true (float32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)

        # 5) ดึง p_t = ความน่าจะเป็นของคลาสจริง
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=1)

        # 6) คำนวณ focal weight
        weight = alpha * tf.pow(1.0 - p_t, gamma)

        # 7) คืน loss per-sample (float32)
        return weight * ce

    return loss_fn

class ConfusionMatrixCallback(Callback):
    def __init__(self, validation_data, labels=None, target_names=None):
        super().__init__()
        self.X_val, self.y_val = validation_data
        # ถ้าไม่ระบุ ให้เป็น 3 คลาสตามเดิม
        self.labels       = labels       if labels       is not None else [0,1,2]
        self.target_names = target_names if target_names is not None else ['Short','Hold','Long']

    def on_epoch_end(self, epoch, logs=None):
        # 1) predict แล้วเลือก label
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=-1)
        # 2) ถ้า y_val เป็น one-hot ให้แปลง
        if self.y_val.ndim > 1:
            y_true = np.argmax(self.y_val, axis=1)
        else:
            y_true = self.y_val
        # 3) คำนวณ CM และ report
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        print(f"\nEpoch {epoch+1} Confusion Matrix (labels={self.labels}):\n{cm}")
        print(classification_report(
            y_true, y_pred,
            labels=self.labels,
            target_names=self.target_names
        ))

def build_stage1_model(input_shape):
    """
    โมเดลสำหรับแยก Long (2) vs Not-Long (0+1)
    คืนค่าโมเดลที่ output เป็น 2 classes, ใช้ sparse_categorical_crossentropy
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_stage2_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(2, activation='softmax', kernel_regularizer=l2(1e-4))
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=sparse_focal_loss(gamma=2.0, alpha=0.25),  # <— ใช้ sparse version
        metrics=['accuracy']
    )
    return model

def find_threshold_by_bisect(
    data_pca: np.ndarray,
    prices: np.ndarray,
    look_back: int,
    hold_bars: int,
    desired_hold: float,
    tol: float = 0.005,
    max_iter: int = 20
) -> float:
    n = len(prices)
    max_i = n - look_back - hold_bars
    future_returns = np.empty(max_i, dtype=np.float32)
    for i in range(max_i):
        future_returns[i] = prices[i + look_back + hold_bars] / prices[i + look_back] - 1.0

    abs_ret = np.abs(future_returns)
    lo = 0.0
    hi = abs_ret.max() 

    thr = (lo + hi) / 2.0
    hold_frac = 0.0

    for _ in range(max_iter):
        thr = (lo + hi) / 2.0
        X_tmp_chk, y_tmp_chk = create_labels_from_price(
            data_pca, prices,
            look_back=look_back,
            hold_bars=hold_bars,
            thr_short=thr, thr_long=thr
        )
        labels_chk = np.argmax(y_tmp_chk, axis=1)
        hold_frac = np.mean(labels_chk == 1)

        if abs(hold_frac - desired_hold) <= tol:
            break
        # ถ้า hold_frac ที่ได้ > desired_hold → thr กว้างเกิน → ลด hi
        if hold_frac > desired_hold:
            hi = thr
        else:
            lo = thr

    return thr

def find_threshold_for_short(
    data_pca: np.ndarray,
    prices: np.ndarray,
    look_back: int,
    hold_bars: int,
    target_short_frac: float,
    low: float = 0.0001,
    high: float = 0.01,
    step: float = 0.0001
) -> tuple[float, None]:
    """
    หา threshold สำหรับ Short โดยให้ fraction ของ Short ≈ target_short_frac
    คืน (thr_short, None)
    """
    from sklearn.metrics import recall_score

    best_thr, best_diff = 0.0, float('inf')
    # สร้าง labels ชั่วคราวโดย varying thr_short
    for thr in np.arange(low, high+step, step):
        X_tmp, y_tmp = create_labels_from_price_nonzero(
            data_pca, prices,
            look_back=look_back, hold_bars=hold_bars,
            thr_short=thr, thr_long=high  # thr_long ใช้ค่าสูงมากเพื่อบังคับไม่จับ Long
        )
        labels = np.argmax(y_tmp, axis=1)
        short_frac = np.mean(labels == 0)
        diff = abs(short_frac - target_short_frac)
        if diff < best_diff:
            best_diff, best_thr = diff, thr
    return best_thr, None

def find_threshold_for_long(
    data_pca: np.ndarray,
    prices: np.ndarray,
    look_back: int,
    hold_bars: int,
    target_long_frac: float,
    low: float = 0.0001,
    high: float = 0.01,
    step: float = 0.0001
) -> tuple[None, float]:
    """
    หา threshold สำหรับ Long โดยให้ fraction ของ Long ≈ target_long_frac
    คืน (None, thr_long)
    """
    best_thr, best_diff = 0.0, float('inf')
    for thr in np.arange(low, high+step, step):
        X_tmp, y_tmp = create_labels_from_price_nonzero(
            data_pca, prices,
            look_back=look_back, hold_bars=hold_bars,
            thr_short=high, thr_long=thr  # thr_short ใช้ค่าสูงมากเพื่อบังคับไม่จับ Short
        )
        labels = np.argmax(y_tmp, axis=1)
        long_frac = np.mean(labels == 2)
        diff = abs(long_frac - target_long_frac)
        if diff < best_diff:
            best_diff, best_thr = diff, thr
    return None, best_thr

class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # ใส่ name ให้ add_weight แต่ละตัว
        self.tp = self.add_weight(
            name='tp',
            shape=(num_classes,),
            initializer='zeros'
        )
        self.fp = self.add_weight(
            name='fp',
            shape=(num_classes,),
            initializer='zeros'
        )
        self.fn = self.add_weight(
            name='fn',
            shape=(num_classes,),
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=1, output_type=tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        for i in range(self.num_classes):
            true_i = tf.equal(y_true, i)
            pred_i = tf.equal(y_pred_labels, i)
            tp = tf.reduce_sum(tf.cast(tf.logical_and(true_i, pred_i), self.dtype))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(~true_i, pred_i), self.dtype))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(true_i, ~pred_i), self.dtype))
            self.tp[i].assign_add(tp)
            self.fp[i].assign_add(fp)
            self.fn[i].assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall    = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = 2 * precision * recall / tf.math.maximum(precision + recall, 1e-8)
        return tf.reduce_mean(f1)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))

# ====================================
# 🔧 Configuration (วางที่นี่)
# ====================================
CONFIG = {
    "symbol": "EURUSDm",
    "file_paths": {
        "M5":  r"C:\\Users\\ACE\\EURUSDm_M5_7.csv",
        "M15": r"C:\\Users\\ACE\\EURUSDm_M15_6.csv",
        "H1":  r"C:\\Users\\ACE\\EURUSDm_H1_6.csv"
    },
    "parameters": {
        "look_back":        30,
        "max_hold_period":  18,
        "min_trades":       50,
        "batch_size":       64,
        "epochs_ea22":      100,
        "epochs_ea27":      30,
        "tcn_tune_iters":   5,
        "tft_tune_iters":   5
    }
}

# ─── Test consistency snippet ───
file_path = CONFIG["file_paths"]["M5"]
symbol    = CONFIG["symbol"]

df_raw  = load_csv(file_path)
df_prep = prepare_base_dataframe(file_path, symbol)

# ตรวจว่า index เป็น DatetimeIndex เหมือนกัน
assert isinstance(df_raw.index, pd.DatetimeIndex)
assert isinstance(df_prep.index, pd.DatetimeIndex)

# ตรวจว่าคอลัมน์หลักยังอยู่ครบ
for col in ['open','high','low','close']:
    assert col in df_raw.columns and col in df_prep.columns

print(f"✔ raw rows: {len(df_raw)} → prepared rows: {len(df_prep)}")

# ====================================
# 🎯 Section 6a: Run EA22
# ====================================
def run_ea22():
    symbol = CONFIG["symbol"]
    paths  = CONFIG["file_paths"]
    params = CONFIG["parameters"]

    # 1) Load & feature‐engineering
    data = load_csv_data(paths)
    df = prepare_base_dataframe(paths["M5"], symbol, dfs_other={'M15': data['M15'], 'H1': data['H1']})
    check_dataset_sufficiency(
        df,
        look_back  = params["look_back"],
        hold_bars  = params["max_hold_period"],
        min_trades = params["min_trades"]
    )

    # 0) Detect regime แล้วเก็บไว้ใน df
    df['regime'] = detect_market_regime(df['close'].values, n_states=3)    

    # 2) Prepare & dimensionality reduction
    features_ea22 = [c for c in df.columns if df[c].dtype != 'O']
    # เพิ่ม 'regime' เข้าไป
    features_ea22 += ['regime']
    features_ea22 += [f'vp_bin_{i+1}' for i in range(20)]
    features_ea22 += [f'imb_{i+1}' for i in range(10)]
    features_ea22 += ['total_bid', 'total_ask', 'spread']
    features_ea22 = list(OrderedDict.fromkeys(features_ea22))
    print(f"EA22 features after dedupe: {len(features_ea22)} items")

    #  ‒‒‒ หากมี NaN ให้ใช้ ffill/bfill แทน dropna() ‒‒‒
    X_raw_df = df[features_ea22].fillna(method='ffill').fillna(method='bfill')
    X_raw = X_raw_df.values
    print("X_raw.shape:", X_raw.shape)

    # 4) Scale เท่านั้น (ไม่ลดมิติ)
    scaler = RobustScaler().fit(X_raw)
    data_pca = scaler.transform(X_raw)   # ใช้ชื่อ data_pca ต่อเนื่องเพื่อไม่ต้องแก้ downstream
    print("Using full feature set → data_pca.shape =", data_pca.shape)

    # บันทึก feature names, scaler, pca
    os.makedirs("models/ea22", exist_ok=True)
    with open("models/ea22/feature_names_ea22.pkl", "wb") as f:
        pickle.dump(features_ea22, f)
    joblib.dump(scaler, "models/ea22/scaler22.pkl")

    prices = df['close'].values
    returns = pd.Series(prices).pct_change().dropna().values
    desired_hold = 0.20
    tol = 0.005

    max_thr = np.max(np.abs(returns))  # หรือช่วงสูงสุดที่ต้องการลอง

    # 1. หา thr_short (e.g. bisect/search เฉพาะ short)
    thr_short, _ = find_threshold_for_short(
        data_pca, prices,
        look_back=params["look_back"],
        hold_bars=params["max_hold_period"],
        target_short_frac=0.20  # ตั้ง value ให้ชัดเจน
    )
    # 2. หา thr_long
    _, thr_long = find_threshold_for_long(
        data_pca, prices,
        look_back=params["look_back"],
        hold_bars=params["max_hold_period"],
        target_long_frac=0.20
    )

    # สร้าง X_tmp, y_tmp ด้วย threshold ที่ได้
    X_tmp, y_tmp = create_labels_from_price_nonzero(
        data_pca, prices,
        look_back=params["look_back"],
        hold_bars=params["max_hold_period"],
        thr_short=thr_short,
        thr_long=thr_long
    )
    hold_frac_final = np.mean(np.argmax(y_tmp, axis=1) == 1)

    # สมมติหลังสร้าง X_tmp, y_tmp แล้ว:
    n_samples, seq_len, n_feats = X_tmp.shape
    labels_tmp = np.argmax(y_tmp, axis=1)
    X_flat = X_tmp.reshape(n_samples, seq_len * n_feats)
    y_labels = np.argmax(y_tmp, axis=1)
    hold_count = np.sum(labels_tmp == 1)
    X_sel_flat, top_idx = select_features_by_mutual_info(X_flat, y_labels, k=40)

    print(f"Final Hold frac = {hold_count/len(labels_tmp):.2%} (target={desired_hold:.2%})")
    print("Before oversample (all):", np.unique(labels_tmp, return_counts=True))

    # ——————————————————————————————————————————
    # 1) ถ้าไม่มี Hold เลยใน y_tmp → inject dummy Hold ก่อนแบ่ง train/val
    if hold_count == 0:
        # เราจะสร้าง dummy บางตัวโดย copy sequence แรกๆ ของ X_tmp
        # แล้วตั้ง label เป็น [0,1,0] (Short=0, Hold=1, Long=0)
        n_dummy = max(100, int(0.02 * len(X_tmp)))  
        # (อย่างน้อย 5 ชิ้น หรือ 1% ของข้อมูลทั้งหมด)  
        dummy_seq = np.repeat(X_tmp[:1], n_dummy, axis=0)
        dummy_lbl = np.tile([0,1,0], (n_dummy, 1))
        X_tmp = np.concatenate([X_tmp, dummy_seq], axis=0)
        y_tmp = np.concatenate([y_tmp, dummy_lbl], axis=0)

        labels_tmp = np.argmax(y_tmp, axis=1)
        hold_count = np.sum(labels_tmp == 1)
        print(f"Injected {n_dummy} dummy Hold → new Hold frac = {hold_count/len(labels_tmp):.2%}")
        print("After injecting dummy, support:", np.unique(labels_tmp, return_counts=True))

    # ——————————————————————————————————————————
    # 2) แบ่ง train/val
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp,
        test_size=0.2,
        shuffle=False,
        random_state=42
    )
    y_true = np.argmax(y_va, axis=1)
    labels_tr_before = np.argmax(y_tr, axis=1)
    if np.sum(labels_tr_before == 1) == 0:
        len_orig = len(labels_tmp) - n_dummy  # จำนวนก่อน inject
        dummy_indices = list(range(len_orig, len_orig + n_dummy))
        # ย้าย 1 dummy ทุกสิบตัว หรือจนกว่า X_tr จะมี Hold อย่างน้อย 1 ตัว
        moved = 0
        for idx in dummy_indices:
            # หา position ของ idx ใน X_tmp ที่ map ไป X_tr / X_va
            # อาศัย fact ว่าการ train_test_split เมื่อ shuffle=False จะเลือกแถวท้าย 20% เป็น val
            if idx < len(X_tmp) * 0.8:
                # idx อยู่ในส่วน train (80%)
                moved += 1
                break  # เจอ dummy ที่อยู่ใน train แล้ว
        # ถ้ายังไม่เจอ dummy ใน train ให้คัดลอก dummy_seq เพิ่มเข้า X_tr โดยตรง
        if moved == 0:
            extra_seq = dummy_seq[:1]
            extra_lbl = dummy_lbl[:1]
            X_tr = np.concatenate([X_tr, extra_seq], axis=0)
            y_tr = np.concatenate([y_tr, extra_lbl], axis=0)
            moved = 1

        print(f"Moved {moved} dummy Hold เข้า X_tr → Now train support:", 
              np.unique(np.argmax(y_tr, axis=1), return_counts=True))
        
    look_back = params["look_back"]
    n_features = X_tr.shape[2]
    input_shape = (look_back, n_features)

    def build_lstm_model(hp):
        units        = hp.Int('lstm_units',   min_value=32, max_value=128, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        lr           = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model = Sequential([
            Input(shape=input_shape),
            LSTM(units),
            Dropout(dropout_rate),
            Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    tuner = kt.RandomSearch(
        build_lstm_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='kt_logs',
        project_name='ea22_lstm'
    )

    earlystop = EarlyStopping(
        monitor='val_accuracy',    # หรือ 'val_loss' ตามต้องการ
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    tuner.search(
        X_tr, np.argmax(y_tr, axis=1),
        epochs=20,
        validation_data=(X_va, np.argmax(y_va, axis=1)),
        callbacks=[earlystop]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best HPS:", best_hps.values)

    def build_tuned_lstm(look_back: int, n_features: int):
        units        = best_hps.get('lstm_units')
        dropout_rate = best_hps.get('dropout_rate')
        lr           = best_hps.get('learning_rate')

        model = Sequential([
            Input(shape=(look_back, n_features)),
            LSTM(units),
            Dropout(dropout_rate),
            Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ——————————————————————————————————————————
    # 4) Oversample เฉพาะใน X_tr
    labels_tr_after = np.argmax(y_tr, axis=1)
    if 1 in labels_tr_after:
        X_tr, y_tr = oversample_hold_only(X_tr, y_tr, hold_class=1, target_frac=0.20)
        labels_tr = np.argmax(y_tr, axis=1)
        print("After oversample classes on train:", np.unique(labels_tr, return_counts=True))
    else:
        print("ยังไม่มี Hold ใน train แม้หลัง inject dummy → skip oversample_hold_only")

    tcn_params = tune_tcn(X_tr, y_tr, params["look_back"])
    tft_params = tune_tft(X_tr, y_tr, params["look_back"])

    # 10) สร้างฟังก์ชัน build_base_fns (นำค่าที่ tune ได้มาใช้)
    base_fns = [
        build_model_lstm_att_hp,
        build_tuned_lstm,
        build_model_cnn_lstm,
        lambda lb, nf, fp=tcn_params['filters'], kp=tcn_params['kernel']: 
            build_model_tcn(lb, nf, filters=int(fp), kernel_size=int(kp)),
        lambda lb, nf, hp=tft_params['head_size'], fd=tft_params['ff_dim']: 
            build_model_transformer(lb, nf, head_size=int(hp), ff_dim=int(fd))
    ]

    cv_scores = cross_validate_ea22(
        X_tr, y_tr,
        build_base_fns=base_fns,
        build_rf_fn=build_rf_model,
        build_meta_fn=build_meta_model,
        look_back=params["look_back"],
        batch_size=params["batch_size"],
        epochs=params["epochs_ea22"],
        n_splits=5
    )
    print(f"[EA22 CV] Mean accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # หลัง oversample_hold_only บน X_tr, y_tr
    labels_res = np.argmax(y_tr, axis=1)
    present = np.unique(labels_res)
    cw_vals = compute_class_weight(class_weight='balanced', classes=present, y=labels_res)
    class_weights_dict = {c: w for c,w in zip(present, cw_vals)}
    class_weights_dict[1] *= 1.0   # ขยับ Hold ขึ้นอีก 50%
    weight_vector = np.array([class_weights_dict[i] for i in [0,1,2]], dtype=np.float32)
    loss_fn = weighted_cce_loss(weight_vector)

    # 14) สร้าง deep_models 4 แบบ แล้ว compile ด้วย loss_fn + metrics per-class
    deep_models = []
    for build_fn in [
        build_model_lstm_att_hp,
        build_tuned_lstm,
        build_model_cnn_lstm,
        lambda lb, nf: build_model_tcn(lb, nf, filters=int(tcn_params['filters']), kernel_size=int(tcn_params['kernel'])),
        lambda lb, nf: build_model_transformer(lb, nf, head_size=int(tft_params['head_size']), ff_dim=int(tft_params['ff_dim']))
    ]:
        m = build_fn(params["look_back"], X_tr.shape[2])
        m.compile(
            optimizer=LossScaleOptimizer(Adam(learning_rate=1e-4)),
            loss=loss_fn,
            metrics=[
                'accuracy',
                MacroF1(num_classes=3),
                tf.keras.metrics.Precision(class_id=1, name='prec_hold'),
                tf.keras.metrics.Recall(class_id=1, name='rec_hold')
            ]
        )
        deep_models.append(m)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=2, min_lr=1e-6, verbose=1
    )

    # 15) สร้าง TensorBoard callback
    tb_cb = TensorBoard(log_dir="logs/ea22_per_class", update_freq='epoch')

    # 16) Two-Stage Training (Stage 1: Long vs Rest)
    y_lbl_tr = np.argmax(y_tr, axis=1)
    y_lbl_va = np.argmax(y_va, axis=1)
    y1_tr = (y_lbl_tr == 2).astype(int)
    y1_va = (y_lbl_va == 2).astype(int)
    
    present1 = np.unique(y1_tr)  # ดูว่ามีคลาส [0] หรือ [1] หรือ [0,1]
    if len(present1) < 2:
    # ถ้ามีแค่คลาสเดียว ก็ให้ default weight =1.0 ทั้งสองคลาส
        class_weight_stage1 = {0:1.5, 1:1.0}
    else:
        cw1_vals = compute_class_weight(
            class_weight='balanced',
            classes=present1,
            y=y1_tr
        )
        class_weight_stage1 = {int(c): float(w) for c, w in zip(present1, cw1_vals)}
    # เติม default weight=1.0 สำหรับคลาสที่หายไป (0,1)
        for c in [0, 1]:
            class_weight_stage1.setdefault(c, 1.0)

    # สร้างและฝึก Stage-1 Model
    m1 = build_stage1_model(input_shape=(params["look_back"], X_tr.shape[2]))
    m1.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    m1.fit(
        X_tr, y1_tr,
        validation_data=(X_va, y1_va),
        epochs=params["epochs_ea22"],
        batch_size=params["batch_size"],
        class_weight=class_weight_stage1,
        callbacks=[tb_cb, earlystop, reduce_lr, ConfusionMatrixCallback(validation_data=(X_va, y1_va))],
        verbose=1
    )

    # 2) เตรียมข้อมูลสำหรับ Stage 2: Short vs Hold บน “Rest”
    #    ก่อนอื่นหา indices ที่ Stage1 ทายเป็น Rest (==0)
    p1_tr = m1.predict(X_tr).argmax(axis=1)  # predictions บน train
    p1_va = m1.predict(X_va).argmax(axis=1)  # predictions บน val

    # แต่เพื่อฝึก Stage 2 ต้องใช้ “label จริง” (not predicted) แยกตาม y_lbl_tr != 2
    mask_tr = (y_lbl_tr != 2)  # ตำแหน่งที่แท้จริงเป็น Short(0) หรือ Hold(1)
    mask_va = (y_lbl_va != 2)

    # สร้าง X_tr2, y2_tr โดยใช้ label จริง y_lbl_tr
    X_tr2 = X_tr[mask_tr]
    rest_idx = np.where(y_lbl_tr != 2)[0]
    y2_tr = y_lbl_tr[mask_tr]  # ตอนนี้ y2_tr จะเป็น 0=Short หรือ 1=Hold

    X_va2 = X_va[mask_va]
    rest_va_idx = np.where(y_lbl_va != 2)[0]
    y2_va = y_lbl_va[mask_va]

    from imblearn.over_sampling import BorderlineSMOTE

    if np.sum(y2_tr == 1) < 100:
        sm = BorderlineSMOTE(sampling_strategy=0.3, random_state=42)
        # reshape → 2D ก่อน resample
        X2_flat, y2_res = sm.fit_resample(
            X_tr2.reshape(len(X_tr2), -1),
            y2_tr
        )
        
        X_tr2 = X2_flat.reshape(-1, LOOK_BACK, X_tr.shape[2])
        y2_tr = y2_res
        print("ใช้ BorderlineSMOTE Stage 2 → support Short vs Hold:", np.unique(y2_tr, return_counts=True))
    
    # กำหนด classes ชัดเจน: 0=Short, 1=Hold
    classes2 = np.array([0, 1])
    if len(np.unique(y2_tr)) < 2:
        # ถ้ามีแค่คลาสเดียว ให้ weight =1 ทั้งคู่
        class_weight_stage2 = {0:1.5, 1:1.0}
    else:
        # คำนวณ weight แล้วขยาย Hold ให้หนักขึ้นอีกเท่าตัว
        cw2_vals = compute_class_weight(
            class_weight='balanced',
            classes=classes2,
            y=y2_tr
        )
        class_weight_stage2 = {
            0: float(cw2_vals[0]),
            1: float(cw2_vals[1]) * 1.3
        }

    # สร้างและฝึก Stage-2 Model
    m2 = build_stage2_model(input_shape=(params["look_back"], X_tr.shape[2]))
    m2.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    m2.fit(
        X_tr2, y2_tr,
        validation_data=(X_va2, y2_va),
        epochs=params["epochs_ea22"],
        batch_size=params["batch_size"],
        class_weight=class_weight_stage2,
        callbacks=[tb_cb, earlystop, reduce_lr, ConfusionMatrixCallback(validation_data=(X_va2, y2_va), labels=[0,1], target_names=['Short','Hold'])],
        verbose=1
    )

    # 19) Train RF + Stacking Meta-Model
    rf22 = build_rf_model()
    rf22.fit(X_tr.reshape(len(X_tr), -1), y_lbl_tr)

    meta_X22 = get_base_predictions(deep_models, X_tr, rf_model=rf22)
    meta22 = build_meta_model(input_dim=meta_X22.shape[1])

    # แบ่ง train/val สำหรับ meta-model
    X_meta_tr, X_meta_val, y_meta_tr, y_meta_val = train_test_split(
        meta_X22,
        y_lbl_tr,         # label เดิม 0/1/2
        test_size=0.2,
        random_state=42,
        shuffle=False
    )

    # 6) คำนวณ class_weight สำหรับ Meta-Model
    labels_meta = np.unique(y_meta_tr)  # ค่าที่มีจริงใน sub-train (0/1/2 บางทีอาจครบทุกคลาส)
    cw_meta_vals = compute_class_weight(
        class_weight='balanced',
        classes=labels_meta,
        y=y_meta_tr
    )
    class_weight_meta = {int(c): float(w) for c, w in zip(labels_meta, cw_meta_vals)}
    # เติม default=1.0 ให้ครบ 0,1,2
    for c in [0, 1, 2]:
        class_weight_meta.setdefault(c, 1.0)

    # —— ข้อ 4: เพิ่มน้ำหนักให้คลาส “Long” (2) —— 
    long_boost = 1.5  # จะปรับเป็น 1.2, 2.0, ฯลฯ ตามต้องการ
    class_weight_meta[2] = class_weight_meta[2] * long_boost
    print("Adjusted class weights:", class_weight_meta)

    # 7) Compile Meta-Model
    meta22.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 8) ฝึก Meta-Model พร้อม class_weight และ EarlyStopping
    meta22.fit(
        X_meta_tr, y_meta_tr,
        validation_data=(X_meta_val, y_meta_val),
        epochs=params["epochs_ea22"],
        batch_size=params["batch_size"],
        class_weight=class_weight_meta,
        callbacks=[earlystop, reduce_lr, ConfusionMatrixCallback(validation_data=(X_meta_val, y_meta_val))],
        verbose=1)

     # ---- Calibration with IsotonicRegression ----
    # (a) ดึง P(Hold) บน meta‐train
    probs_tr = meta22.predict(X_meta_tr)    # shape=(n_tr,3)
    P_hold_tr = probs_tr[:,1]
    y_hold_tr = (y_meta_tr == 1).astype(int)
    # เตรียม true label สำหรับ Hold
    mask = ~np.isnan(P_hold_tr)
    
    uniq = np.unique(y_hold_tr[mask])
    if len(uniq) == 2:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(P_hold_tr[mask], y_hold_tr[mask])
        def calibrate(p): 
            return iso.predict(np.nan_to_num(p, nan=0.5))
    else:
    # ถ้า meta-train ไม่มีทั้ง 0 และ 1 สำหรับคลาส Hold ให้ข้าม calibration
        def calibrate(p): 
            return np.nan_to_num(p, nan=0.5)

    # (b) คำนวณ P(Hold) บน meta‐val แล้ว calibrate
    probs_val_meta   = meta22.predict(X_meta_val)  # อันนี้ยัง raw
    P_hold_cal_val = calibrate(probs_val_meta[:, 1])
    y_true_hold_val = (y_meta_val == 1).astype(int)

    # ---- 1) รันด้วย Fixed Threshold = 0.48 ----
    hold_thr_fixed = 0.48

    # กัน NaN ก่อนเลือก Short(0)/Long(2)
    p0 = np.nan_to_num(probs_val_meta[:, 0], nan=0.0)
    p2 = np.nan_to_num(probs_val_meta[:, 2], nan=0.0)
    other_pred_val = np.where(p0 >= p2, 0, 2)

    final_pred_fixed = np.where(
        P_hold_cal_val > hold_thr_fixed,
        1,                          # Hold
        other_pred_val)

    y_meta_val_labels = y_meta_val if y_meta_val.ndim == 1 else np.argmax(y_meta_val, axis=1)
    print(f"\n=== Fixed hold_thr = {hold_thr_fixed:.2f} ===")
    print(classification_report(
        y_meta_val_labels,
        final_pred_fixed,
        target_names=['Short','Hold','Long'],
        zero_division=0
    ))
    print("Confusion Matrix:\n",  confusion_matrix(y_meta_val_labels, final_pred_fixed, labels=[0,1,2])
    )

    # ---- หา optimal threshold ด้วย F1‐macro ----
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.30, 0.60, num=301):
        pred_is_hold = (P_hold_cal_val > thr).astype(int)
        if y_true_hold_val.sum() == 0:
            f1 = 0.0
        else:
            f1 = f1_score(y_true_hold_val, pred_is_hold, average='binary', pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    hold_thr = best_thr 
    print(f"Optimal hold_thr (binary-F1 for Hold): {best_thr:.3f}, F1 = {best_f1:.4f}")

    # — Stage 1 & 2 evaluate with confidence threshold —
    # 1) พยากรณ์ Stage1 (Long vs Rest)
    probs1     = m1.predict(X_va)          # shape=(n_val,2)
    long_probs = probs1[:, 1]              # P(Long)
    long_thr   = 0.6                       # ปรับตามผล CV
    long_mask  = long_probs > long_thr     # Boolean array ยาว n_val

    meta_X_va   = get_base_predictions(deep_models, X_va, rf_model=rf22, batch_size=params["batch_size"])
    prob_va    = meta22.predict(meta_X_va)
    P_hold_cal_va  = calibrate(prob_va[:, 1]) 

    final_pred = np.zeros(len(X_va), dtype=int)
    final_pred[long_mask] = 2
    hold_mask = (~long_mask) & (P_hold_cal_va > hold_thr)
    final_pred[hold_mask] = 1 
                                 
    # 6) ประเมินผล
    y_true = np.argmax(y_va, axis=1)
    print(classification_report(y_true, final_pred, target_names=['Short','Hold','Long']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, final_pred, labels=[0,1,2]))

    # คำนวณ calibration curve
    fraction_of_pos, mean_pred_value = calibration_curve(
        y_true_hold_val,
        P_hold_cal_val,          
        n_bins=10,
        strategy='uniform'
    )

    # วาดกราฟ
    plt.figure(figsize=(6,6))
    plt.plot(mean_pred_value, fraction_of_pos, "s-", label="Calibrated")
    plt.plot([0,1], [0,1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram — Hold Class")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.hist(P_hold_cal, bins=10, range=(0,1), alpha=0.3, label="Prob histogram")
    plt.legend()
    plt.show()

    # ---- สรุปผลที่ threshold ใหม่ ----
    final_pred = np.where(P_hold_cal > best_thr,
                          1,
                          np.argmax(probs_val, axis=1))
    print(classification_report(y_meta_val, final_pred, target_names=['Short','Hold','Long']))
    print("Confusion Matrix:\n", confusion_matrix(y_meta_val, final_pred))

    # 20) Save artifacts
    artifacts = {
        "ea22_lstm":       deep_models[0],
        "ea22_cnn_lstm":   deep_models[1],
        "ea22_tcn":        deep_models[2],
        "ea22_transformer": deep_models[3],
        "scaler22":        scaler,
        "rf22":            rf22,
        "meta22":          meta22
    }
    save_model_bundle(artifacts, folder="models/ea22")
    logger.info("✅ EA22 Single-TF Training Complete")

    # 21) Cleanup memory (ลบเฉพาะตัวแปรที่มีอยู่จริง)
    to_delete = [
        'df','X_raw','scaled','data_pca','X_tmp','y_tmp',
        'X_tr','X_va','y_tr','y_va','deep_models',
        'rf22','meta_X22','meta22'
    ]
    for name in to_delete:
        if name in locals():
            del locals()[name]

    K.clear_session()
    gc.collect()

# ====================================
# 🎯 Section 6b: Run EA27
# ====================================
def run_ea27():
    symbol = CONFIG["symbol"]
    paths  = CONFIG["file_paths"]
    params = CONFIG["parameters"]

    # 1) โหลดข้อมูล M5, M15, H1
    data = load_csv_data(paths)
    df_m5 = prepare_base_dataframe(paths["M5"], symbol, dfs_other={'M15': data['M15'], 'H1': data['H1']})
    df_m15 = prepare_base_dataframe(paths["M15"], symbol, dfs_other={})
    df_h1 = prepare_base_dataframe(paths["H1"], symbol, dfs_other={})
    # 2) Join multi-TF; ใช้ left join + ffill เพื่อไม่ลด row หลักลงเยอะ
    df27 = (
        df_m5
        .join(df_m15.add_suffix("_M15"), how="left")
        .fillna(method='ffill')
        .join(df_h1.add_suffix("_H1"), how="left")
        .fillna(method='ffill')
    )

    check_dataset_sufficiency(
        df27,
        look_back=params["look_back"],
        hold_bars=params["max_hold_period"],
        min_trades=params["min_trades"]
    )

    # 3) เตรียม features (ตัวเลขเท่านั้น) + orderbook
    features_ea27 = [c for c in df27.columns if df27[c].dtype != 'O']
    features_ea27 += [f"vp_bin_{i+1}" for i in range(20)]
    features_ea27 += [f"imb_{i+1}" for i in range(10)]
    features_ea27 += ["total_bid", "total_ask", "spread"]
    features_ea27 = list(OrderedDict.fromkeys(features_ea27))

    X_raw = df27[features_ea27].dropna().values
    scaler_27 = RobustScaler().fit(X_raw)
    scaled_27 = scaler_27.transform(X_raw)
    pca_27 = PCA(n_components=0.95).fit(scaled_27)
    data_pca_27 = pca_27.transform(scaled_27)

    os.makedirs("models/ea27", exist_ok=True)
    with open("models/ea27/feature_names_ea27.pkl", "wb") as f:
        pickle.dump(features_ea27, f)
    joblib.dump(scaler_27, "models/ea27/scaler27.pkl")
    joblib.dump(pca_27, "models/ea27/pca27.pkl")

    # 4) หา threshold จาก price-driven label
    thr_short27, thr_long27 = find_threshold_by_grid(
        df27, data_pca_27,
        look_back=params["look_back"],
        hold_bars=params["max_hold_period"],
        low=0.0001, high=0.005, step=0.00005,
        target_hold_frac=(0.005, 0.05)
    )
    with open("models/ea27/best_threshold.pkl", "wb") as f:
        pickle.dump((thr_short27, thr_long27), f)

    # 5) สร้าง dataset classification
    X_tmp, y_tmp = create_labels_from_price(
        df27,
        look_back=params["look_back"],
        hold_bars=params["max_hold_period"],
        thr_short=thr_short27,
        thr_long=thr_long27
    )
    print("Before oversample, classes:", np.unique(np.argmax(y_tmp, axis=1)))

    # 6) inject dummy hold ถ้าไม่มี
    if 1 not in np.unique(np.argmax(y_tmp, axis=1)):
        dummy_seq = np.repeat(X_tmp[:1], 20, axis=0)
        dummy_lbl = np.tile([0,1,0], (20,1))
        X_tmp = np.concatenate([X_tmp, dummy_seq], axis=0)
        y_tmp = np.concatenate([y_tmp, dummy_lbl], axis=0)

    # 7) oversample hold ~25%
    X_res, y_res = oversample_hold_only(X_tmp, y_tmp, hold_class=1, target_frac=0.25)
    print("Post-oversample classes:", np.unique(np.argmax(y_res, axis=1), return_counts=True))

    # สมมติหลังสร้าง X_tmp, y_tmp แล้ว:
    labels_tmp = np.argmax(y_tmp, axis=1)
    unique, counts = np.unique(labels_tmp, return_counts=True)
    print("Before oversample:", dict(zip(unique, counts)))

    X_res, y_res = oversample_hold_only(X_tmp, y_tmp, hold_class=1, target_frac=0.12)
    labels_res = np.argmax(y_res, axis=1)
    unique2, counts2 = np.unique(labels_res, return_counts=True)
    print("After oversample:", dict(zip(unique2, counts2)))

    # 8) CV ด้วย cross_validate_ea22
    base_fns27 = [
        build_model_lstm_att_hp,
        build_model_cnn_lstm,
        build_model_tcn,
        build_model_transformer
    ]
    cv27_scores = cross_validate_ea22(
        X_res, y_res,
        build_base_fns=base_fns27,
        build_rf_fn=build_rf_model,
        build_meta_fn=build_meta_model,
        look_back=params["look_back"],
        batch_size=params["batch_size"],
        epochs=params["epochs_ea27"],
        n_splits=5
    )
    print(f"[EA27 CV] Mean accuracy: {np.mean(cv27_scores):.4f} ± {np.std(cv27_scores):.4f}")

    # 9) train/val split สุดท้าย
    X_tr, X_va, y_tr, y_va = train_test_split(X_res, y_res, test_size=0.2, random_state=42, shuffle=False)

    # 10) คำนวณ class_weight แบบ multi-class
    labels_res = np.argmax(y_tr, axis=1)
    classes = np.array([0,1,2])
    cw_list = compute_class_weight('balanced', classes=classes, y=labels_res)
    class_weight_multi = {int(c): float(w) for c, w in zip(classes, cw_list)}

    loss_fn = weighted_cce_loss(cw_list.astype(np.float32))

    # 11) Tune TCN/TFT (ซ้ำกับ EA22)
    tcn_params = tune_tcn(X_res, y_res, params["look_back"])
    tft_params = tune_tft(X_res, y_res, params["look_back"])

    # 12) สร้าง deep_models 4 แบบ แล้ว compile
    deep_models = []
    for build_fn in [
        build_model_lstm_att_hp,
        build_model_cnn_lstm,
        lambda lb, nf: build_model_tcn(lb, nf, filters=int(tcn_params['filters']), kernel_size=int(tcn_params['kernel'])),
        lambda lb, nf: build_model_transformer(lb, nf, head_size=int(tft_params['head_size']), ff_dim=int(tft_params['ff_dim']))
    ]:
        m = build_fn(params["look_back"], X_tr.shape[2])
        m.compile(
            optimizer=LossScaleOptimizer(Adam(learning_rate=1e-4)),
            loss=loss_fn,
            metrics=['accuracy']
        )
        deep_models.append(m)

    tb27 = TensorBoard(log_dir="logs/ea27_per_class", update_freq='epoch')

    # 13) Two-Stage Training (Stage 1: Long vs Rest)
    lbl_tr27 = np.argmax(y_tr, axis=1)
    lbl_va27 = np.argmax(y_va, axis=1)
    y1_tr27 = (lbl_tr27 == 2).astype(int)
    y1_va27 = (lbl_va27 == 2).astype(int)
    present1_27 = np.unique(y1_tr27)
    if len(present1_27) < 2:
        class_weight_stage1_27 = {0:1.0, 1:1.0}
    else:
        cw1_27 = compute_class_weight('balanced', classes=present1_27, y=y1_tr27)
        class_weight_stage1_27 = {int(c): float(w) for c, w in zip(present1_27, cw1_27)}

    m1_27 = build_stage1_model(input_shape=(params["look_back"], X_tr.shape[2]))
    m1_27.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    m1_27.fit(
        X_tr, y1_tr27,
        validation_data=(X_va, y1_va27),
        epochs=params["epochs_ea27"],
        batch_size=params["batch_size"],
        class_weight=class_weight_stage1_27,
        callbacks=[tb27, EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # 14) Stage 2: Short vs Hold ปรับใช้ label จริง (ไม่ใช้ prediction จาก Stage 1)
    mask_tr27 = (lbl_tr27 != 2)
    X_tr2_27 = X_tr[mask_tr27]
    y2_tr27 = lbl_tr27[mask_tr27]
    mask_va27 = (lbl_va27 != 2)
    X_va2_27 = X_va[mask_va27]
    y2_va27 = lbl_va27[mask_va27]
    present2_27 = np.unique(y2_tr27)
    if len(present2_27) < 2:
        class_weight_stage2_27 = {0:1.0, 1:1.0}
    else:
        cw2_27 = compute_class_weight('balanced', classes=present2_27, y=y2_tr27)
        class_weight_stage2_27 = {int(c): float(w) for c, w in zip(present2_27, cw2_27)}
        for c in [0,1]:
            class_weight_stage2_27.setdefault(c, 1.0)

    m2_27 = build_stage2_model(input_shape=(params["look_back"], X_tr.shape[2]))
    m2_27.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    m2_27.fit(
        X_tr2_27, y2_tr27,
        validation_data=(X_va2_27, y2_va27),
        epochs=params["epochs_ea27"],
        batch_size=params["batch_size"],
        class_weight=class_weight_stage2_27,
        callbacks=[tb27, EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # 15) Evaluate Two-Stage
    p1_va27 = m1_27.predict(X_va).argmax(axis=1)
    final_va27 = np.zeros_like(p1_va27)
    final_va27[p1_va27 == 1] = 2
    idx_nl27 = np.where(p1_va27 == 0)[0]
    final_va27[idx_nl27] = m2_27.predict(X_va[idx_nl27]).argmax(axis=1)
    print(classification_report(lbl_va27, final_va27, target_names=['Short','Hold','Long']))
    print("Confusion Matrix:\n", confusion_matrix(lbl_va27, final_va27))

    # 16) RF + XGB + Meta-model
    rf27 = build_rf_model()
    rf27.fit(X_tr.reshape(len(X_tr), -1), lbl_tr27)

    # ตรวจ injection dummy หากไม่มีคลาส Hold ใน y_tr
    if 1 not in np.unique(lbl_tr27):
        print("Injecting dummy Hold sample for XGB")
        dummy_x = X_tr[:1]
        dummy_y = np.array([1])
        X_tr = np.concatenate([X_tr, dummy_x], axis=0)
        lbl_tr27 = np.concatenate([lbl_tr27, dummy_y], axis=0)

    xgb27 = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    xgb27.fit(X_tr.reshape(len(X_tr), -1), lbl_tr27)

    meta_X_tr27 = get_base_predictions(deep_models, X_tr, rf_model=rf27)
    print("Meta-feature shape:", meta_X_tr27.shape)

    Xm_tr27, Xm_va27, ym_tr27, ym_va27 = train_test_split(
        meta_X_tr27, lbl_tr27,
        test_size=0.2, random_state=42, shuffle=False
    )
    labels_meta27 = np.unique(ym_tr27)
    cw_meta_vals27 = compute_class_weight('balanced', classes=labels_meta27, y=ym_tr27)
    class_weight_meta27 = {int(c): float(w) for c, w in zip(labels_meta27, cw_meta_vals27)}
    for c in [0,1,2]:
        class_weight_meta27.setdefault(c, 1.0)

    meta27 = build_meta_model(input_dim=meta_X_tr27.shape[1])
    meta27.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    meta27.fit(
        Xm_tr27, ym_tr27,
        validation_data=(Xm_va27, ym_va27),
        epochs=params["epochs_ea27"],
        batch_size=params["batch_size"],
        class_weight=class_weight_meta27,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    # 17) Save artifacts
    artifacts = {
        "ea27_lstm":       deep_models[0],
        "ea27_cnn_lstm":   deep_models[1],
        "ea27_tcn":        deep_models[2],
        "ea27_transformer": deep_models[3],
        "scaler27":        scaler_27,
        "pca27":           pca_27,
        "rf27":            rf27,
        "xgb27":           xgb27,
        "meta27":          meta27
    }
    save_model_bundle(artifacts, folder="models/ea27")
    logger.info("✅ EA27 Ensemble Training Complete")

    # 18) Cleanup
    del df_m5, df_m15, df_h1, df27, X_raw, scaled_27, data_pca_27
    del X_res, y_res, X_tr, X_va, y_tr, y_va, deep_models, rf27, xgb27, meta_X_tr27, meta27
    K.clear_session()
    gc.collect()

# Paths & params (adjust as needed)
M5_DATA_PATH   = CONFIG["file_paths"]["M5"]
M15_DATA_PATH  = CONFIG["file_paths"]["M15"]
H1_DATA_PATH   = CONFIG["file_paths"]["H1"]
LOOK_BACK      = CONFIG["parameters"]["look_back"]
SYMBOL         = CONFIG["symbol"]

def evaluate_ea22():
    print("\n--- Evaluating EA22 ---")

    # 1) โหลดโมเดล + artifacts
    with custom_object_scope({'AttentionLayer': AttentionLayer, 'TCN': TCN}):
        meta22  = load_model("models/ea22/meta22.h5", compile=False,
                             custom_objects={'AttentionLayer': AttentionLayer, 'TCN': TCN})
        m_lstm  = load_model("models/ea22/ea22_lstm.h5",       compile=False)
        m_cnn   = load_model("models/ea22/ea22_cnn_lstm.h5",   compile=False)
        m_tcn   = load_model("models/ea22/ea22_tcn.h5",        compile=False)
        m_trans = load_model("models/ea22/ea22_transformer.h5", compile=False)

    rf22     = joblib.load("models/ea22/rf22.pkl")
    scaler22 = joblib.load("models/ea22/scaler22.pkl")

    # โหลดรายชื่อฟีเจอร์ที่บันทึกตอนฝึก
    with open("models/ea22/feature_names_ea22.pkl", "rb") as f:
        saved_feats = pickle.load(f)

    # 2) เตรียม DataFrame (M5) ด้วย pipeline เดียวกับตอน train
    df = load_csv(CONFIG["file_paths"]["M5"])  # index_col='time'
    df = calculate_indicators(df)
    df = calculate_support_resistance(df)
    df = add_additional_features(df)
    df = add_confirm_entry_feature(df)
    df = enrich_df_with_tick_orderbook(df, CONFIG["symbol"])

    # 3) เติมคอลัมน์ที่ขาดให้ครบตาม saved_feats
    for feat in saved_feats:
        if feat not in df.columns:
            df[feat] = 0.0

    # 4) สร้าง X_raw → scale (no PCA)
    X_raw_df = df[saved_feats] \
                    .fillna(method='ffill') \
                    .fillna(method='bfill') \
                    .fillna(0.0)
    X_raw    = X_raw_df.values
    X_scaled = scaler22.transform(X_raw)

    # 5) หา threshold บน X_scaled
    prices      = df['close'].values
    desired_hold = 0.20
    tol          = 0.005

    thr_final = find_threshold_by_bisect(
        X_scaled,                                   # ← ใช้ X_scaled แทน data_pca
        prices,
        look_back   = CONFIG["parameters"]["look_back"],
        hold_bars   = CONFIG["parameters"]["max_hold_period"],
        desired_hold= desired_hold,
        tol         = tol
    )
    thr_short, thr_long = thr_final, thr_final
    print(f"[Evaluate] ใช้ thr_final = {thr_final:.6f} เพื่อ Hold≈{desired_hold:.0%}")

    # 6) สร้าง dataset ด้วย create_labels_from_price
    X, y = create_labels_from_price(
        X_scaled,                                   # ← ใช้ X_scaled
        prices,
        look_back  = CONFIG["parameters"]["look_back"],
        hold_bars  = CONFIG["parameters"]["max_hold_period"],
        thr_short  = thr_short,
        thr_long   = thr_long
    )

    # 7) แยก validation set (20% ท้าย)
    n_seq  = len(X)
    n_test = int(n_seq * 0.2)
    X_train, X_val = X[:-n_test], X[-n_test:]
    y_train, y_val = y[:-n_test], y[-n_test:]

    labels_va = np.argmax(y_val, axis=1)
    print("Validation support (Short, Hold, Long):",
          np.unique(labels_va, return_counts=True))

    # 8) ทำ stacking & predict
    base_models = [m_lstm, m_cnn, m_tcn, m_trans]
    meta_inputs = get_base_predictions(base_models, X_val, rf_model=rf22)

    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(meta22.predict(meta_inputs), axis=1)

    print("EA22 Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(
        y_true, y_pred,
        labels=[0,1,2],
        target_names=['Short','Hold','Long'],
        zero_division=0
    ))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    return df, y_pred

def backtest_signals(prices: np.ndarray, signals: np.ndarray, hold_period: int) -> np.ndarray:
    returns = []
    for i, sig in enumerate(signals):
        if i + hold_period >= len(prices):
            break
        if sig == 2:       # long
            ret = prices[i + hold_period] / prices[i] - 1
        elif sig == 0:     # short
            ret = prices[i] / prices[i + hold_period] - 1
        else:              # hold
            ret = 0.0
        returns.append(ret)
    return np.array(returns)

# ───────────────────────────────────────────────────────
# Execute evaluation + backtest
# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) ฝึก EA22 ก่อน
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    try:
        run_ea22()
    finally:
        mt5.shutdown()

# ------------------------
# 4) View first few returns
# ------------------------
# result_df = pd.DataFrame({
#     'return': trade_returns,
#     'cum_return': equity_curve[1:]  # skip the initial zero
# })
# result_df.head()

def evaluate_ea27():
    print("\n=== EA27 Evaluation ===")

    with custom_object_scope({'AttentionLayer': AttentionLayer, 'TCN': TCN}):
        meta27 = load_model("models/ea27/meta27.h5", compile=False, custom_objects={'AttentionLayer': AttentionLayer, 'TCN': TCN})
        m_lstm27 = load_model("models/ea27/ea27_lstm.h5", compile=False)
        m_cnn27 = load_model("models/ea27/ea27_cnn_lstm.h5", compile=False)
        m_tcn27 = load_model("models/ea27/ea27_tcn.h5", compile=False)
        m_trans27 = load_model("models/ea27/ea27_transformer.h5", compile=False)

    rf27 = joblib.load("models/ea27/rf27.pkl")
    scaler27 = joblib.load("models/ea27/scaler27.pkl")
    pca27 = joblib.load("models/ea27/pca27.pkl")
    with open("models/ea27/feature_names_ea27.pkl", "rb") as f:
        features_ea27 = pickle.load(f)
    with open("models/ea27/best_threshold.pkl", "rb") as f:
        thr_short27, thr_long27 = pickle.load(f)

    # เตรียม DataFrame multi-TF
    data = load_csv_data(CONFIG["file_paths"])
    df_m5 = prepare_base_dataframe(CONFIG["file_paths"]["M5"], CONFIG["symbol"], dfs_other={'M15': data['M15'], 'H1': data['H1']})
    df_m15 = prepare_base_dataframe(CONFIG["file_paths"]["M15"], CONFIG["symbol"], dfs_other={})
    df_h1 = prepare_base_dataframe(CONFIG["file_paths"]["H1"], CONFIG["symbol"], dfs_other={})
    df27 = (
        df_m5
        .join(df_m15.add_suffix("_M15"), how="left").fillna(method='ffill')
        .join(df_h1.add_suffix("_H1"), how="left").fillna(method='ffill')
    )

    # Feature transform
    X_raw27 = df27[features_ea27].dropna().values
    X_scaled27 = scaler27.transform(X_raw27)
    X_pca27 = pca27.transform(X_scaled27)

    # สร้าง dataset hold-out
    X27, y27 = create_labels_from_price(
        df27,
        look_back=CONFIG["parameters"]["look_back"],
        hold_bars=CONFIG["parameters"]["max_hold_period"],
        thr_short=thr_short27,
        thr_long=thr_long27
    )
    _, X_val27, _, y_val27 = train_test_split(X27, y27, test_size=0.2, random_state=42, shuffle=False)
    print("Eval classes:", np.unique(np.argmax(y_val27, axis=1)))

    # stacking + predict
    base_models27 = [m_lstm27, m_cnn27, m_tcn27, m_trans27]
    meta_inputs27 = get_base_predictions(base_models27, X_val27, rf_model=rf27)

    y_true27 = np.argmax(y_val27, axis=1)
    y_pred27 = np.argmax(meta27.predict(meta_inputs27), axis=1)

    print("EA27 Accuracy:", accuracy_score(y_true27, y_pred27))
    print(classification_report(y_true27, y_pred27, labels=[0,1,2], target_names=['Short','Hold','Long'], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true27, y_pred27))
    return df27, y_pred27
    
# ====================================
# 🎯 Section 6c: Updated main()
# ====================================
def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    try:
        run_ea22()
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
