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
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import ClassifierMixin
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

# Custom exceptions
class DLLLoadError(RuntimeError):
    """Raised when the DLL cannot be loaded."""


class InsufficientDataError(RuntimeError):
    """Raised when the dataset is insufficient for training."""

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
        raise DLLLoadError(str(e)) from e

DLL_PATH = r"C:\Users\ACE\OneDrive - Khon Kaen University\Desktop\MyLibrary\x64\Debug\MyLibrary.dll"
try:
    DLL = load_dll(DLL_PATH)
except DLLLoadError as e:
    logger.error("DLL not loaded: %s", e)
    DLL = None

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
    # อาจรวม Garman-Klass, Parkinson, และ ATR-based volatility (ATR_vol) โดยไม่ทับคอลัมน์ ATR เดิม
    df['ATR_vol'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
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
        raise InsufficientDataError(f"insufficient data: {avail} < {required}")

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
    """
    preds = []
    # 1) พยากรณ์แต่ละโมเดลทีละ batch
    for idx, m in enumerate(models):
        if isinstance(m, ClassifierMixin):
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
def _best_val_acc(hist):
    for k in ('val_accuracy','val_sparse_categorical_accuracy','val_acc'):
        if k in hist: return float(np.max(hist[k]))
    raise KeyError("No validation accuracy key found")

def _to_label_1d(y):
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.reshape(-1) if y.shape[1] == 1 else np.argmax(y, axis=1)
    return y.astype(np.int32)

def tune_tcn(X, y, look_back, pbounds=None):
    if pbounds is None:
        pbounds = {'filters': (16,128), 'kernel': (2,8)}
    X = np.asarray(X, dtype=np.float32)
    y_lbl = _to_label_1d(y)
    tscv = TimeSeriesSplit(n_splits=3)

    def cv(filters, kernel):
        filters = int(filters); kernel = int(kernel)
        kernel = max(2, min(kernel, look_back))
        vals = []
        for tr, va in tscv.split(X):
            K.clear_session(); gc.collect()
            m = build_model_tcn(look_back, X.shape[2], filters=filters, kernel_size=kernel)
            # ต้อง compile ใน build_model_tcn ตามข้อ (1)
            es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            h = m.fit(X[tr], y_lbl[tr], validation_data=(X[va], y_lbl[va]),
                      epochs=10, batch_size=64, shuffle=False, callbacks=[es], verbose=0)
            vals.append(_best_val_acc(h.history))
        return float(np.mean(vals))

    opt = BayesianOptimization(f=cv, pbounds=pbounds, random_state=42)
    opt.maximize(init_points=5, n_iter=10)
    logger.info("✅ TCN tuning done: %s", opt.max)
    return opt.max['params']

def tune_tft(X, y, look_back, pbounds=None):
    if pbounds is None:
        pbounds = {'ff_dim': (64,256), 'head_size': (16,64), 'dropout': (0.0,0.5)}
    X = np.asarray(X, dtype=np.float32)
    y_lbl = _to_label_1d(y)
    tscv = TimeSeriesSplit(n_splits=3)

    def cv(ff_dim, head_size, dropout):
        ff_dim = int(ff_dim); head_size = int(head_size); dropout = float(dropout)
        vals = []
        for tr, va in tscv.split(X):
            K.clear_session(); gc.collect()
            m = build_model_tft(look_back, X.shape[2],
                                head_size=head_size, ff_dim=ff_dim,
                                dropout=dropout, n_blocks=3, num_heads=4)
            # ต้อง compile ใน build_model_tft ตามข้อ (1)
            es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            h = m.fit(X[tr], y_lbl[tr], validation_data=(X[va], y_lbl[va]),
                      epochs=10, batch_size=64, shuffle=False, callbacks=[es], verbose=0)
            vals.append(_best_val_acc(h.history))
        return float(np.mean(vals))

    opt = BayesianOptimization(f=cv, pbounds=pbounds, random_state=42)
    opt.maximize(init_points=3, n_iter=5)
    logger.info("✅ TFT tuning done: %s", opt.max)
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
import joblib, json, os
import gc
import keras_tuner as kt
import xgboost as xgb
import tensorflow as tf
from hmmlearn.hmm import GaussianHMM
from typing import Optional
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, fbeta_score
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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
    inner_splits: int = 3,
    embargo: int | None = None,
    long_boost: float = 1.0,
    hold_band: float = 0.15
):
    """
    - purged TSS + embargo สำหรับ inner OOF ภายในชุด train ของแต่ละ fold (กัน leakage)
    - meta เทรนบน OOF เท่านั้น
    - calibrate P(Hold) ด้วย Platt (โลจิสติก) บน OOF เท่านั้น
    - เลือก hold_thr โดยให้ coverage ใกล้สัดส่วนจริงใน fold (± hold_band)
    """
    if embargo is None:
        embargo = look_back  # กันไว้ขั้นต่ำ

    y_lbl = np.argmax(y, axis=1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs, f1_hold_list, f1_macro_list = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X, y_lbl), start=1):
        print(f"\n[EA22 CV] Fold {fold}/{n_splits} tr={len(tr_idx)} va={len(va_idx)}")

        # ---------- สร้าง slice ของ fold นี้ ----------
        X_tr, X_va  = X[tr_idx], X[va_idx]
        y_tr_lbl    = y_lbl[tr_idx]
        y_va_lbl    = y_lbl[va_idx]

        # ---------- (1) OOF ด้านในบน tr_idx ด้วย purged_tss ----------
        n_tr = len(X_tr)
        oof_holder = None
        filled = np.zeros(n_tr, dtype=bool)

        for in_fold, (in_tr_idx, in_va_idx) in enumerate(
            purged_tss(n_tr, n_splits=inner_splits, embargo=embargo), 1
        ):
            # base-in
            base_in = []
            for fn in build_base_fns:
                m = fn(look_back, X_tr.shape[2])
                m.fit(X_tr[in_tr_idx], y_tr_lbl[in_tr_idx],
                      epochs=epochs, batch_size=batch_size, verbose=0)
                base_in.append(m)

            # RF-in
            rf_in = build_rf_fn()
            rf_in.fit(X_tr[in_tr_idx].reshape(len(in_tr_idx), -1), y_tr_lbl[in_tr_idx])

            # meta features บน in_va_idx
            meta_va = get_base_predictions(base_in, X_tr[in_va_idx],
                                           rf_model=rf_in, batch_size=batch_size)

            if oof_holder is None:
                oof_holder = np.full((n_tr, meta_va.shape[1]), np.nan, dtype=np.float32)
            oof_holder[in_va_idx] = meta_va
            filled[in_va_idx] = True

        if not filled.all():
            dropped = int((~filled).sum())
            print(f"[CV][OOF] drop {dropped}/{n_tr} rows ({dropped/n_tr:.2%}) due to embargo/split edges.")

        # ใช้เฉพาะแถวที่มี OOF จริงมาเทรน meta
        mask_fit       = filled
        meta_X_tr_oof  = oof_holder[mask_fit]
        y_meta_tr_fit  = y_tr_lbl[mask_fit]

        row_ok = ~np.any(np.isnan(meta_X_tr_oof), axis=1)
        meta_X_tr_oof = meta_X_tr_oof[row_ok]
        y_meta_tr_fit = y_meta_tr_fit[row_ok]

        # ---------- (2) base_full & rf_full บน tr_idx ทั้งก้อน (ใช้ predict X_va) ----------
        base_full = []
        for fn in build_base_fns:
            m = fn(look_back, X_tr.shape[2])
            m.fit(X_tr, y_tr_lbl, epochs=epochs, batch_size=batch_size, verbose=0)
            base_full.append(m)

        rf_full = build_rf_fn()
        rf_full.fit(X_tr.reshape(len(X_tr), -1), y_tr_lbl)

        meta_X_va = get_base_predictions(base_full, X_va, rf_model=rf_full, batch_size=batch_size)

        # ---------- (3) ฝึก meta บน OOF เท่านั้น ----------
        classes_meta = np.unique(y_meta_tr_fit)
        cw_vals = compute_class_weight('balanced', classes=classes_meta, y=y_meta_tr_fit)
        class_weight_meta = {int(c): float(w) for c, w in zip(classes_meta, cw_vals)}
        for c in [0, 1, 2]:
            class_weight_meta.setdefault(c, 1.0)
        class_weight_meta[2] *= long_boost

        meta = build_meta_fn(input_dim=meta_X_tr_oof.shape[1])
        meta.compile(optimizer=Adam(1e-4),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        meta.fit(meta_X_tr_oof, y_meta_tr_fit,
                 epochs=epochs, batch_size=batch_size,
                 class_weight=class_weight_meta, verbose=0)

        # ---------- (4) Calibrate บน OOF (Platt) ----------
        probs_tr   = meta.predict(meta_X_tr_oof, verbose=0)
        P_hold_tr  = probs_tr[:, 1]
        y_hold_tr  = (y_meta_tr_fit == 1).astype(int)

        mask = ~np.isnan(P_hold_tr)
        P_fit = np.asarray(P_hold_tr[mask], dtype=np.float64)
        y_fit = np.asarray(y_hold_tr[mask], dtype=np.int32)

        cal_kind, cal_obj = "identity", None
        if np.unique(y_fit).size == 2 and (y_fit==0).sum()>=20 and (y_fit==1).sum()>=20:
        # guard: ถ้า variance ของ P_fit แทบเป็นศูนย์ ให้ข้าม calibration
            if np.std(P_fit) > 1e-6:
                iso = IsotonicRegression(out_of_bounds='clip').fit(P_fit, y_fit)
                lr  = LogisticRegression(solver='lbfgs', max_iter=1000).fit(P_fit.reshape(-1,1), y_fit)
                brier_iso = np.mean((iso.predict(P_fit)-y_fit)**2)
                brier_lr  = np.mean((lr.predict_proba(P_fit.reshape(-1,1))[:,1]-y_fit)**2)
                cal_kind, cal_obj = ("iso", iso) if brier_iso <= brier_lr else ("platt", lr)

        def calibrate(p):
            p = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.5)
            if cal_kind == "iso":
                return np.clip(cal_obj.predict(p), 0.0, 1.0)
            if cal_kind == "platt":
                return np.clip(cal_obj.predict_proba(p.reshape(-1,1))[:,1], 0.0, 1.0)
            return np.clip(p, 0.0, 1.0)

        # ---------- (5) Predict/threshold บน va_idx ----------
        probs_va = meta.predict(meta_X_va, verbose=0)
        p0 = np.nan_to_num(probs_va[:, 0], nan=0.0)
        p2 = np.nan_to_num(probs_va[:, 2], nan=0.0)
        other_pred_va = np.where(p0 >= p2, 0, 2)

        P_hold_cal = calibrate(probs_va[:, 1])

        # ============== เลือก hold_thr ให้คุม coverage + ดีต่อ F1(Hold) ==============
        y_true = y_va_lbl
        y_hold_bin = (y_true == 1).astype(int)

        target_hold = float((y_true == 1).mean())
        band = hold_band  # e.g. 0.15
        low_frac  = max(0.0, target_hold * (1 - band))
        high_frac = min(1.0, target_hold * (1 + band))

        P = np.asarray(P_hold_cal, dtype=np.float64)

        print(f"  DEBUG: low={low_frac:.2%} high={high_frac:.2%} "
              f"min={P.min():.3f} p25={np.percentile(P,25):.3f} "
              f"p50={np.percentile(P,50):.3f} p75={np.percentile(P,75):.3f} max={P.max():.3f}")

        n = len(P)
        # สร้างกริด threshold ที่ทำให้ coverage อยู่ในช่วง [low, high]
        q_low  = 1.0 - high_frac
        q_high = 1.0 - low_frac
        if q_low > q_high:
            q_low, q_high = q_high, q_low

        qs = np.linspace(q_low, q_high, 31)  # 31 candidates ในช่วงที่ coverage เข้าเงื่อนไข
        cand_thrs = np.unique(np.quantile(P, qs))

        best = None
        for thr in cand_thrs:
            cov = (P > thr).mean()
            # เผื่อกรณีค่าซ้ำเยอะ → ดัน threshold เล็กน้อยให้เข้าเป้า
            if cov < low_frac - 1e-6:
                # ลด threshold เพื่อเพิ่ม coverage
                thr = np.nextafter(thr, -np.inf)
                cov = (P > thr).mean()
            elif cov > high_frac + 1e-6:
                # เพิ่ม threshold เพื่อลด coverage
                thr = np.nextafter(thr, +np.inf)
                cov = (P > thr).mean()

            if low_frac - 1e-6 <= cov <= high_frac + 1e-6:
                pred_hold = (P > thr).astype(int)
                f1h = f1_score(y_hold_bin, pred_hold, zero_division=0)
                # จะใส่ penalty จาก |cov-target| ก็ได้ เช่น f1h - 0.05*abs(cov-target_hold)
                score = f1h
                if (best is None) or (score > best[0]):
                    best = (score, float(thr), float(cov))

        # fallback: ถ้าไม่มี candidate เข้า band (เช่น distribution แปลกมาก)
        if best is None:
            # เอาค่าใกล้ target ที่สุด
            thr_target = float(np.quantile(P, 1.0 - target_hold))
            # ดันเข้าขอบถ้าหลุด
            cov0 = (P > thr_target).mean()
            if cov0 < low_frac:
                hold_thr = float(np.quantile(P, 1.0 - low_frac))
            elif cov0 > high_frac:
                hold_thr = float(np.quantile(P, 1.0 - high_frac))
            else:
                hold_thr = thr_target
            cov = (P > hold_thr).mean()
        else:
            hold_thr, cov = best[1], best[2]

        # ============== ใช้ threshold ที่ได้ ==============
        y_pred = np.where(P > hold_thr, 1, other_pred_va).astype(int)

        acc      = accuracy_score(y_true, y_pred)
        f1_hold  = f1_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"  hold_thr={hold_thr:.3f} (target={target_hold:.2%}, "
              f"pred≈{cov:.2%}, band=[{low_frac:.2%},{high_frac:.2%}])"
              f" -> acc={acc:.4f}, F1(Hold)={f1_hold:.4f}, F1-macro={f1_macro:.4f}")
        print("  Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=[0,1,2]))

        accs.append(acc); f1_hold_list.append(f1_hold); f1_macro_list.append(f1_macro)

        # ---------- เคลียร์กราฟ/เมมโมรีต่อ fold ----------
        from tensorflow.keras import backend as K
        import gc; K.clear_session(); gc.collect()

    # ---------- สรุป ----------
    print(f"[EA22 CV] Mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
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
    weight_vector: ลิสต์/อาเรย์น้ำหนักต่อคลาส [w0, w1, w2]
    คืนค่า loss ที่เป็น CCE * weight ต่อ sample
    """
    # บังคับให้เป็น float32 ตั้งแต่ต้น (กัน float64 จาก numpy)
    w_f32 = tf.constant(weight_vector, dtype=tf.float32)

    def loss(y_true, y_pred):
        dtype = y_pred.dtype
        w   = tf.cast(w_f32, dtype)
        y_t = tf.cast(y_true, dtype)
        cce = tf.keras.losses.categorical_crossentropy(y_t, y_pred)
        samp_w = tf.reduce_sum(y_t * w, axis=1)
        return cce * samp_w
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
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']
    )
    return model

def build_stage2_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(2, activation='softmax', kernel_regularizer=l2(1e-4))
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),loss=sparse_focal_loss(gamma=2.0, alpha=0.25),metrics=['accuracy'])
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
    def __init__(self, num_classes=3, name="macro_f1", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        # เก็บสถิติเป็นเวกเตอร์ต่อคลาส แล้วค่อย assign_add ทีเดียว
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros", dtype=dtype)
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros", dtype=dtype)
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros", dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # รองรับทั้ง one-hot และ sparse
        if y_true.shape.rank == 2 and y_true.shape[-1] == self.num_classes:
            y_true_int = tf.argmax(y_true, axis=-1)
        else:
            y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        y_pred_int = tf.argmax(y_pred, axis=-1)

        # one-hot ทั้งสองอัน
        y_true_oh = tf.one_hot(y_true_int, depth=self.num_classes, dtype=self.dtype)
        y_pred_oh = tf.one_hot(y_pred_int, depth=self.num_classes, dtype=self.dtype)

        # คำนวณเวกเตอร์ tp/fp/fn ต่อคลาส
        tp = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)                  # (C,)
        fp = tf.reduce_sum((1.0 - y_true_oh) * y_pred_oh, axis=0)          # (C,)
        fn = tf.reduce_sum(y_true_oh * (1.0 - y_pred_oh), axis=0)          # (C,)

        # อัปเดตสถานะด้วย assign_add ทีละเวกเตอร์
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        eps = tf.constant(1e-7, dtype=self.dtype)
        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        f1_per_c  = 2.0 * precision * recall / (precision + recall + eps)
        return tf.reduce_mean(f1_per_c)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))

def purged_tss(n_samples, n_splits=5, embargo=0):
    """
    TimeSeriesSplit แบบ purge/embargo:
    - train อยู่ก่อน, val อยู่หลัง
    - ตัด train ส่วนปลายให้ห่าง val ต้นด้วยระยะ embargo
    คืนรายการ (train_idx, val_idx)
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tss.split(np.arange(n_samples)):
        cut = max(0, va_idx[0] - embargo)
        tr_idx = tr_idx[tr_idx < cut]
        yield tr_idx, va_idx

def build_and_fit_base(fn, X_tr, y_tr, X_va=None, y_va=None,
                       lr=1e-4, batch_size=64, epochs=10):
    m = fn(X_tr.shape[1], X_tr.shape[2])  # (look_back, n_features)
    # ฝึก base models ด้วย sparse (ส่ง label เป็น 1D)
    m.compile(optimizer=Adam(learning_rate=lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    if X_va is not None and y_va is not None:
        m.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0)
    else:
        m.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
              shuffle=False, verbose=0)
    return m

def _to_label_1d(y):
    y = np.asarray(y)
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=1)
    return y.astype(np.int32)

def make_oof_meta_features(
    X_tr, y_tr_onehot,
    build_base_fns,                # ลำดับต้อง “เหมือนตอนเทรนจริง”
    rf_builder,                    # ฟังก์ชันสร้าง RF เช่น build_rf_model
    look_back, batch_size, epochs,
    n_splits=4, embargo=None
):
    y_tr_lbl = _to_label_1d(y_tr_onehot)
    n = len(X_tr)

    if embargo is None:
        # ค่าปริยาย: กันอย่างน้อยเท่ากับ look_back (ถ้าอยากเคร่งครัด ใส่ look_back+hold_bars จากภายนอก)
        embargo = look_back

    oof_holder = None
    filled = np.zeros(n, dtype=bool)

    # ทำ K-fold แบบ purged
    folds = list(purged_tss(n, n_splits=n_splits, embargo=embargo))
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        # 1) ฝึก base แต่ละตัวบน tr_idx แล้วพยากรณ์ va_idx
        base_fold = []
        for fn in build_base_fns:
            m = build_and_fit_base(
                fn,
                X_tr[tr_idx], y_tr_lbl[tr_idx],
                X_tr[va_idx], y_tr_lbl[va_idx],
                batch_size=batch_size, epochs=epochs
            )
            base_fold.append(m)

        # 2) RF บน tr_idx
        rf_fold = rf_builder()
        rf_fold.fit(X_tr[tr_idx].reshape(len(tr_idx), -1), y_tr_lbl[tr_idx])

        # 3) สร้าง meta-features บน va_idx
        meta_va = get_base_predictions(
            base_fold, X_tr[va_idx],
            rf_model=rf_fold,
            batch_size=batch_size
        )
        if oof_holder is None:
            oof_holder = np.full((n, meta_va.shape[1]), np.nan, dtype=np.float32)
        oof_holder[va_idx] = meta_va
        filled[va_idx] = True

    # ใช้เฉพาะแถวที่มี OOF ครบ (ไม่เป็น NaN)
    if oof_holder is None:
        raise RuntimeError("OOF holder was never initialized; check your folds/indices.")

    valid_mask = filled & (~np.any(np.isnan(oof_holder), axis=1))
    n_valid = int(valid_mask.sum())
    if n_valid < n:
        miss = n - n_valid
        print(f"[OOF] Warning: drop {miss}/{n} rows ({miss/n:.1%}) w/out valid OOF due to embargo/split edges.")

    meta_X_tr = oof_holder[valid_mask]
    y_meta_tr = y_tr_lbl[valid_mask]

    # 4) ฝึก base_full + rf_full บน train ทั้งก้อน (ไว้ทำนาย holdout/val)
    base_full = []
    for fn in build_base_fns:
        m = build_and_fit_base(fn, X_tr, y_tr_lbl, None, None,
                               batch_size=batch_size, epochs=epochs)
        base_full.append(m)

    rf_full = rf_builder()
    rf_full.fit(X_tr.reshape(len(X_tr), -1), y_tr_lbl)

    return meta_X_tr, y_meta_tr, base_full, rf_full

def pick_gate(probs_gate, base_pred, y_true, long_mask, target_range=(0.15,0.30)):
    rest_mask = ~long_mask
    p = probs_gate[rest_mask]
    top = p.max(axis=1)
    second = np.partition(p, -2, axis=1)[:, -2]
    margin = top - second

    # กริดเริ่มค่อนข้างเข้ม และ "ไม่อนุญาต" margin=0
    conf_grid   = np.linspace(np.quantile(top, 0.65), np.quantile(top, 0.85), 6)
    margin_grid = np.linspace(0.04, 0.12, 5)   # <- ขั้นต่ำ 0.04

    best = None
    mid  = sum(target_range)/2
    for c in conf_grid:
        for m in margin_grid:
            mask_rest = (top >= c) & (margin >= m)
            eval_mask = long_mask.copy()
            eval_mask[rest_mask] = mask_rest
            cov = eval_mask.mean()
            if not (target_range[0] <= cov <= target_range[1]):
                continue
            y_hat = base_pred.copy()
            y_hat[rest_mask] = np.where(mask_rest, base_pred[rest_mask], 1).astype(int)
            f1m = f1_score(y_true, y_hat, average='macro', zero_division=0)
            score = (f1m, -abs(cov - mid))  # ดีสุด + ใกล้กลางช่วง
            if (best is None) or (score > best[0]): best = (score, c, m, eval_mask)

    if best is None:  # fallback นุ่มๆ
        c = float(np.quantile(top, 0.70)); m = 0.06
        mask_rest = (top >= c) & (margin >= m)
        eval_mask = long_mask.copy(); eval_mask[rest_mask] = mask_rest
        return c, m, eval_mask

    _, c, m, eval_mask = best
    return float(c), float(m), eval_mask

# ====================================
# 🔧 Configuration (วางที่นี่)
# ====================================
CONFIG = {
    "symbol": "EURUSDm",
    "file_paths": {
        "M5":  r"C:\\Users\\ACE\\EURUSDm_M5_8.csv",
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
        "tft_tune_iters":   5}}

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
    try:
        check_dataset_sufficiency(
            df,
            look_back=params["look_back"],
            hold_bars=params["max_hold_period"],
            min_trades=params["min_trades"]
        )
    except InsufficientDataError as e:
        logger.error("EA22 aborted: %s", e)
        return

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

    look_back = CONFIG["parameters"]["look_back"]
    hold_bars = CONFIG["parameters"]["max_hold_period"]

    #  ‒‒‒ หากมี NaN ให้ใช้ ffill/bfill แทน dropna() ‒‒‒
    X_raw_df = df[features_ea22].fillna(method='ffill').fillna(method='bfill')
    X_raw = X_raw_df.values
    print("X_raw.shape:", X_raw.shape)

    # 4) Scale เท่านั้น (ไม่ลดมิติ)
    split_i = int(len(X_raw) * 0.8)
    scaler = RobustScaler().fit(X_raw[:split_i])
    data_pca = scaler.transform(X_raw).astype(np.float32)   # ใช้ชื่อ data_pca ต่อเนื่องเพื่อไม่ต้องแก้ downstream
    print("Using full feature set → data_pca.shape =", data_pca.shape)

    # บันทึก feature names, scaler, pca
    os.makedirs("models/ea22", exist_ok=True)
    with open("models/ea22/feature_names_ea22.pkl", "wb") as f:
        pickle.dump(features_ea22, f)
    joblib.dump(scaler, "models/ea22/scaler22.pkl")

    prices = df['close'].values
    lb     = params["look_back"]
    hb     = params["max_hold_period"]
    target = 0.25        # เล็ง 25% (จะได้ 20–30%)

    # 1) สร้างกรอบค้นหา threshold จาก distribution ของ forward return
    fwd = prices[hb:] / prices[:-hb] - 1.0
    lo  = 0.0
    hi  = float(np.quantile(np.abs(fwd), 0.99))  # เพดานสมเหตุสมผล

    def make_labels(thr):
        X_tmp, y_tmp = create_labels_from_price_nonzero(
            data_pca, prices,
            look_back=lb, hold_bars=hb,
            thr_short=float(thr), thr_long=float(thr)
        )
        hold_frac = (np.argmax(y_tmp, axis=1) == 1).mean()
        return X_tmp, y_tmp, hold_frac

    # 2) binary search หา thr ที่ให้ Hold ใกล้ target
    best_thr = None
    for _ in range(18):  # พอใกล้เคียงในไม่กี่รอบ
        mid = (lo + hi) / 2
        _, y_mid, hold_mid = make_labels(mid)
        if hold_mid > target:
            # Hold มากไป -> thr ใหญ่ไป -> ลดลง
            hi = mid
        else:
            # Hold น้อยไป -> thr เล็กไป -> เพิ่มขึ้น
            lo = mid
        best_thr = mid

    # 3) ใช้ threshold ที่ได้จริงมาสร้างชุดสุดท้าย
    X_tmp, y_tmp, hold_frac_all = make_labels(best_thr)
    print(f"[LABEL] picked thr={best_thr:.6f} → Hold frac={hold_frac_all:.2%}")

    # 4) time-series cross-validation (ห้าม dummy/oversample)
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    for tr_idx, va_idx in tscv.split(X_tmp):
        X_tr, X_va = X_tmp[tr_idx], X_tmp[va_idx]
        y_tr, y_va = y_tmp[tr_idx], y_tmp[va_idx]
        print("[SPLIT] train:", np.unique(np.argmax(y_tr,axis=1), return_counts=True),
              "val:", np.unique(np.argmax(y_va,axis=1), return_counts=True))
        # model.fit(X_tr, y_tr)
        # fold_metrics.append(model.evaluate(X_va, y_va))
    # print("Mean metric across folds:", np.mean(fold_metrics))

    with open("models/ea22/labeling.json", "w") as f:
        json.dump({
            "look_back": params["look_back"],
            "hold_bars": params["max_hold_period"],
            "thr_short": float(best_thr),
            "thr_long":  float(best_thr)}, f)

    # ===== Sanity A: หลัง split =====
    def _to_lbl(y):
        y = np.asarray(y)
        return (np.argmax(y, 1) if y.ndim == 2 and y.shape[1] > 1 else y.reshape(-1)).astype(int)

    def _nan_inf_info(name, arr):
        a = np.asarray(arr)
        n = a.size
        nan = np.isnan(a).sum()
        inf = np.isinf(a).sum()
        print(f"  {name}: shape={a.shape}, NaN={nan} ({nan/n:.4%}), Inf={inf} ({inf/n:.4%})")

    look_back = CONFIG["parameters"]["look_back"]
    hold_bars = CONFIG["parameters"]["max_hold_period"]

    print("\n[SANITY A] Split/shape/NaN check")
    print(f"  look_back={look_back}, hold_bars={hold_bars}")
    print(f"  X_tr shape={X_tr.shape}, X_va shape={X_va.shape}")
    assert X_tr.ndim == 3 and X_tr.shape[1] == look_back, "X_tr shape ผิด (ต้องเป็น (n, look_back, n_features))"
    assert X_va.ndim == 3 and X_va.shape[1] == look_back, "X_va shape ผิด"

    _nan_inf_info("X_tr", X_tr)
    _nan_inf_info("X_va", X_va)

    y_tr_lbl = _to_lbl(y_tr)
    y_va_lbl = _to_lbl(y_va)

    def _dist(lbl, name):
        u, c = np.unique(lbl, return_counts=True)
        pct = {int(k): f"{v/len(lbl):.2%}" for k, v in zip(u, c)}
        print(f"  {name} class dist: {dict(zip(u, c))}  pct={pct}")

    _dist(y_tr_lbl, "train")
    _dist(y_va_lbl, "val")

    # ถือว่าไม่มี shuffle ต้องเรียงตามเวลา
    assert not np.any(np.isnan(X_tr)), "X_tr ยังมี NaN"
    assert not np.any(np.isnan(X_va)), "X_va ยังมี NaN"
        
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
    # [No-oversample] รายงาน distribution เฉยๆ แล้วไปใช้ class weights
    labels_tr = np.argmax(y_tr, axis=1)
    hold_frac_tr = (labels_tr == 1).mean()
    print(f"[SANITY train] Hold frac = {hold_frac_tr:.2%} (counts={dict(zip(*np.unique(labels_tr, return_counts=True)))})")
    print("[Resample] Disabled (Option A). Will use class weights only.")

    y_tr_lbl = np.argmax(y_tr, axis=1)
    tcn_params = tune_tcn(X_tr, y_tr_lbl, params["look_back"])
    tft_params = tune_tft(X_tr, y_tr_lbl, params["look_back"])

    # 10) สร้างฟังก์ชัน build_base_fns (นำค่าที่ tune ได้มาใช้)
    base_fns = [build_model_lstm_att_hp,build_tuned_lstm,build_model_cnn_lstm,
        lambda lb, nf, fp=tcn_params['filters'], kp=tcn_params['kernel']: 
            build_model_tcn(lb, nf, filters=int(fp), kernel_size=int(kp)),
        lambda lb, nf, hp=tft_params['head_size'], fd=tft_params['ff_dim']: 
            build_model_transformer(lb, nf, head_size=int(hp), ff_dim=int(fd))
    ]

    accs, f1h, f1m = cross_validate_ea22(
        X_tr, y_tr,
        build_base_fns=base_fns,
        build_rf_fn=build_rf_model,
        build_meta_fn=build_meta_model,
        look_back=params["look_back"],
        batch_size=params["batch_size"],
        epochs=params["epochs_ea22"],
        n_splits=4,                                     # 4–5 ได้ เลือกตามความยาวข้อมูล
        inner_splits=3,                                 # OOF ด้านใน
        embargo=params["look_back"] + params["max_hold_period"],
        long_boost=1.0,                                 # จะ boost long ค่อยปรับ >1
        hold_band=0.15                                  # coverage band ±15% รอบสัดส่วน Hold จริง
    )

    print(f"[EA22 CV] acc     : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"[EA22 CV] F1-Hold : {np.mean(f1h):.4f} ± {np.std(f1h):.4f}")
    print(f"[EA22 CV] F1-macro: {np.mean(f1m):.4f} ± {np.std(f1m):.4f}")

    # [No-oversample] ใช้ class weights จาก y_tr ปัจจุบัน
    labels_res = np.argmax(y_tr, axis=1)

    present = np.unique(labels_res)
    cw_vals = compute_class_weight(
        class_weight='balanced',
        classes=present,
        y=labels_res
    )
    class_weights_dict = {int(c): float(w) for c, w in zip(present, cw_vals)}
    # ถ้าบางคลาสไม่มีใน train ให้ default=1.0
    for c in [0, 1, 2]:
        class_weights_dict.setdefault(c, 1.0)

    # เรียงตาม [0,1,2]
    weight_vector = np.array([class_weights_dict[i] for i in [0, 1, 2]], dtype=np.float32)

    # loss แบบถ่วงน้ำหนัก (กัน dtype mismatch)
    def weighted_cce_loss(weight_vector):
        weights = tf.constant(weight_vector, dtype=tf.float32)
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            sample_weights = tf.reduce_sum(y_true * weights, axis=1)
            return cce * sample_weights
        return loss

    loss_fn = weighted_cce_loss(weight_vector)

    # (สำคัญ) ให้ label เป็น float32 เสมอ
    y_tr = y_tr.astype(np.float32)
    y_va = y_va.astype(np.float32)

    print("[ClassWeight] ", class_weights_dict)

    # 14) สร้าง deep_models 4 แบบ แล้ว compile ด้วย loss_fn + metrics per-class
    deep_models = []
    for build_fn in [build_model_lstm_att_hp,build_tuned_lstm,build_model_cnn_lstm,lambda lb, nf: build_model_tcn(lb, nf, filters=int(tcn_params['filters']), kernel_size=int(tcn_params['kernel'])),lambda lb, nf: build_model_transformer(lb, nf, head_size=int(tft_params['head_size']), ff_dim=int(tft_params['ff_dim']))]:
        m = build_fn(params["look_back"], X_tr.shape[2])
        m.compile(optimizer=LossScaleOptimizer(Adam(learning_rate=1e-4)),loss=loss_fn,
            metrics=[
                'accuracy',
                MacroF1(num_classes=3, dtype=tf.float32),
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

    for i, m in enumerate(deep_models):
        print(f"[EA22] Train base model #{i+1}/{len(deep_models)}")
        m.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=params["epochs_ea22"],
            batch_size=params["batch_size"],
            callbacks=[earlystop, reduce_lr],
            verbose=1
        )
    
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
        callbacks=[tb_cb, earlystop, reduce_lr,ConfusionMatrixCallback(validation_data=(X_va, y1_va),labels=[0,1],target_names=['Rest','Long'])],
        verbose=1
    )

    m1.save("models/ea22/m1_long_vs_rest.h5")
    m1.save("models/ea22/stage1_long_rest.h5")

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
        
        X_tr2 = X2_flat.reshape(-1, params["look_back"], X_tr.shape[2])
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
            1: float(cw2_vals[1]) * 1.3}

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

    m2.save("models/ea22/stage2_short_hold.h5")

    build_base_fns = [
        lambda lb, nf: build_model_lstm_att_hp(lb, nf),
        lambda lb, nf: build_tuned_lstm(lb, nf),
        lambda lb, nf: build_model_cnn_lstm(lb, nf),
        lambda lb, nf: build_model_tcn(lb, nf,
                                       filters=int(tcn_params['filters']),
                                       kernel_size=int(tcn_params['kernel'])),
        lambda lb, nf: build_model_transformer(lb, nf,
                                               head_size=int(tft_params['head_size']),
                                               ff_dim=int(tft_params['ff_dim']))
    ]

    # ❶ ทำ OOF meta-features บน train (กัน leakage)
    meta_X_tr, y_meta_tr, base_full, rf22 = make_oof_meta_features(
        X_tr, y_tr,
        build_base_fns=build_base_fns,
        rf_builder=build_rf_model,
        look_back=params["look_back"],
        batch_size=params["batch_size"],
        epochs=params["epochs_ea22"],
        n_splits=4,
        embargo=params["look_back"] + params["max_hold_period"]
    )

    # ❷ meta บน holdout/val ใช้ base_full ที่ฝึกบน train ทั้งก้อน
    meta_X_va = get_base_predictions(
        base_full, X_va,
        rf_model=rf22,
        batch_size=params["batch_size"]
    )

    # ===== Sanity B: OOF/meta =====
    print("\n[SANITY B] OOF/meta features")
    print(f"  meta_X_tr shape={meta_X_tr.shape}, meta_X_va shape={meta_X_va.shape}")

    mask_oof = ~np.any(np.isnan(meta_X_tr), axis=1)
    print(f"  OOF valid rows = {mask_oof.sum()}/{len(mask_oof)} ({mask_oof.mean():.2%})")
    if mask_oof.mean() < 0.75:
        print("  ⚠️ OOF coverage ต่ำ — ลองลด embargo หรือเพิ่มสัดส่วน train")

    # ดู performance ของ base models ทีละตัวบน val (ยังไม่รวม meta)
    print("  [Base models on VAL]")

    # ❸ เตรียม label แบบ 1D
    y_lbl_va = _to_lbl(y_va)

    for i, bm in enumerate(base_full, 1):
        p = bm.predict(X_va, verbose=0)
        pred = np.argmax(p, axis=1)
        acc = (pred == y_lbl_va).mean()
        hold_frac = (pred == 1).mean()
        print(f"    Base#{i}: acc={acc:.3f}, pred_hold={hold_frac:.2%}")

    meta_X_tr_fit = meta_X_tr[mask_oof]
    y_meta_tr_fit = y_meta_tr[mask_oof]

    # ❹ ฝึก meta22 บน OOF
    meta22 = build_meta_model(input_dim=meta_X_tr_fit.shape[1])

    labels_meta  = np.unique(y_meta_tr_fit)
    cw_meta_vals = compute_class_weight('balanced', classes=labels_meta, y=y_meta_tr_fit)
    class_weight_meta = {int(c): float(w) for c,w in zip(labels_meta, cw_meta_vals)}
    for c in [0,1,2]:
        class_weight_meta.setdefault(c, 1.0)
    # ลองไม่ boost long ก่อน
    print("Adjusted class weights (no long boost):", class_weight_meta)
 
    meta22.compile(optimizer=Adam(1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    meta22.fit(
        meta_X_tr_fit, y_meta_tr_fit,
        validation_data=(meta_X_va, y_lbl_va),
        epochs=params["epochs_ea22"],
        batch_size=params["batch_size"],
        class_weight=class_weight_meta,
        callbacks=[earlystop, reduce_lr,
                   ConfusionMatrixCallback(validation_data=(meta_X_va, y_lbl_va))],
        verbose=1
    )

    # ❺ Calibration/threshold (ใช้เฉพาะตำแหน่งที่ OOF valid)
    probs_tr  = meta22.predict(meta_X_tr_fit, verbose=0)   # OOF train probs
    P_hold_tr  = probs_tr[:, 1]
    y_hold_tr = (y_meta_tr_fit == 1).astype(int)

    mask_nan = ~np.isnan(P_hold_tr)
    P_fit = np.asarray(P_hold_tr[mask_nan], dtype=np.float64).reshape(-1, 1)
    y_fit = np.asarray(y_hold_tr[mask_nan], dtype=np.int32)

    # 2) ต้องมีทั้ง 0/1 และมีอย่างน้อย N ตัวอย่างต่อคลาส
    min_per_class = 10
    use_platt = False
    cal_lr = None
    if (np.unique(y_fit).size == 2 and
        (y_fit == 0).sum() >= min_per_class and
        (y_fit == 1).sum() >= min_per_class):
        cal_lr = LogisticRegression(
            solver='lbfgs', class_weight='balanced', max_iter=500, n_jobs=None
        )
        cal_lr.fit(P_fit, y_fit)
        use_platt = True

    def calibrate(p):
        p = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.5).reshape(-1, 1)
        if use_platt and (cal_lr is not None):
            return np.clip(cal_lr.predict_proba(p)[:, 1], 0.0, 1.0)
        return np.clip(p.ravel(), 0.0, 1.0)

    # ตัดสิน Hold ด้วย threshold; ที่ไม่ Hold เลือก 0/2 ตาม p0 vs p2
    probs_val_meta = meta22.predict(meta_X_va, verbose=0)
    p0 = np.nan_to_num(probs_val_meta[:, 0], nan=0.0)
    p1 = np.nan_to_num(probs_val_meta[:, 1], nan=0.5)
    p2 = np.nan_to_num(probs_val_meta[:, 2], nan=0.0)
    P_hold_cal_val = calibrate(p1)
    other_pred_val = np.where((p2 >= p0) & (P_long > long_thr), 2, 0)

    # (ออปชัน) ทดสอบ threshold คงที่
    hold_thr_fixed = 0.48
    final_pred_fixed = np.where(P_hold_cal_val > hold_thr_fixed, 1, other_pred_val)
    print(f"\n=== Fixed hold_thr = {hold_thr_fixed:.2f} ===")
    print(classification_report(y_lbl_va, final_pred_fixed,
                                target_names=['Short','Hold','Long'],
                                labels=[0,1,2], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_lbl_va, final_pred_fixed, labels=[0,1,2]))

    # ---- pick hold_thr to match target (with band) ----
    P = np.asarray(P_hold_cal_val, dtype=np.float64)
    P = np.nan_to_num(P, nan=0.5)

    y_hold_true = (y_lbl_va == 1).astype(int)
    target_hold = y_hold_true.mean()                  # สัดส่วน Hold จริงบน val

    # กำหนดช่วง coverage ที่ยอมรับได้รอบ target (เช่น ±15%)
    band = 0.15
    min_hold_frac = max(0.10, target_hold * (1 - band))
    max_hold_frac = min(0.60, target_hold * (1 + band))   # กันล้น ไม่เกิน 60%

    # ค่าเริ่มต้นด้วย quantile: อยากได้ frac ≈ target -> thr = quantile(1 - target)
    thr_q = float(np.quantile(P, max(0.0, min(1.0, 1.0 - target_hold))))

    # ไล่หาค่าที่ดีกว่าในกรอบรอบๆ thr_q
    grid = np.linspace(max(0.05, thr_q - 0.15), min(0.95, thr_q + 0.15), 201)

    best_score, best_thr = -1.0, thr_q
    best_frac, best_f1 = (P > best_thr).mean(), f1_score(y_hold_true, (P > best_thr).astype(int), zero_division=0)

    for thr in grid:
        pred_is_hold = (P > thr).astype(int)
        frac = pred_is_hold.mean()
        # ต้องอยู่ใน band ที่ยอมรับได้
        if not (min_hold_frac <= frac <= max_hold_frac):
            continue
        f1 = f1_score(y_hold_true, pred_is_hold, zero_division=0)
        score = f1 - 0.20 * abs(frac - target_hold)   # ลงโทษการเบี่ยงฐาน
        if score > best_score:
            best_score, best_thr, best_frac, best_f1 = score, float(thr), float(frac), float(f1)

    # ถ้าไม่มีค่าบน grid ผ่าน band เลย → fallback ใช้ thr_q (ใกล้ target สุด)
    hold_thr = float(best_thr)
    print(f"Picked hold_thr = {hold_thr:.3f} "
          f"(target={target_hold:.2%}, frac={best_frac:.2%}, F1={best_f1:.4f}, "
          f"band=[{min_hold_frac:.2%},{max_hold_frac:.2%}], start={thr_q:.3f})")

    # จูนด้วย F1(Long vs Rest) เฉพาะเมื่อมีทั้งบวก/ลบเพียงพอ
    def pick_long_threshold(P, y, p_min=0.70, beta=0.3, grid=None):
        if grid is None:
            grid = np.linspace(0.55, 0.90, 71)
        best = (-1.0, 0.60)
        for thr in grid:
            pred = (P > thr).astype(int)
            if precision_score(y, pred, zero_division=0) < p_min:
                continue
            f = fbeta_score(y, pred, beta=beta, zero_division=0)
            if f > best[0]:
                best = (f, float(thr))
        return best[1]

    # ---- กำหนด/จูน long_thr *ก่อน* เซฟไฟล์ ----
    probs1 = m1.predict(X_va, batch_size=params["batch_size"], verbose=0)  # shape=(n_val, 2)
    P_long = probs1[:, 1]
    y_true_long = (y_lbl_va == 2).astype(int)
    long_thr = pick_long_threshold(P_long, (y_lbl_va==2).astype(int),
                               p_min=0.55, beta=0.5, min_trades=30)
    print(f"Picked long_thr = {long_thr:.2f}") 

    # ===== Sanity C: Thresholds/gates =====
    print("\n[SANITY C] Thresholds & gates")
    print(f"  hold_thr = {hold_thr:.3f}, long_thr = {long_thr:.3f}")

    # สถิติความถี่หลัง calibrate
    print(f"[P(Hold) stats] min={P_hold_cal_val.min():.3f}, p25={np.percentile(P_hold_cal_val,25):.3f}, "
          f"p50={np.median(P_hold_cal_val):.3f}, p75={np.percentile(P_hold_cal_val,75):.3f}, max={P_hold_cal_val.max():.3f}")

    # อัตราการทริกเกอร์แต่ละ gate (ก่อนบังคับ Long)
    hold_flag = (P_hold_cal_val > hold_thr)
    print(f"  Hold flag (by hold_thr): {hold_flag.mean():.2%}")
    print(f"  Long flag (by long_thr): {(P_long > long_thr).mean():.2%}")

    # เซฟ calibrator/threshold
    os.makedirs("models/ea22", exist_ok=True)
    if use_platt and (cal_lr is not None):
        joblib.dump(cal_lr, "models/ea22/cal_hold_platt.pkl")
    with open("models/ea22/thresholds.json", "w") as f:
        json.dump({"hold_thr": float(hold_thr),
                   "long_thr": float(long_thr),
                   "calibrator": "platt" if use_platt else "identity"}, f)

    # --- เตรียมสำหรับ gate ---
    long_mask  = (P_long > long_thr)
    base_pred  = np.where(P_hold_cal_val > hold_thr, 1, other_pred_val).astype(int)
    probs_gate = np.stack([p0, P_hold_cal_val, p2], axis=1)

    # เลือกเกณฑ์ให้ coverage อยู่ในช่วง 15–30%
    CONF_MIN, MARGIN_MIN, eval_mask = pick_gate(probs_gate, long_mask, target_range=(0.15, 0.30))

    # สร้าง final_pred ตามเกณฑ์ที่เลือก
    final_pred = base_pred.copy()
    rest_mask  = ~long_mask
    if rest_mask.any():
        p_rest  = probs_gate[rest_mask]
        top     = p_rest.max(axis=1)
        second  = np.partition(p_rest, -2, axis=1)[:, -2]
        trade_rest = (top >= CONF_MIN) & ((top - second) >= MARGIN_MIN)
        final_pred[rest_mask] = np.where(trade_rest, final_pred[rest_mask], 1).astype(int)

    nonlong_cov = ((eval_mask & ~long_mask).mean() if eval_mask.any() else 0.0)
    print(f"[Gate] CONF_MIN={CONF_MIN:.3f}, MARGIN_MIN={MARGIN_MIN:.3f}, "
          f"coverage={eval_mask.mean():.2%} (long {long_mask.mean():.2%}, non-long {nonlong_cov:.2%})")

    # รายงานผลเฉพาะที่ “ยอมเทรด”
    if eval_mask.any():
        print(classification_report(
            y_lbl_va[eval_mask], final_pred[eval_mask],
            target_names=['Short','Hold','Long'], labels=[0,1,2], zero_division=0))
        print("Confusion Matrix (on-trade only):\n",
              confusion_matrix(y_lbl_va[eval_mask], final_pred[eval_mask], labels=[0,1,2]))
    else:
        print("⚠️ ไม่มีบาร์ผ่านเกณฑ์ — ผ่อน target_range หรือปรับกริดค้นหา")

    # — Calibration plots (ทั้งชุด val ไม่ต้องใช้ mask)
    y_true_hold_val = (y_lbl_va == 1).astype(int)
    assert len(y_true_hold_val) == len(P_hold_cal_val)
    frac_pos, mean_pred = calibration_curve(y_true_hold_val, P_hold_cal_val, n_bins=10, strategy='quantile')
    plt.figure(figsize=(5,5))
    plt.plot(mean_pred, frac_pos, "s-", label="Calibrated")
    plt.plot([0,1], [0,1], "k--", label="Perfect")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Reliability — Hold"); plt.grid(True); plt.legend(); plt.show()

    plt.figure(figsize=(5,3))
    plt.hist(P_hold_cal_val, bins=10, range=(0,1), alpha=0.3)
    plt.title("Distribution of calibrated P(Hold)")
    plt.xlabel("P(Hold)"); plt.ylabel("Count"); plt.grid(True); plt.show()

    # 20) Save artifacts
    artifacts = {"ea22_lstm": deep_models[0], "ea22_tuned_lstm": deep_models[1], "ea22_cnn_lstm": deep_models[2], "ea22_tcn": deep_models[3], "ea22_transformer": deep_models[4], "scaler22": scaler, "rf22": rf22, "meta22": meta22}
    save_model_bundle(artifacts, folder="models/ea22")
    logger.info("✅ EA22 Single-TF Training Complete")

    # 21) Cleanup memory (ลบเฉพาะตัวแปรที่มีอยู่จริง)
    to_delete = ['df','X_raw','scaled','data_pca','X_tmp','y_tmp','X_tr','X_va','y_tr','y_va','deep_models','rf22','meta_X22','meta22']
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
    df27 = (df_m5.join(df_m15.add_suffix("_M15"), how="left").fillna(method='ffill').join(df_h1.add_suffix("_H1"), how="left").fillna(method='ffill'))

    try:
        check_dataset_sufficiency(df27,look_back=params["look_back"],hold_bars=params["max_hold_period"],min_trades=params["min_trades"])
    except InsufficientDataError as e:
        logger.error("EA27 aborted: %s", e)
        return

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
    thr_short27, thr_long27 = find_threshold_by_grid(df27, data_pca_27,look_back=params["look_back"],hold_bars=params["max_hold_period"],low=0.0001, high=0.005, step=0.00005,target_hold_frac=(0.005, 0.05))
    with open("models/ea27/best_threshold.pkl", "wb") as f:
        pickle.dump((thr_short27, thr_long27), f)

    # 5) สร้าง dataset classification
    X_tmp, y_tmp = create_labels_from_price(df27,look_back=params["look_back"],hold_bars=params["max_hold_period"],thr_short=thr_short27,thr_long=thr_long27)
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
    base_fns27 = [build_model_lstm_att_hp,build_model_cnn_lstm,build_model_tcn,build_model_transformer]
    cv27_scores = cross_validate_ea22(X_res, y_res,build_base_fns=base_fns27,build_rf_fn=build_rf_model,build_meta_fn=build_meta_model,look_back=params["look_back"],batch_size=params["batch_size"],epochs=params["epochs_ea27"],n_splits=5)
    print(f"[EA27 CV] Mean accuracy: {np.mean(cv27_scores):.4f} ± {np.std(cv27_scores):.4f}")

    # 9) time-series cross-validation สุดท้าย
    tscv_final = TimeSeriesSplit(n_splits=5)
    final_scores = []
    for tr_idx, va_idx in tscv_final.split(X_res):
        X_tr, X_va = X_res[tr_idx], X_res[va_idx]
        y_tr, y_va = y_res[tr_idx], y_res[va_idx]
        final_scores.append(train_and_evaluate(X_tr, y_tr, X_va, y_va))
    print("Mean final score:", np.mean(final_scores))

    # 10) คำนวณ class_weight แบบ multi-class
    labels_res = np.argmax(y_res, axis=1)
    classes = np.array([0,1,2])
    cw_list = compute_class_weight('balanced', classes=classes, y=labels_res)
    class_weight_multi = {int(c): float(w) for c, w in zip(classes, cw_list)}

    loss_fn = weighted_cce_loss(cw_list.astype(np.float32))

    # 11) Tune TCN/TFT (ซ้ำกับ EA22)
    tcn_params = tune_tcn(X_res, y_res, params["look_back"])
    tft_params = tune_tft(X_res, y_res, params["look_back"])

    # 12) สร้าง deep_models 4 แบบ แล้ว compile
    deep_models = []
    for build_fn in [build_model_lstm_att_hp,build_model_cnn_lstm,lambda lb, nf: build_model_tcn(lb, nf, filters=int(tcn_params['filters']), kernel_size=int(tcn_params['kernel'])),lambda lb, nf: build_model_transformer(lb, nf, head_size=int(tft_params['head_size']), ff_dim=int(tft_params['ff_dim']))]:
        m = build_fn(params["look_back"], X_tr.shape[2])
        m.compile(optimizer=LossScaleOptimizer(Adam(learning_rate=1e-4)),loss=loss_fn,metrics=['accuracy'])
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
    m1_27.compile(optimizer=Adam(learning_rate=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    m1_27.fit(X_tr, y1_tr27,validation_data=(X_va, y1_va27),epochs=params["epochs_ea27"],batch_size=params["batch_size"],class_weight=class_weight_stage1_27,callbacks=[tb27, EarlyStopping(patience=5, restore_best_weights=True)],verbose=1)

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
    m2_27.compile(optimizer=Adam(learning_rate=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    m2_27.fit(X_tr2_27, y2_tr27,validation_data=(X_va2_27, y2_va27),epochs=params["epochs_ea27"],batch_size=params["batch_size"],class_weight=class_weight_stage2_27,callbacks=[tb27, EarlyStopping(patience=5, restore_best_weights=True)],verbose=1)

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

    xgb27 = xgb.XGBClassifier(objective='multi:softprob',num_class=3,use_label_encoder=False,eval_metric='mlogloss',random_state=42)
    xgb27.fit(X_tr.reshape(len(X_tr), -1), lbl_tr27)

    meta_X_tr27 = get_base_predictions(deep_models, X_tr, rf_model=rf27)
    print("Meta-feature shape:", meta_X_tr27.shape)

    tscv_meta = TimeSeriesSplit(n_splits=5)
    meta_scores = []
    for tr_idx, va_idx in tscv_meta.split(meta_X_tr27):
        Xm_tr27, Xm_va27 = meta_X_tr27[tr_idx], meta_X_tr27[va_idx]
        ym_tr27, ym_va27 = lbl_tr27[tr_idx], lbl_tr27[va_idx]
        meta_scores.append(train_and_evaluate_meta(Xm_tr27, ym_tr27, Xm_va27, ym_va27))
    print("Mean meta score:", np.mean(meta_scores))
    labels_meta27 = np.unique(lbl_tr27)
    cw_meta_vals27 = compute_class_weight('balanced', classes=labels_meta27, y=lbl_tr27)
    class_weight_meta27 = {int(c): float(w) for c, w in zip(labels_meta27, cw_meta_vals27)}
    for c in [0,1,2]:
        class_weight_meta27.setdefault(c, 1.0)

    meta27 = build_meta_model(input_dim=meta_X_tr27.shape[1])
    meta27.compile(optimizer=Adam(learning_rate=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    meta27.fit(Xm_tr27, ym_tr27,validation_data=(Xm_va27, ym_va27),epochs=params["epochs_ea27"],batch_size=params["batch_size"],class_weight=class_weight_meta27,callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],verbose=1
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
    print("\n--- Evaluating EA22 (calibrated) ---")

    with custom_object_scope({'AttentionLayer': AttentionLayer, 'TCN': TCN}):
        meta22  = load_model("models/ea22/meta22.h5", compile=False,
                             custom_objects={'AttentionLayer': AttentionLayer, 'TCN': TCN})
        # base models (เรียงลำดับเดียวกับตอนเทรน)
        m_lstm  = load_model("models/ea22/ea22_lstm.h5",       compile=False)
        m_tuned = load_model("models/ea22/ea22_tuned_lstm.h5", compile=False)
        m_cnn   = load_model("models/ea22/ea22_cnn_lstm.h5",   compile=False)
        m_tcn   = load_model("models/ea22/ea22_tcn.h5",        compile=False)
        m_trans = load_model("models/ea22/ea22_transformer.h5", compile=False)
        # NEW: stage-1/2
        m1      = load_model("models/ea22/stage1_long_rest.h5", compile=False)
        m2      = load_model("models/ea22/stage2_short_hold.h5", compile=False)

    rf22     = joblib.load("models/ea22/rf22.pkl")
    scaler22 = joblib.load("models/ea22/scaler22.pkl")
    
    try:
        iso = joblib.load("models/ea22/iso_hold.pkl")
        fitted_iso = True
    except Exception:
        iso = None
        fitted_iso = False

    def calibrate(p):
        p = np.nan_to_num(p, nan=0.5)
        return iso.predict(p) if fitted_iso else p

    with open("models/ea22/feature_names_ea22.pkl","rb") as f:
        saved_feats = pickle.load(f)
    with open("models/ea22/thresholds.json","r") as f:
        thr_conf = json.load(f)
    hold_thr = float(thr_conf.get("hold_thr", 0.50))
    long_thr = float(thr_conf.get("long_thr", 0.60))

    # --- build features + sequences (เหมือนเทรน) ---
    df = load_csv(CONFIG["file_paths"]["M5"])
    df = calculate_indicators(df)
    df = calculate_support_resistance(df)
    df = add_additional_features(df)
    df = add_confirm_entry_feature(df)
    df = enrich_df_with_tick_orderbook(df, CONFIG["symbol"])

    for c in saved_feats:
        if c not in df.columns: df[c] = 0.0
    X_raw = (df[saved_feats].ffill().bfill().fillna(0.0)).values.astype(np.float32)
    X_scaled = scaler22.transform(X_raw).astype(np.float32)

    prices = df['close'].values
    X_seq, y_onehot = create_labels_from_price(
        X_scaled, prices,
        look_back=CONFIG["parameters"]["look_back"],
        hold_bars=CONFIG["parameters"]["max_hold_period"],
        thr_short=None, thr_long=None  # ถ้าอยากคำนวณ metric จากราคา ให้ใส่ threshold เดิมของคุณ
    )
    y_true = np.argmax(y_onehot, axis=1)

    # --- Stage-1: Long vs Rest ---
    p1 = m1.predict(X_seq, verbose=0)[:, 1]      # P(Long)
    long_mask = p1 > long_thr

    # --- Meta features & calibrated P(Hold) ---
    base_models = [m_lstm, m_tuned, m_cnn, m_tcn, m_trans]
    meta_X = get_base_predictions(base_models, X_seq, rf_model=rf22,
                                  batch_size=CONFIG["parameters"]["batch_size"])
    # sanity: มิติต้องเท่ากับ meta22.input_shape[-1]
    assert meta_X.shape[1] == meta22.input_shape[-1]

    prob_meta   = meta22.predict(meta_X, verbose=0)    # (n,3)
    P_hold_cal  = iso.predict(np.nan_to_num(prob_meta[:,1], nan=0.5))

    p_long = m1.predict(X_seq, verbose=0)[:,1]
    long_mask = p_long > float(thr_conf.get("long_thr", 0.60))

    rest_idx   = np.where(~long_mask)[0]
    # --- Stage-2: Short vs Hold เฉพาะที่เหลือจาก Rest ---
    final_pred = np.zeros(len(X_seq), dtype=int)       # default=Short
    final_pred[long_mask] = 2

    if len(rest_idx) > 0:
        # วิธี A (ตรงกับแนวคิดตอนเทรน): ใช้ m2 เป็นตัวตัดสินฐาน แล้วใช้ calibrator เป็นเกณฑ์
        p_hold_m2 = m2.predict(X_seq[rest_idx], verbose=0)[:,1]
        p_hold = 0.5*P_hold_cal[rest_idx] + 0.5*p_hold_m2
        final_pred[rest_idx] = (p_hold > hold_thr).astype(int)

    # --- Metrics/plots ---
    print("EA22 Accuracy:", accuracy_score(y_true, final_pred))
    print(classification_report(y_true, final_pred, labels=[0,1,2],
                                target_names=['Short','Hold','Long'], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, final_pred, labels=[0,1,2]))

    frac_pos, mean_pred = calibration_curve((y_true==1).astype(int), P_hold_cal, n_bins=10)
    plt.figure(figsize=(5,5)); plt.plot(mean_pred, frac_pos, "s-", label="Calibrated")
    plt.plot([0,1],[0,1],"k--", label="Perfect"); plt.grid(True); plt.legend(); plt.title("Reliability — Hold"); plt.show()

    return df, final_pred

def backtest_signals(prices: np.ndarray, signals: np.ndarray, hold_period: int, cost_bps: float = 0.0) -> np.ndarray:
    returns = []
    c = cost_bps / 10000.0
    for i, sig in enumerate(signals):
        if i + hold_period >= len(prices):
            break
        if sig == 2:       # long
            ret = prices[i + hold_period] / prices[i] - 1.0
            ret -= c
        elif sig == 0:     # short
            ret = prices[i] / prices[i + hold_period] - 1.0
            ret -= c
        else:              # hold
            ret = 0.0
        returns.append(ret)
    return np.array(returns, dtype=np.float64)

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
    except (DLLLoadError, InsufficientDataError) as e:
        logger.error("EA22 failed: %s", e)
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
    df27 = (df_m5.join(df_m15.add_suffix("_M15"), how="left").fillna(method='ffill').join(df_h1.add_suffix("_H1"), how="left").fillna(method='ffill'))

    # Feature transform
    X_raw27 = df27[features_ea27].dropna().values
    X_scaled27 = scaler27.transform(X_raw27)
    X_pca27 = pca27.transform(X_scaled27)

    # สร้าง dataset hold-out
    X27, y27 = create_labels_from_price(df27,look_back=CONFIG["parameters"]["look_back"],hold_bars=CONFIG["parameters"]["max_hold_period"],thr_short=thr_short27,thr_long=thr_long27)
    tscv_eval = TimeSeriesSplit(n_splits=5)
    eval_acc = []
    for _, val_idx in tscv_eval.split(X27):
        X_val27, y_val27 = X27[val_idx], y27[val_idx]
        print("Eval classes:", np.unique(np.argmax(y_val27, axis=1)))

        # stacking + predict
        base_models27 = [m_lstm27, m_cnn27, m_tcn27, m_trans27]
        meta_inputs27 = get_base_predictions(base_models27, X_val27, rf_model=rf27)

        y_true27 = np.argmax(y_val27, axis=1)
        y_pred27 = np.argmax(meta27.predict(meta_inputs27), axis=1)

        eval_acc.append(accuracy_score(y_true27, y_pred27))
        print(classification_report(y_true27, y_pred27, labels=[0,1,2], target_names=['Short','Hold','Long'], zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_true27, y_pred27))
    print("EA27 Accuracy:", np.mean(eval_acc))
    return df27, y_pred27
    
# ====================================
# 🎯 Section 6c: Updated main()
# ====================================

if __name__ == "__main__":
