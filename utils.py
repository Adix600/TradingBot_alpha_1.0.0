from config import SPREAD_PCT, MAX_PATIENCE_BONUS


def simulate_trade(entry, exit_, pos_type, held=0, no_trade_steps=0, is_consolidating=False, price_history=None,
                   volume_history=None, cumulative_pnl=0):
    reward = 0  # Initialize reward before any additions
    ab_bc_ratio = None
    bc_cd_ratio = None
    macd_signal = False
    stoch_signal = False
    avg_volume = 1
    fib_38 = fib_61 = 1
    ab_bc_ratio = None
    bc_cd_ratio = None
    """
    Trade simulation with signal-based reward estimation.
    Implements multiple strategies:
    - Strategy 1: MACD + Stochastic
    - Strategy 2: Candle Formations (Engulfing, Harami, Star)
    - Strategy 3: Fibonacci retracements
    - Strategy 4: Harmonic patterns
    - Strategy 5: Volume confirmation
    """
    spread_value = entry * SPREAD_PCT
    pnl = (exit_ - entry - spread_value) if pos_type == 'long' else (entry - exit_ - spread_value)

    dx = 0  # default value for trend strength (ADX)

    # --- ADX Trend Strength Bonus (Improved) ---
    if price_history and len(price_history) >= 30:
        import pandas as pd
        import numpy as np

        highs = pd.Series(price_history[-30:]) * 1.001
        lows = pd.Series(price_history[-30:]) * 0.999
        closes = pd.Series(price_history[-30:])

        tr_list = []
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, len(highs)):
            high = highs.iloc[i]
            low = lows.iloc[i]
            prev_close = closes.iloc[i - 1]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)

            up_move = highs.iloc[i] - highs.iloc[i - 1]
            down_move = lows.iloc[i - 1] - lows.iloc[i]

            plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0)

        tr_smooth = pd.Series(tr_list).rolling(window=14).mean().iloc[-1]
        plus_di = 100 * pd.Series(plus_dm_list).rolling(window=14).mean().iloc[-1] / tr_smooth if tr_smooth else 0
        minus_di = 100 * pd.Series(minus_dm_list).rolling(window=14).mean().iloc[-1] / tr_smooth if tr_smooth else 0
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) else 0

        if dx >= 25:
            reward += min(dx / 100, 0.05)
            reward += min(dx / 100, 0.05)

    # Reward based on realized profit or loss
    reward += pnl

    # Reward for effective exit strategy
    if pnl > 0 and dx >= 25 and not is_consolidating:
        reward += 0.03  # closing profitable trade during strong trend
    elif pnl < 0 and dx < 15:
        reward += 0.02  # loss during weak trend, smaller penalty

    # Trend strength estimation (simple version)
    if price_history and len(price_history) >= 10:
        trend_strength = sum([(price_history[i] - price_history[i - 1]) for i in range(-5, 0)])
        if trend_strength > 0 and pos_type == 'long':
            reward += min(trend_strength * 5, 0.03)
        elif trend_strength < 0 and pos_type == 'short':
            reward += min(abs(trend_strength) * 5, 0.03)

    # Bonus for reaching higher cumulative profit
    if cumulative_pnl > 0:
        reward += cumulative_pnl * 0.05  # 5% of total profit as bonus

    # Example price action check (if price history provided)
    if price_history and len(price_history) >= 20:
        last = price_history[-1]
        prev1 = price_history[-2]
        prev2 = price_history[-3]
        prev3 = price_history[-4]
        prev4 = price_history[-5]

        # Detect support/resistance zones from recent prices
        support = min(price_history[-20:])
        resistance = max(price_history[-20:])

        near_support = abs(last - support) / support < 0.005
        near_resistance = abs(last - resistance) / resistance < 0.005

        # Simple trend
        if last > prev1 > prev2 and pos_type == 'long':
            reward += 0.01
        elif last < prev1 < prev2 and pos_type == 'short':
            reward += 0.01

        # Signal candle: bullish engulfing
        if prev1 < prev2 and last > prev1 and last > prev2 and pos_type == 'long':
            reward += 0.015  # bullish engulfing
            if near_support:
                reward += 0.02  # bonus at support

        # Signal candle: bearish engulfing
        if prev1 > prev2 and last < prev1 and last < prev2 and pos_type == 'short':
            reward += 0.015  # bearish engulfing
            if near_resistance:
                reward += 0.02  # bonus at resistance

        # Signal candle: pin bar (basic version)
        if abs(prev1 - prev2) < 0.001 and abs(last - prev1) > 0.01:
            reward += 0.01

        # Harami pattern
        if prev2 > prev1 < last and abs(prev1 - last) < abs(prev2 - prev1) and pos_type == 'long':
            reward += 0.012  # bullish harami
            if near_support:
                reward += 0.015
        if prev2 < prev1 > last and abs(prev1 - last) < abs(prev2 - prev1) and pos_type == 'short':
            reward += 0.012  # bearish harami
            if near_resistance:
                reward += 0.015

        # Doji
        if abs(last - prev1) < 0.001 and abs(prev2 - prev1) > 0.01:
            reward += 0.008  # doji signal

        # Morning star (simplified)
        if prev3 > prev2 and abs(prev2 - prev1) < 0.002 and last > prev1 > prev2 and pos_type == 'long':
            reward += 0.018  # morning star
            if near_support:
                reward += 0.02  # extra bonus

        # Evening star (simplified)
        if prev3 < prev2 and abs(prev2 - prev1) < 0.002 and last < prev1 < prev2 and pos_type == 'short':
            reward += 0.018  # evening star
            if near_resistance:
                reward += 0.02  # extra bonus
        if prev3 < prev2 and abs(prev2 - prev1) < 0.002 and last < prev1 < prev2 and pos_type == 'short':
            reward += 0.018  # evening star
            if abs(prev1 - prev2) / prev2 < 0.01:
                reward += 0.01  # bonus if on resistance
        if prev3 < prev2 and abs(prev2 - prev1) < 0.002 and last < prev1 < prev2 and pos_type == 'short':
            reward += 0.018  # evening star

    # Reward for recent MACD and Stochastic crossover at entry
    if price_history and len(price_history) >= 5:
        macd_now = price_history[-1] - price_history[-2]
        macd_prev = price_history[-2] - price_history[-3]

        stoch_now = price_history[-1] - min(price_history[-5:])
        stoch_range = max(price_history[-5:]) - min(price_history[-5:])
        stoch_value = (stoch_now / stoch_range) * 100 if stoch_range != 0 else 50

        macd_signal = (macd_prev < 0 and macd_now > 0 and pos_type == 'long') or (
                    macd_prev > 0 and macd_now < 0 and pos_type == 'short')
        stoch_signal = (stoch_value < 20 and pos_type == 'long') or (stoch_value > 80 and pos_type == 'short')

        if macd_signal and stoch_signal:
            reward += 0.03

    # Volume confirmation bonus
    if volume_history and len(volume_history) >= 3:
        avg_volume = sum(volume_history[-3:]) / 3
        if volume_history[-1] > 1.2 * avg_volume:
            reward += 0.015  # higher recent volume than average

    # Fibonacci retracement bonus (if price hits key retracement levels)
    if price_history and len(price_history) >= 50:
        recent_high = max(price_history[-50:])
        recent_low = min(price_history[-50:])
        fib_38 = recent_high - 0.382 * (recent_high - recent_low)
        fib_61 = recent_high - 0.618 * (recent_high - recent_low)

        if abs(price_history[-1] - fib_38) / fib_38 < 0.003:
            reward += 0.015  # bonus for proximity to 38.2%
        if abs(price_history[-1] - fib_61) / fib_61 < 0.003:
            reward += 0.02  # bonus for proximity to 61.8%

    # Harmonic pattern reward (simplified check)
    if price_history and len(price_history) >= 10:
        a = price_history[-10]
        b = price_history[-7]
        c = price_history[-5]
        d = price_history[-1]

        ab = b - a
        bc = c - b
        cd = d - c

        ab_bc_ratio = abs(bc / ab) if ab != 0 else None
        bc_cd_ratio = abs(cd / bc) if bc != 0 else None

        # Gartley: AB ~61.8%, CD ~78.6%
        if ab_bc_ratio is not None and bc_cd_ratio is not None and 0.6 < ab_bc_ratio < 0.7 and 0.75 < bc_cd_ratio < 0.85:
            reward += 0.025

        # Bat pattern: AB ~50%, CD ~88.6%
        if ab_bc_ratio is not None and bc_cd_ratio is not None and 0.45 < ab_bc_ratio < 0.55 and 0.85 < bc_cd_ratio < 0.92:
            reward += 0.025

        # Butterfly: AB ~78.6%, CD ~127%-161.8%
        if ab_bc_ratio is not None and bc_cd_ratio is not None and 0.75 < ab_bc_ratio < 0.80 and 1.25 < bc_cd_ratio < 1.65:
            reward += 0.03

    # Moving Average Crossover (Strategy 6)
    if price_history and len(price_history) >= 21:
        ma_short = sum(price_history[-5:]) / 5
        ma_long = sum(price_history[-20:]) / 20
        prev_ma_short = sum(price_history[-6:-1]) / 5
        prev_ma_long = sum(price_history[-21:-1]) / 20

        # Cross up (long signal)
        if prev_ma_short < prev_ma_long and ma_short > ma_long and pos_type == 'long':
            reward += 0.02
            if volume_history and len(volume_history) >= 3:
                avg_volume = sum(volume_history[-3:]) / 3
                if volume_history[-1] > 1.2 * avg_volume:
                    reward += 0.01

        # Cross down (short signal)
        if prev_ma_short > prev_ma_long and ma_short < ma_long and pos_type == 'short':
            reward += 0.02
            if volume_history and len(volume_history) >= 3:
                avg_volume = sum(volume_history[-3:]) / 3
                if volume_history[-1] > 1.2 * avg_volume:
                    reward += 0.01

    # --- Signal Quality Summary ---
    signal_score = 0

    if price_history and len(price_history) >= 5:
        if macd_signal and stoch_signal:
            signal_score += 2
    if volume_history and len(volume_history) >= 3 and volume_history[-1] > 1.2 * avg_volume:
        signal_score += 1
    if price_history and len(price_history) >= 50:
        if abs(price_history[-1] - fib_38) / fib_38 < 0.003:
            signal_score += 1
        if abs(price_history[-1] - fib_61) / fib_61 < 0.003:
            signal_score += 1
    if ab_bc_ratio is not None and bc_cd_ratio is not None:
        signal_score += 1

    reward += 0.01 * signal_score  # reward based on signal quality

    return reward, pnl


def should_close_position(entry, current, pos_type, held, atr=None):
    # ATR-based or time-based threshold
    if atr:
        threshold = atr * (1 + 0.1 * held)
    else:
        threshold = 0.002 + (0.0005 * held)

    change = (current - entry) / entry if pos_type == 'long' else (entry - current) / entry

    # Optional stop loss / take profit logic
    if change <= -0.03:
        return True  # stop loss at -3%
    if change >= 0.05:
        return True  # take profit at +5%

    return abs(change) >= threshold
