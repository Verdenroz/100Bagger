import pandas as pd


class TradingStrategy:
    """Base class for trading strategies with improved performance tracking"""

    def __init__(self, name):
        self.name = name
        self.initial_capital = 10000  # Default value

    def generate_signals(self, df):
        """Generate buy/sell signals - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_performance(self, df, initial_capital=10000):
        """Calculate performance metrics for the strategy with enhanced tracking"""
        self.initial_capital = initial_capital
        buy_signal, sell_signal = self.generate_signals(df)

        # Create performance DataFrame
        performance = pd.DataFrame(index=df.index)

        # Track signals and positions
        performance['buy_signal'] = buy_signal
        performance['sell_signal'] = sell_signal
        performance['position'] = 0

        # Calculate positions based on signals (state machine approach)
        current_position = 0
        entry_price = 0
        entry_date = None
        trades = []

        for i, date in enumerate(performance.index):
            # Update position based on signals
            if buy_signal.iloc[i] and current_position == 0:
                # Buy signal when not in position
                current_position = 1
                entry_price = df['close'].iloc[i]
                entry_date = date
            elif sell_signal.iloc[i] and current_position == 1:
                # Sell signal when in position
                current_position = 0
                exit_price = df['close'].iloc[i]

                # Record trade details
                profit_pct = (exit_price / entry_price - 1) * 100
                trade_length = (date - entry_date).days
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'days_held': trade_length
                })

                # Reset entry tracking
                entry_price = 0
                entry_date = None

            # Store current position
            performance.loc[date, 'position'] = current_position

        # Calculate returns
        performance['close'] = df['close']
        performance['stock_returns'] = df['close'].pct_change()

        # Strategy returns (only earn returns when holding a position)
        performance['strategy_returns'] = performance['position'].shift(1) * performance['stock_returns']
        performance['strategy_returns'] = performance['strategy_returns'].fillna(0)

        # Calculate cumulative returns
        performance['cumulative_stock_returns'] = (1 + performance['stock_returns']).cumprod()
        performance['cumulative_strategy_returns'] = (1 + performance['strategy_returns']).cumprod()

        # Calculate portfolio values
        performance['stock_portfolio'] = initial_capital * performance['cumulative_stock_returns']
        performance['strategy_portfolio'] = initial_capital * performance['cumulative_strategy_returns']

        # Calculate key metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['profit_pct'] > 0)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate average metrics if we have trades
        avg_profit = sum(trade['profit_pct'] for trade in trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(trade['profit_pct'] for trade in trades if
                      trade['profit_pct'] > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(trade['profit_pct'] for trade in trades if trade['profit_pct'] <= 0) / (
                total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0

        # Calculate final values
        final_stock_value = performance['stock_portfolio'].iloc[-1]
        final_strategy_value = performance['strategy_portfolio'].iloc[-1]

        stock_return_pct = (final_stock_value / initial_capital - 1) * 100
        strategy_return_pct = (final_strategy_value / initial_capital - 1) * 100

        # Calculate annualized return
        start_date = performance.index[0]
        end_date = performance.index[-1]
        years = (end_date - start_date).days / 365.25

        annualized_stock_return = ((1 + stock_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        annualized_strategy_return = ((1 + strategy_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Calculate max drawdown
        rolling_max = performance['strategy_portfolio'].cummax()
        drawdown = (performance['strategy_portfolio'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Detailed metrics
        metrics = {
            "Strategy Name": self.name,
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Win Rate": win_rate,
            "Average Profit (%)": avg_profit,
            "Average Win (%)": avg_win,
            "Average Loss (%)": avg_loss,
            "Final Buy & Hold Value": final_stock_value,
            "Final Strategy Value": final_strategy_value,
            "Buy & Hold Return (%)": stock_return_pct,
            "Strategy Return (%)": strategy_return_pct,
            "Annualized Buy & Hold Return (%)": annualized_stock_return,
            "Annualized Strategy Return (%)": annualized_strategy_return,
            "Maximum Drawdown (%)": max_drawdown,
            "Profit Factor": abs(avg_win * winning_trades) / abs(avg_loss * (total_trades - winning_trades)) if (
                                                                                                                        total_trades - winning_trades) > 0 and avg_loss != 0 else 0,
            "Trade Details": trades
        }

        return performance, metrics


class EMA200CrossoverStrategy(TradingStrategy):
    """Strategy based on price crossing above/below the 200-day EMA"""

    def __init__(self):
        super().__init__("200 EMA Crossover")

    def generate_signals(self, df):
        """Generate buy/sell signals based on EMA crossover/crossunder"""
        buy_signal = (df['previous_close'] < df['previous_ema200']) & (df['close'] > df['EMA_200'])
        sell_signal = (df['previous_close'] > df['previous_ema200']) & (df['close'] < df['EMA_200'])

        return buy_signal, sell_signal


class GoldenCrossStrategy(TradingStrategy):
    """Strategy based on 50-day SMA crossing above/below the 200-day SMA"""

    def __init__(self):
        super().__init__("Golden/Death Cross")

    def generate_signals(self, df):
        """Generate buy/sell signals based on Golden Cross and Death Cross"""
        buy_signal = (df['previous_sma50'] < df['previous_sma200']) & (df['SMA_50'] > df['SMA_200'])
        sell_signal = (df['previous_sma50'] > df['previous_sma200']) & (df['SMA_50'] < df['SMA_200'])

        return buy_signal, sell_signal


class MungerStrategy(TradingStrategy):
    """Strategy inspired by Charlie Munger's approach to buy at 200-week SMA support
    with enhanced profit-taking capabilities"""

    def __init__(self):
        super().__init__("Munger 200-Week SMA with Profit Taking")

    def generate_signals(self, df):
        """Generate buy/sell signals based on Munger approach with profit taking

        Buy signals:
        - When price approaches or touches the 200-week SMA (within 5%)
        - Price shows recovery (close > previous close) after approaching support

        Sell signals:
        - Significant profit achieved (30%+ gain)
        - Price falls below 200-week SMA by more than 20% (stop loss)
        - Significant trend change (price crosses below 50-day SMA after being above)
        """
        # Create tracking arrays
        buy_signal = pd.Series(False, index=df.index)
        sell_signal = pd.Series(False, index=df.index)

        # Use .loc for proper DataFrame assignment to avoid SettingWithCopyWarning
        df_copy = df.copy()  # Create a copy to avoid modifying the original
        df_copy.loc[:, 'price_to_sma200w_ratio'] = df_copy['close'] / df_copy['SMA_200W']

        # Initialize position and entry price trackers
        current_position = 0
        entry_price = 0

        # Near 200-week SMA condition (buy zone)
        near_sma200w = (df_copy['low'] <= df_copy['SMA_200W'] * 1.05) & (df_copy['low'] >= df_copy['SMA_200W'] * 0.95)

        # Iterate through the DataFrame to properly track positions and generate signals
        for i in range(len(df_copy)):
            # Calculate profit if in a position
            profit_pct = ((df_copy['close'].iloc[i] / entry_price) - 1) * 100 if current_position == 1 else 0

            # BUY CONDITIONS
            # Only generate buy signal if not already in a position
            if current_position == 0:
                # Buy near 200W SMA with confirmation of upward movement
                buy_condition = (
                        near_sma200w.iloc[i] and
                        df_copy['close'].iloc[i] > df_copy['previous_close'].iloc[i] and
                        df_copy['close'].iloc[i] > df_copy['SMA_50'].iloc[i] * 0.95  # Near or above 50-day SMA
                )

                if buy_condition:
                    buy_signal.iloc[i] = True
                    current_position = 1
                    entry_price = df_copy['close'].iloc[i]

            # SELL CONDITIONS
            # Only generate sell signals if in a position
            elif current_position == 1:
                # 1. Substantial profit achieved (30%+)
                profit_target_reached = profit_pct >= 30

                # 2. Stop loss - price falls significantly below 200-week SMA
                stop_loss_triggered = df_copy['close'].iloc[i] < df_copy['SMA_200W'].iloc[i] * 0.8

                # 3. Trend change - price falls below 50-day SMA after being above
                trend_change = (
                        df_copy['previous_close'].iloc[i] > df_copy['SMA_50'].iloc[i] and
                        df_copy['close'].iloc[i] < df_copy['SMA_50'].iloc[i]
                )

                # Combined sell condition
                sell_condition = profit_target_reached or stop_loss_triggered or trend_change

                if sell_condition:
                    sell_signal.iloc[i] = True
                    current_position = 0
                    entry_price = 0

        return buy_signal, sell_signal
