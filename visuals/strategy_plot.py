import os
import queue
import threading
from matplotlib import pyplot as plt
import seaborn as sns
from strategies.trading import EMA200CrossoverStrategy, GoldenCrossStrategy, MungerStrategy


class StrategyPlotVisualizer:
    def __init__(self):
        # Ensure the directory exists
        os.makedirs('./visuals/visualizations/', exist_ok=True)
        # Create a queue for visualization tasks
        self.visualization_queue = queue.Queue()
        # Create a dedicated thread for processing visualization tasks
        self.visualization_thread = threading.Thread(target=self._process_visualization_queue, daemon=True)
        self.visualization_thread.start()

    def _process_visualization_queue(self):
        """Process visualization tasks from the queue sequentially"""
        while True:
            try:
                # Get the next task from the queue
                task_type, args, kwargs = self.visualization_queue.get()

                # Execute the appropriate visualization function
                if task_type == 'equity_curves':
                    self._plot_equity_curves(*args, **kwargs)
                elif task_type == 'signals':
                    self._plot_signals(*args, **kwargs)

                # Mark the task as done
                self.visualization_queue.task_done()
            except Exception as e:
                print(f"Error processing visualization task: {e}")
                import traceback
                traceback.print_exc()

    def plot_equity_curves(self, performances, metrics_list, ticker):
        """Queue a task to plot equity curves"""
        self.visualization_queue.put(('equity_curves', (performances, metrics_list, ticker), {}))

    def plot_signals(self, df, ticker):
        """Queue a task to plot signals"""
        self.visualization_queue.put(('signals', (df, ticker), {}))

    def wait_for_completion(self):
        """Wait for all queued visualization tasks to complete"""
        self.visualization_queue.join()

    def _plot_equity_curves(self, performances, metrics_list, ticker):
        """Plot equity curves for all strategies with better error handling and display"""
        # Create a new figure with white background
        plt.figure(figsize=(14, 8), facecolor='white')
        ax = plt.gca()

        # Verify we have valid data before attempting to plot
        if not performances or len(performances) == 0:
            print(f"No performance data available for {ticker}")
            plt.close()
            return

        has_valid_data = False

        # Plot Buy & Hold strategy first
        if performances[0] is not None and 'stock_portfolio' in performances[0].columns and not performances[0][
            'stock_portfolio'].isna().all():
            buy_hold_data = performances[0]['stock_portfolio'].dropna()
            if not buy_hold_data.empty:
                try:
                    # Ensure data is numeric
                    buy_hold_data = buy_hold_data.astype(float)
                    plt.plot(buy_hold_data.index, buy_hold_data.values,
                             label='Buy & Hold', linewidth=2, alpha=0.7)
                    print(f"Plotted Buy & Hold data for {ticker} with {len(buy_hold_data)} points")
                    has_valid_data = True
                except Exception as e:
                    print(f"Error plotting Buy & Hold for {ticker}: {e}")

        # Plot each strategy
        for i, perf in enumerate(performances):
            if perf is None or i >= len(metrics_list):
                continue

            if 'strategy_portfolio' in perf.columns and not perf['strategy_portfolio'].isna().all():
                strategy_data = perf['strategy_portfolio'].dropna()
                if not strategy_data.empty:
                    try:
                        # Ensure data is numeric and finite
                        strategy_data = strategy_data.astype(float)
                        import numpy as np
                        if not np.isfinite(strategy_data).all():
                            print(f"Warning: Non-finite values in strategy data for {metrics_list[i]['Strategy Name']}")
                            strategy_data = strategy_data[np.isfinite(strategy_data)]

                        if not strategy_data.empty:
                            plt.plot(strategy_data.index, strategy_data.values,
                                     label=metrics_list[i]["Strategy Name"],
                                     linewidth=2, alpha=0.8)
                            print(
                                f"Plotted {metrics_list[i]['Strategy Name']} data for {ticker} with {len(strategy_data)} points")
                            has_valid_data = True
                    except Exception as e:
                        print(f"Error plotting {metrics_list[i]['Strategy Name']} for {ticker}: {e}")

        # Format the plot only if we have valid data
        if has_valid_data:
            plt.title(f'Portfolio Value Comparison ({ticker})', fontsize=16)
            plt.ylabel('Portfolio Value ($)', fontsize=14)
            plt.xlabel('Date', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        else:
            print(f"Warning: No valid data lines to plot for {ticker}")
            plt.title(f'No valid data available for {ticker}', fontsize=16)

        plt.tight_layout()

        # Try saving the plot
        file_path = f'./visuals/visualizations/{ticker}_portfolio.png'
        try:
            plt.savefig(file_path, facecolor='white', bbox_inches='tight', dpi=100)
            print(f"Successfully saved plot to {file_path}")
        except Exception as e:
            print(f"Error saving plot to {file_path}: {e}")

        plt.close()

    def _plot_signals(self, df, ticker):
        """Plot price action with buy/sell signals for all strategies"""
        # Check if data exists
        if df is None or df.empty:
            print(f"No data available for plotting signals for {ticker}")
            return

        # Trim data to last 3 years for better visibility
        last_3_years = df.iloc[-756:] if len(df) > 756 else df

        # Check if we have enough data
        if len(last_3_years) < 10:
            print(f"Not enough data for plotting signals for {ticker}")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("colorblind")

        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot price
        ax.plot(last_3_years.index, last_3_years['close'], label='Price', linewidth=1.5)

        # Check if indicators exist before plotting
        if 'EMA_200' in last_3_years.columns:
            ax.plot(last_3_years.index, last_3_years['EMA_200'], label='EMA 200', linewidth=1.5, alpha=0.7)
        if 'SMA_50' in last_3_years.columns:
            ax.plot(last_3_years.index, last_3_years['SMA_50'], label='SMA 50', linewidth=1.5, alpha=0.7)
        if 'SMA_200' in last_3_years.columns:
            ax.plot(last_3_years.index, last_3_years['SMA_200'], label='SMA 200', linewidth=1.5, alpha=0.7)

        # EMA Crossover signals
        try:
            ema_strat = EMA200CrossoverStrategy()
            buy_signal, sell_signal = ema_strat.generate_signals(last_3_years)

            if not buy_signal.empty:
                buy_signals = last_3_years[buy_signal]
                ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100,
                           label='EMA Buy Signal')

            if not sell_signal.empty:
                sell_signals = last_3_years[sell_signal]
                ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100,
                           label='EMA Sell Signal')
        except Exception as e:
            print(f"Error plotting EMA signals for {ticker}: {e}")

        # Golden Cross signals
        try:
            gc_strat = GoldenCrossStrategy()
            buy_signal, sell_signal = gc_strat.generate_signals(last_3_years)

            if not buy_signal.empty:
                golden_cross_signals = last_3_years[buy_signal]
                ax.scatter(golden_cross_signals.index, golden_cross_signals['close'], marker='^', color='gold', s=120,
                           label='Golden Cross')

            if not sell_signal.empty:
                death_cross_signals = last_3_years[sell_signal]
                ax.scatter(death_cross_signals.index, death_cross_signals['close'], marker='v', color='purple', s=120,
                           label='Death Cross')
        except Exception as e:
            print(f"Error plotting Golden Cross signals for {ticker}: {e}")

        # Munger signals
        try:
            munger_strat = MungerStrategy()
            buy_signal, sell_signal = munger_strat.generate_signals(last_3_years)

            if not buy_signal.empty:
                munger_buy = last_3_years[buy_signal]
                if not munger_buy.empty:
                    ax.scatter(munger_buy.index, munger_buy['close'], marker='*', color='blue', s=150,
                               label='Munger Buy Signal')

            if not sell_signal.empty:
                munger_sell = last_3_years[sell_signal]
                if not munger_sell.empty:
                    ax.scatter(munger_sell.index, munger_sell['close'], marker='x', color='black', s=150,
                               label='Munger Sell Signal')
        except Exception as e:
            print(f"Error plotting Munger signals for {ticker}: {e}")

        # Format the plot
        ax.set_title(f'Price Action and Trading Signals - Last 3 Years ({ticker})', fontsize=16)
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        plt.tight_layout()
        plt.savefig(f'./visuals/visualizations/{ticker}_signals.png', bbox_inches='tight')
        plt.close(fig)