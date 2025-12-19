"""
XAU/USD H4 Bias Trading System - Complete Web Application
Features: Live signals, backtesting, email alerts, performance analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Trading system configuration"""
    SYMBOL = "GC=F"  # Gold futures as proxy for XAU/USD
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 1.5
    EMA_FAST = 20
    EMA_SLOW = 50
    MIN_PULLBACK_PIPS = 20
    TP1_PIPS = 300
    TP2_PIPS = 500
    TP3_PIPS = 700
    MIN_SL_PIPS = 40
    MAX_SL_PIPS = 80
    POINT = 0.10 # For XAU/USD

# ==================== DATA HANDLER ====================
class DataHandler:
    """Handle live and historical data fetching"""
    
    @staticmethod
    def get_live_data(period="5d", interval="15m"):
        """Fetch live market data"""
        try:
            ticker = yf.Ticker(Config.SYMBOL)
            df = ticker.history(period=period, interval=interval)
            df['Symbol'] = 'XAU/USD'
            return df
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            return None
    
    @staticmethod
    def get_historical_data(start_date, end_date, interval="1h"):
        """Fetch historical data for backtesting"""
        try:
            ticker = yf.Ticker(Config.SYMBOL)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            df['Symbol'] = 'XAU/USD'
            return df
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return None

# ==================== INDICATOR LOGIC ====================
class H4BiasSystem:
    """Core trading logic implementation"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.signals = []
        self.reference_candles = {}
        
    def find_reference_candle(self, date):
        """Find first H4 candle of the day"""
        day_data = self.data[self.data.index.date == date]
        if len(day_data) > 0:
            first_candle = day_data.iloc[0]
            return {
                'high': first_candle['High'],
                'low': first_candle['Low'],
                'open': first_candle['Open'],
                'close': first_candle['Close'],
                'time': first_candle.name
            }
        return None
    
    def check_bias_confirmation(self, current_idx):
        """Check if bias is confirmed by H4 close"""
        current_date = self.data.index[current_idx].date()
        
        if current_date not in self.reference_candles:
            ref_candle = self.find_reference_candle(current_date)
            if ref_candle:
                self.reference_candles[current_date] = ref_candle
            else:
                return None
        
        ref = self.reference_candles[current_date]
        current_close = self.data.iloc[current_idx]['Close']
        
        # Bullish bias: close above reference high
        if current_close > ref['high']:
            return 'bullish'
        # Bearish bias: close below reference low
        elif current_close < ref['low']:
            return 'bearish'
        
        return None
    
    def calculate_indicators(self):
        """Calculate EMAs and ATR"""
        # EMA indicators
        ema_fast = EMAIndicator(self.data['Close'], window=Config.EMA_FAST)
        ema_slow = EMAIndicator(self.data['Close'], window=Config.EMA_SLOW)
        self.data['EMA_Fast'] = ema_fast.ema_indicator()
        self.data['EMA_Slow'] = ema_slow.ema_indicator()
        
        # ATR for dynamic stop loss
        atr = AverageTrueRange(self.data['High'], self.data['Low'], 
                              self.data['Close'], window=Config.ATR_PERIOD)
        self.data['ATR'] = atr.average_true_range()
        
    def check_entry_signal(self, idx, bias, ref_candle):
        """Check for M15 entry signals based on bias"""
        if idx < Config.EMA_SLOW:
            return None
        
        row = self.data.iloc[idx]
        prev_row = self.data.iloc[idx-1]
        
        ema_fast = row['EMA_Fast']
        ema_slow = row['EMA_Slow']
        atr = row['ATR']
        
        signal = None
        
        # BULLISH ENTRY
        if bias == 'bullish':
            pullback_pips = (ref_candle['high'] - row['Low']) / Config.POINT
            
            if (pullback_pips >= Config.MIN_PULLBACK_PIPS and
                row['Close'] > ema_fast and
                ema_fast > ema_slow and
                row['Close'] > row['Open']):
                
                entry_price = row['Close']
                stop_loss = row['Low'] - (atr * Config.ATR_MULTIPLIER)
                sl_pips = (entry_price - stop_loss) / Config.POINT
                
                # Adjust SL to reasonable range
                if sl_pips < Config.MIN_SL_PIPS:
                    stop_loss = entry_price - Config.MIN_SL_PIPS * Config.POINT
                    sl_pips = Config.MIN_SL_PIPS
                elif sl_pips > Config.MAX_SL_PIPS:
                    stop_loss = entry_price - Config.MAX_SL_PIPS * Config.POINT
                    sl_pips = Config.MAX_SL_PIPS
                
                signal = {
                    'type': 'BUY',
                    'time': row.name,
                    'entry': entry_price,
                    'sl': stop_loss,
                    'tp1': ref_candle['high'] + Config.TP1_PIPS * Config.POINT,
                    'tp2': ref_candle['high'] + Config.TP2_PIPS * Config.POINT,
                    'tp3': ref_candle['high'] + Config.TP3_PIPS * Config.POINT,
                    'sl_pips': sl_pips,
                    'bias': 'bullish',
                    'ref_high': ref_candle['high'],
                    'ref_low': ref_candle['low']
                }
        
        # BEARISH ENTRY
        elif bias == 'bearish':
            pullback_pips = (row['High'] - ref_candle['low']) / Config.POINT
            
            if (pullback_pips >= Config.MIN_PULLBACK_PIPS and
                row['Close'] < ema_fast and
                ema_fast < ema_slow and
                row['Close'] < row['Open']):
                
                entry_price = row['Close']
                stop_loss = row['High'] + (atr * Config.ATR_MULTIPLIER)
                sl_pips = (stop_loss - entry_price) / Config.POINT
                
                # Adjust SL to reasonable range
                if sl_pips < Config.MIN_SL_PIPS:
                    stop_loss = entry_price + Config.MIN_SL_PIPS * Config.POINT
                    sl_pips = Config.MIN_SL_PIPS
                elif sl_pips > Config.MAX_SL_PIPS:
                    stop_loss = entry_price + Config.MAX_SL_PIPS * Config.POINT
                    sl_pips = Config.MAX_SL_PIPS
                
                signal = {
                    'type': 'SELL',
                    'time': row.name,
                    'entry': entry_price,
                    'sl': stop_loss,
                    'tp1': ref_candle['low'] - Config.TP1_PIPS * Config.POINT,
                    'tp2': ref_candle['low'] - Config.TP2_PIPS * Config.POINT,
                    'tp3': ref_candle['low'] - Config.TP3_PIPS * Config.POINT,
                    'sl_pips': sl_pips,
                    'bias': 'bearish',
                    'ref_high': ref_candle['high'],
                    'ref_low': ref_candle['low']
                }
        
        return signal
    
    def generate_signals(self):
        """Main signal generation logic"""
        self.calculate_indicators()
        self.signals = []
        current_bias = None
        signal_active = False
        current_date = None
        
        for idx in range(len(self.data)):
            date = self.data.index[idx].date()
            
            # Reset on new day
            if date != current_date:
                current_date = date
                current_bias = None
                signal_active = False
            
            # Check bias confirmation
            if current_bias is None:
                current_bias = self.check_bias_confirmation(idx)
            
            # Check for entry signal
            if current_bias and not signal_active:
                if date in self.reference_candles:
                    ref = self.reference_candles[date]
                    signal = self.check_entry_signal(idx, current_bias, ref)
                    if signal:
                        self.signals.append(signal)
                        signal_active = True
        
        return self.signals

# ==================== BACKTESTING ENGINE ====================
class Backtester:
    """Backtest the trading strategy"""
    
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals
        self.trades = []
        
    def run_backtest(self):
        """Execute backtest on all signals"""
        for signal in self.signals:
            trade = self.simulate_trade(signal)
            if trade:
                self.trades.append(trade)
        
        return self.analyze_results()
    
    def simulate_trade(self, signal):
        """Simulate individual trade execution"""
        entry_time = signal['time']
        entry_idx = self.data.index.get_loc(entry_time)
        
        # Look forward from entry
        future_data = self.data.iloc[entry_idx:]
        
        trade_result = {
            'signal': signal,
            'entry_time': entry_time,
            'entry_price': signal['entry'],
            'type': signal['type']
        }
        
        # Check each bar for SL or TP hit
        for idx, row in future_data.iterrows():
            if signal['type'] == 'BUY':
                # Check stop loss
                if row['Low'] <= signal['sl']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['sl']
                    trade_result['result'] = 'SL Hit'
                    trade_result['pips'] = -signal['sl_pips']
                    return trade_result
                
                # Check take profits
                if row['High'] >= signal['tp3']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['tp3']
                    trade_result['result'] = 'TP3 Hit'
                    trade_result['pips'] = Config.TP3_PIPS
                    return trade_result
                elif row['High'] >= signal['tp2']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['tp2']
                    trade_result['result'] = 'TP2 Hit'
                    trade_result['pips'] = Config.TP2_PIPS
                    return trade_result
                elif row['High'] >= signal['tp1']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['tp1']
                    trade_result['result'] = 'TP1 Hit'
                    trade_result['pips'] = Config.TP1_PIPS
                    return trade_result
            
            else:  # SELL
                # Check stop loss
                if row['High'] >= signal['sl']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['sl']
                    trade_result['result'] = 'SL Hit'
                    trade_result['pips'] = -signal['sl_pips']
                    return trade_result
                
                # Check take profits
                if row['Low'] <= signal['tp3']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['tp3']
                    trade_result['result'] = 'TP3 Hit'
                    trade_result['pips'] = Config.TP3_PIPS
                    return trade_result
                elif row['Low'] <= signal['tp2']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['tp2']
                    trade_result['result'] = 'TP2 Hit'
                    trade_result['pips'] = Config.TP2_PIPS
                    return trade_result
                elif row['Low'] <= signal['tp1']:
                    trade_result['exit_time'] = idx
                    trade_result['exit_price'] = signal['tp1']
                    trade_result['result'] = 'TP1 Hit'
                    trade_result['pips'] = Config.TP1_PIPS
                    return trade_result
        
        # Trade still open
        trade_result['result'] = 'Open'
        return None
    
    def analyze_results(self):
        """Generate comprehensive performance analytics"""
        if not self.trades:
            return None
        
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        winners = df[df['pips'] > 0]
        losers = df[df['pips'] < 0]
        
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        total_pips = df['pips'].sum()
        avg_win = winners['pips'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pips'].mean() if len(losers) > 0 else 0
        
        # Risk-reward metrics
        if avg_loss != 0:
            profit_factor = abs(avg_win * len(winners) / (avg_loss * len(losers)))
        else:
            profit_factor = float('inf') if len(winners) > 0 else 0
        
        # Time analysis
        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['entry_time']).dt.dayofweek
        
        best_hour = df.groupby('hour')['pips'].mean().idxmax() if len(df) > 0 else None
        best_day = df.groupby('day_of_week')['pips'].mean().idxmax() if len(df) > 0 else None
        
        # Consecutive wins/losses
        df['win'] = df['pips'] > 0
        df['streak'] = (df['win'] != df['win'].shift()).cumsum()
        max_consecutive_wins = df[df['win']].groupby('streak').size().max() if len(winners) > 0 else 0
        max_consecutive_losses = df[~df['win']].groupby('streak').size().max() if len(losers) > 0 else 0
        
        # Drawdown analysis
        df['cumulative_pips'] = df['pips'].cumsum()
        df['running_max'] = df['cumulative_pips'].expanding().max()
        df['drawdown'] = df['running_max'] - df['cumulative_pips']
        max_drawdown = df['drawdown'].max()
        
        results = {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_hour': best_hour,
            'best_day': best_day,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_drawdown,
            'trades_df': df
        }
        
        return results

# ==================== EMAIL ALERTS ====================
class EmailAlerts:
    """Send email notifications for trading signals"""
    
    @staticmethod
    def send_alert(signal, smtp_config):
        """Send email alert for new signal"""
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = smtp_config['to_email']
            msg['Subject'] = f"🚨 XAU/USD {signal['type']} SIGNAL ALERT"
            
            body = f"""
            <html>
            <body>
            <h2>XAU/USD H4 Bias Trading Signal</h2>
            <h3 style="color: {'green' if signal['type'] == 'BUY' else 'red'};">
                {signal['type']} SIGNAL GENERATED
            </h3>
            
            <p><strong>Entry Price:</strong> {signal['entry']:.2f}</p>
            <p><strong>Stop Loss:</strong> {signal['sl']:.2f} ({signal['sl_pips']:.0f} pips)</p>
            
            <h4>Take Profit Levels:</h4>
            <ul>
                <li>TP1: {signal['tp1']:.2f} (+{Config.TP1_PIPS} pips)</li>
                <li>TP2: {signal['tp2']:.2f} (+{Config.TP2_PIPS} pips)</li>
                <li>TP3: {signal['tp3']:.2f} (+{Config.TP3_PIPS} pips)</li>
            </ul>
            
            <p><strong>Bias:</strong> {signal['bias'].upper()}</p>
            <p><strong>Time:</strong> {signal['time']}</p>
            
            <hr>
            <p><em>Automated alert from XAU/USD H4 Bias Trading System</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(smtp_config['from_email'], smtp_config['password'])
                server.send_message(msg)
            
            return True
        except Exception as e:
            st.error(f"Email alert failed: {e}")
            return False

# ==================== VISUALIZATION ====================
class Visualizer:
    """Create interactive charts and dashboards"""
    
    @staticmethod
    def plot_backtest_equity_curve(results):
        """Plot equity curve from backtest"""
        df = results['trades_df']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['entry_time'],
            y=df['cumulative_pips'],
            mode='lines',
            name='Cumulative Pips',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['entry_time'],
            y=df['running_max'],
            mode='lines',
            name='Peak Equity',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Equity Curve - Cumulative Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Pips',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_trade_distribution(results):
        """Plot trade results distribution"""
        df = results['trades_df']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Pips Distribution', 'Win/Loss by Hour')
        )
        
        # Histogram of pips
        fig.add_trace(
            go.Histogram(x=df['pips'], nbinsx=20, name='Pips',
                        marker_color='lightblue'),
            row=1, col=1
        )
        
        # Win rate by hour
        hourly = df.groupby('hour').agg({
            'pips': ['mean', 'count']
        }).reset_index()
        
        fig.add_trace(
            go.Bar(x=hourly['hour'], y=hourly['pips']['mean'],
                  name='Avg Pips', marker_color='purple'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    @staticmethod
    def plot_live_chart(data, signal=None):
        """Plot live price chart with signals"""
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='XAU/USD'
        ))
        
        # Add EMAs
        if 'EMA_Fast' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['EMA_Fast'],
                mode='lines',
                name='EMA 20',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['EMA_Slow'],
                mode='lines',
                name='EMA 50',
                line=dict(color='blue', width=1)
            ))
        
        # Add signal markers
        if signal:
            fig.add_trace(go.Scatter(
                x=[signal['time']],
                y=[signal['entry']],
                mode='markers+text',
                name=f"{signal['type']} Signal",
                marker=dict(
                    size=15,
                    color='green' if signal['type'] == 'BUY' else 'red',
                    symbol='triangle-up' if signal['type'] == 'BUY' else 'triangle-down'
                ),
                text=[signal['type']],
                textposition='top center'
            ))
            
            # Add TP/SL lines
            fig.add_hline(y=signal['tp1'], line_dash="dash", line_color="green",
                         annotation_text="TP1")
            fig.add_hline(y=signal['sl'], line_dash="dash", line_color="red",
                         annotation_text="SL")
        
        fig.update_layout(
            title='XAU/USD Live Chart',
            yaxis_title='Price',
            xaxis_title='Date',
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        return fig

# ==================== STREAMLIT WEB APP ====================
def main():
    st.set_page_config(page_title="XAU/USD H4 Trading System", layout="wide", page_icon="📈")
    
    st.title("🏆 XAU/USD H4 Bias Trading System")
    st.markdown("**Advanced Trading Signal Generator with Backtesting Dashboard**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio("Select Mode", ["📊 Live Signals", "🔬 Backtest Dashboard"])
    
    # Email configuration
    with st.sidebar.expander("📧 Email Alert Settings"):
        enable_email = st.checkbox("Enable Email Alerts")
        if enable_email:
            smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            from_email = st.text_input("From Email")
            to_email = st.text_input("To Email")
            password = st.text_input("Password", type="password")
            
            smtp_config = {
                'smtp_server': smtp_server,
                'smtp_port': smtp_port,
                'from_email': from_email,
                'to_email': to_email,
                'password': password
            }
    
    # ========== LIVE SIGNALS MODE ==========
    if mode == "📊 Live Signals":
        st.header("📡 Live Trading Signals")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🔄 Refresh Data", type="primary"):
                st.rerun()
            
            st.info("**System Status**\n\n✅ Active\n\n🔍 Monitoring XAU/USD")
        
        with col1:
            # Fetch live data
            with st.spinner("Fetching live market data..."):
                live_data = DataHandler.get_live_data(period="5d", interval="15m")
            
            if live_data is not None and len(live_data) > 0:
                # Generate signals
                system = H4BiasSystem(live_data)
                signals = system.generate_signals()
                
                # Display current signal
                if signals:
                    latest_signal = signals[-1]
                    
                    signal_type = latest_signal['type']
                    color = "green" if signal_type == "BUY" else "red"
                    
                    st.success(f"### 🎯 {signal_type} SIGNAL DETECTED")
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Entry Price", f"${latest_signal['entry']:.2f}")
                    col_b.metric("Stop Loss", f"${latest_signal['sl']:.2f}", 
                                f"-{latest_signal['sl_pips']:.0f} pips")
                    col_c.metric("Risk/Reward", f"1:{Config.TP1_PIPS/latest_signal['sl_pips']:.1f}")
                    
                    st.markdown("**Take Profit Levels:**")
                    tp_col1, tp_col2, tp_col3 = st.columns(3)
                    tp_col1.info(f"TP1: ${latest_signal['tp1']:.2f}\n\n+{Config.TP1_PIPS} pips")
                    tp_col2.info(f"TP2: ${latest_signal['tp2']:.2f}\n\n+{Config.TP2_PIPS} pips")
                    tp_col3.info(f"TP3: ${latest_signal['tp3']:.2f}\n\n+{Config.TP3_PIPS} pips")
                    
                    # Send email alert
                    if enable_email and 'smtp_config' in locals():
                        if st.button("📧 Send Email Alert"):
                            if EmailAlerts.send_alert(latest_signal, smtp_config):
                                st.success("Email alert sent successfully!")
                    
                    # Plot chart
                    st.plotly_chart(
                        Visualizer.plot_live_chart(system.data, latest_signal),
                        use_container_width=True
                    )
                else:
                    st.warning("⏳ No active signals. Waiting for bias confirmation...")
                    st.plotly_chart(
                        Visualizer.plot_live_chart(system.data),
                        use_container_width=True
                    )
                
                # Display all signals
                if signals:
                    st.subheader("📋 Recent Signals History")
                    signals_df = pd.DataFrame(signals)
                    st.dataframe(signals_df[['time', 'type', 'entry', 'sl', 'tp1', 'sl_pips']],
                               use_container_width=True)
    
    # ========== BACKTEST MODE ==========
    else:
        st.header("🔬 Backtesting Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      value=datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running comprehensive backtest..."):
                # Fetch historical data
                hist_data = DataHandler.get_historical_data(start_date, end_date, interval="1h")
                
                if hist_data is not None and len(hist_data) > 100:
                    # Generate signals
                    system = H4BiasSystem(hist_data)
                    signals = system.generate_signals()
                    
                    if signals:
                        # Run backtest
                        backtester = Backtester(system.data, signals)
                        results = backtester.run_backtest()
                        
                        if results:
                            # Performance metrics
                            st.success("### 📊 Backtest Results")
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Total Trades", results['total_trades'])
                            metric_col2.metric("Win Rate", f"{results['win_rate']:.1f}%")
                            metric_col3.metric("Total Pips", f"{results['total_pips']:.0f}")
                            metric_col4.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                            
                            # Additional metrics
                            metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
                            metric_col5.metric("Avg Win", f"{results['avg_win']:.0f} pips")
                            metric_col6.metric("Avg Loss", f"{results['avg_loss']:.0f} pips")
                            metric_col7.metric("Max Drawdown", f"{results['max_drawdown']:.0f} pips")
                            metric_col8.metric("Winners/Losers", f"{results['winners']}/{results['losers']}")
                            
                            # Equity curve
                            st.plotly_chart(
                                Visualizer.plot_backtest_equity_curve(results),
                                use_container_width=True
                            )
                            
                            # Trade distribution
                            st.plotly_chart(
                                Visualizer.plot_trade_distribution(results),
                                use_container_width=True
                            )
                            
                            # Performance insights
                            st.subheader("💡 Performance Insights & Recommendations")
                            
                            insights_col1, insights_col2 = st.columns(2)
                            
                            with insights_col1:
                                st.markdown("**🕐 Best Trading Time**")
                                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                best_day_name = day_names[results['best_day']] if results['best_day'] is not None else 'N/A'
                                st.info(f"Best Hour: {results['best_hour']}:00 UTC\n\nBest Day: {best_day_name}")
                                
                                st.markdown("**📈 Risk Management**")
                                if results['win_rate'] >= 60:
                                    st.success(f"✅ Strong win rate ({results['win_rate']:.1f}%). Consider increasing position size.")
                                elif results['win_rate'] >= 50:
                                    st.info(f"⚖️ Decent win rate ({results['win_rate']:.1f}%). Maintain current risk levels.")
                                else:
                                    st.warning(f"⚠️ Win rate needs improvement ({results['win_rate']:.1f}%). Consider tightening entry criteria.")
                            
                            with insights_col2:
                                st.markdown("**🎯 Optimization Suggestions**")
                                suggestions = []
                                
                                if results['profit_factor'] > 2.0:
                                    suggestions.append("✅ Excellent profit factor - system is robust")
                                elif results['profit_factor'] < 1.5:
                                    suggestions.append("⚠️ Consider widening TP levels or tightening entry filters")
                                
                                if results['max_consecutive_losses'] > 5:
                                    suggestions.append(f"⚠️ Max consecutive losses: {results['max_consecutive_losses']}. Consider implementing cooling-off period")
                                
                                if results['max_drawdown'] > 500:
                                    suggestions.append(f"⚠️ High drawdown ({results['max_drawdown']:.0f} pips). Reduce position sizing")
                                
                                if results['avg_win'] / abs(results['avg_loss']) < 2.0:
                                    suggestions.append("💡 Risk-reward ratio could be improved. Consider partial TP strategy")
                                
                                if not suggestions:
                                    suggestions.append("✅ Strategy performance is well-balanced")
                                
                                for suggestion in suggestions:
                                    st.write(suggestion)
                            
                            # Trade log
                            st.subheader("📋 Complete Trade Log")
                            trades_display = results['trades_df'][['entry_time', 'type', 'entry_price', 
                                                                   'exit_price', 'result', 'pips']].copy()
                            trades_display['entry_time'] = trades_display['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
                            st.dataframe(trades_display, use_container_width=True)
                            
                            # Download results
                            csv = trades_display.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Trade Log (CSV)",
                                data=csv,
                                file_name=f"backtest_results_{start_date}_{end_date}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No completed trades in backtest period")
                    else:
                        st.warning("No signals generated in the selected period. Try a longer date range.")
                else:
                    st.error("Insufficient historical data. Please adjust date range.")

if __name__ == "__main__":
    main()
