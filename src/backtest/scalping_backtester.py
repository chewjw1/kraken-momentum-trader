"""
Multi-pair backtester for scalping strategy.

Tests scalping strategy across multiple pairs and generates
profitability reports to identify which pairs are viable.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..exchange.kraken_client import KrakenClient, OHLC
from ..strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from ..strategy.base_strategy import MarketData
from ..strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime, REGIME_ADJUSTMENTS
from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScalpingTradeResult:
    """Result of a single scalping trade."""
    pair: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_percent: float
    pnl_usd: float
    exit_reason: str
    hold_time_minutes: int


@dataclass
class PairBacktestResult:
    """Backtest results for a single pair."""
    pair: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_percent: float
    total_pnl_usd: float
    avg_pnl_percent: float
    avg_hold_time_minutes: float
    profit_factor: float
    max_drawdown_percent: float
    sharpe_ratio: float
    trades: List[ScalpingTradeResult] = field(default_factory=list)
    is_profitable: bool = False
    recommendation: str = ""


@dataclass
class ScalpingBacktestConfig:
    """Configuration for scalping backtest."""
    initial_capital: float = 10000.0
    position_size_percent: float = 20.0  # Use 20% per trade
    fee_percent: float = 0.26  # Kraken taker fee
    candle_interval: int = 5  # 5-minute candles
    lookback_candles: int = 50  # Need 50 candles for indicators
    slippage_percent: float = 0.05  # Simulated slippage per trade (entry + exit)


class ScalpingBacktester:
    """
    Backtests scalping strategy across multiple pairs.

    Features:
    - Tests configurable list of pairs
    - Generates per-pair profitability metrics
    - Identifies which pairs are viable for scalping
    - Accounts for fees in all calculations
    """

    def __init__(
        self,
        config: Optional[ScalpingBacktestConfig] = None,
        scalping_config: Optional[ScalpingConfig] = None
    ):
        """
        Initialize the scalping backtester.

        Args:
            config: Backtest configuration.
            scalping_config: Scalping strategy configuration.
        """
        self.config = config or ScalpingBacktestConfig()
        self.scalping_config = scalping_config or ScalpingConfig()
        self.strategy = ScalpingStrategy(self.scalping_config)
        self.client = KrakenClient(paper_trading=True)

        # Regime detector — adapts strategy parameters during backtest
        self.regime_detector = RegimeDetector(RegimeConfig())
        self._current_regime = MarketRegime.UNKNOWN

    def fetch_candles(
        self,
        pair: str,
        interval: int = 5,
        since: Optional[datetime] = None
    ) -> List[OHLC]:
        """
        Fetch OHLC candles for backtesting.

        Args:
            pair: Trading pair.
            interval: Candle interval in minutes.
            since: Fetch candles since this time.

        Returns:
            List of OHLC candles.
        """
        since_ts = int(since.timestamp()) if since else None
        return self.client.get_ohlc(pair, interval=interval, since=since_ts)

    def run_pair_backtest(
        self,
        pair: str,
        candles: List[OHLC]
    ) -> PairBacktestResult:
        """
        Run backtest for a single pair.

        Args:
            pair: Trading pair.
            candles: OHLC candles for backtesting.

        Returns:
            PairBacktestResult with metrics.
        """
        trades: List[ScalpingTradeResult] = []
        capital = self.config.initial_capital
        peak_capital = capital

        # Position state
        in_position = False
        entry_price = 0.0
        entry_time = None
        position_size_usd = 0.0
        position_side = "long"

        # Reset regime for each pair backtest
        self._current_regime = MarketRegime.UNKNOWN
        self.strategy = ScalpingStrategy(self.scalping_config)

        # Process each candle
        for i in range(self.config.lookback_candles, len(candles)):
            # Detect regime every 50 candles (avoid overhead)
            if i % 50 == 0:
                self._detect_regime_and_adjust(candles[:i + 1])

            lookback = candles[i - self.config.lookback_candles:i + 1]
            current = candles[i]

            # Build market data
            market_data = self._build_market_data(pair, lookback)

            if in_position:
                # Check exit conditions
                from ..strategy.base_strategy import Position
                position = Position(
                    pair=pair,
                    side=position_side,
                    entry_price=entry_price,
                    current_price=current.close,
                    size=position_size_usd / entry_price,
                    entry_time=entry_time
                )

                signal = self.strategy.analyze(market_data, position)

                exit_signals = ("sell", "close_long") if position_side == "long" else ("close_short",)
                if signal.signal_type.value in exit_signals:
                    # Exit trade (apply slippage: worse exit price)
                    if position_side == "short":
                        # Short exit: slippage makes cover price worse (higher)
                        exit_price = current.close * (1 + self.config.slippage_percent / 100)
                        gross_pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                    else:
                        exit_price = current.close * (1 - self.config.slippage_percent / 100)
                        gross_pnl_percent = ((exit_price - entry_price) / entry_price) * 100

                    fee_cost = self.config.fee_percent * 2  # Round trip
                    net_pnl_percent = gross_pnl_percent - fee_cost
                    pnl_usd = position_size_usd * (net_pnl_percent / 100)

                    capital += pnl_usd
                    peak_capital = max(peak_capital, capital)

                    hold_time = int((current.timestamp - entry_time).total_seconds() / 60)

                    trade = ScalpingTradeResult(
                        pair=pair,
                        entry_time=entry_time,
                        exit_time=current.timestamp,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_percent=net_pnl_percent,
                        pnl_usd=pnl_usd,
                        exit_reason=signal.reason,
                        hold_time_minutes=hold_time
                    )
                    trades.append(trade)

                    in_position = False
                    entry_price = 0.0
                    entry_time = None

            else:
                # Check entry conditions (long and short)
                signal = self.strategy.analyze(market_data, None)

                if signal.signal_type.value == "buy":
                    # Long entry (apply slippage: worse entry price)
                    entry_price = current.close * (1 + self.config.slippage_percent / 100)
                    entry_time = current.timestamp
                    position_size_usd = capital * (self.config.position_size_percent / 100)
                    position_side = "long"
                    in_position = True
                elif signal.signal_type.value == "sell_short":
                    # Short entry (apply slippage: worse entry price for shorts = lower)
                    entry_price = current.close * (1 - self.config.slippage_percent / 100)
                    entry_time = current.timestamp
                    position_size_usd = capital * (self.config.position_size_percent / 100)
                    position_side = "short"
                    in_position = True

        # Calculate metrics
        return self._calculate_pair_metrics(pair, trades, capital)

    def _detect_regime_and_adjust(self, candles: List[OHLC]) -> None:
        """
        Detect market regime from candle history and adjust strategy parameters.

        Matches the live trader behavior: when regime changes, strategy parameters
        (TP multiplier, SL multiplier, confirmations, EMA filter) are adjusted.
        """
        if len(candles) < 200:
            return  # Need enough data for regime detection

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        result = self.regime_detector.detect(closes, highs, lows)

        if result.regime != self._current_regime:
            self._current_regime = result.regime
            adjustments = REGIME_ADJUSTMENTS.get(result.regime, REGIME_ADJUSTMENTS[MarketRegime.UNKNOWN])

            tp_mult = adjustments.get('take_profit_multiplier', 1.0)
            sl_mult = adjustments.get('stop_loss_multiplier', 1.0)
            conf_offset = adjustments.get('min_confirmations_offset', 0)
            ema_enabled = adjustments.get('ema_filter_enabled', True)

            adjusted_config = ScalpingConfig(
                take_profit_percent=self.scalping_config.take_profit_percent * tp_mult,
                stop_loss_percent=self.scalping_config.stop_loss_percent * sl_mult,
                rsi_period=self.scalping_config.rsi_period,
                rsi_oversold=self.scalping_config.rsi_oversold,
                rsi_overbought=self.scalping_config.rsi_overbought,
                bb_period=self.scalping_config.bb_period,
                bb_std_dev=self.scalping_config.bb_std_dev,
                vwap_threshold_percent=self.scalping_config.vwap_threshold_percent,
                volume_spike_threshold=self.scalping_config.volume_spike_threshold,
                min_confirmations=max(1, self.scalping_config.min_confirmations + conf_offset),
                fee_percent=self.scalping_config.fee_percent,
                ema_filter_enabled=ema_enabled,
                # New indicator params
                stoch_k_period=self.scalping_config.stoch_k_period,
                stoch_d_period=self.scalping_config.stoch_d_period,
                stoch_oversold=self.scalping_config.stoch_oversold,
                stoch_overbought=self.scalping_config.stoch_overbought,
                macd_fast=self.scalping_config.macd_fast,
                macd_slow=self.scalping_config.macd_slow,
                macd_signal=self.scalping_config.macd_signal,
                obv_sma_period=self.scalping_config.obv_sma_period,
                atr_period=self.scalping_config.atr_period,
                atr_stop_multiplier=self.scalping_config.atr_stop_multiplier,
                atr_tp_multiplier=self.scalping_config.atr_tp_multiplier,
                use_atr_stops=self.scalping_config.use_atr_stops,
                shorting_enabled=self.scalping_config.shorting_enabled,
                short_min_confirmations=self.scalping_config.short_min_confirmations,
            )
            self.strategy = ScalpingStrategy(adjusted_config)

            logger.info(
                f"Backtest regime change: {result.regime.value} "
                f"(TP x{tp_mult}, SL x{sl_mult}, conf +{conf_offset})"
            )

    def _build_market_data(self, pair: str, candles: List[OHLC]) -> MarketData:
        """Build MarketData from candles."""
        return MarketData(
            pair=pair,
            ohlc=candles,
            prices=[c.close for c in candles],
            volumes=[c.volume for c in candles],
            ticker=None
        )

    def _calculate_pair_metrics(
        self,
        pair: str,
        trades: List[ScalpingTradeResult],
        final_capital: float
    ) -> PairBacktestResult:
        """Calculate performance metrics for a pair."""
        if not trades:
            return PairBacktestResult(
                pair=pair,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                total_pnl_percent=0.0,
                total_pnl_usd=0.0,
                avg_pnl_percent=0.0,
                avg_hold_time_minutes=0.0,
                profit_factor=0.0,
                max_drawdown_percent=0.0,
                sharpe_ratio=0.0,
                trades=[],
                is_profitable=False,
                recommendation="No trades - insufficient signals"
            )

        wins = sum(1 for t in trades if t.pnl_percent > 0)
        losses = len(trades) - wins
        win_rate = wins / len(trades) if trades else 0

        total_pnl_percent = sum(t.pnl_percent for t in trades)
        total_pnl_usd = sum(t.pnl_usd for t in trades)
        avg_pnl_percent = total_pnl_percent / len(trades)
        avg_hold_time = sum(t.hold_time_minutes for t in trades) / len(trades)

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        peak = self.config.initial_capital
        max_dd = 0
        running_capital = self.config.initial_capital
        for t in trades:
            running_capital += t.pnl_usd
            peak = max(peak, running_capital)
            dd = (peak - running_capital) / peak * 100
            max_dd = max(max_dd, dd)

        # Sharpe ratio estimate
        returns = [t.pnl_percent for t in trades]
        avg_return = sum(returns) / len(returns)
        if len(returns) > 1:
            variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = variance ** 0.5
            sharpe = avg_return / std_dev if std_dev > 0 else 0
        else:
            sharpe = 0

        # Determine profitability and recommendation
        is_profitable = total_pnl_usd > 0 and win_rate > 0.45 and profit_factor > 1.0

        if is_profitable and profit_factor > 1.5 and win_rate > 0.55:
            recommendation = "HIGHLY RECOMMENDED - Strong performance"
        elif is_profitable:
            recommendation = "RECOMMENDED - Profitable with acceptable metrics"
        elif win_rate > 0.45 and profit_factor > 0.9:
            recommendation = "MARGINAL - May work with optimization"
        else:
            recommendation = "NOT RECOMMENDED - Poor performance"

        return PairBacktestResult(
            pair=pair,
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl_percent=total_pnl_percent,
            total_pnl_usd=total_pnl_usd,
            avg_pnl_percent=avg_pnl_percent,
            avg_hold_time_minutes=avg_hold_time,
            profit_factor=profit_factor,
            max_drawdown_percent=max_dd,
            sharpe_ratio=sharpe,
            trades=trades,
            is_profitable=is_profitable,
            recommendation=recommendation
        )

    def run_multi_pair_backtest(
        self,
        pairs: List[str],
        days: int = 30
    ) -> Dict[str, PairBacktestResult]:
        """
        Run backtest across multiple pairs.

        Args:
            pairs: List of trading pairs to test.
            days: Number of days of historical data.

        Returns:
            Dictionary mapping pair to results.
        """
        results = {}
        since = datetime.now(timezone.utc) - timedelta(days=days)

        for pair in pairs:
            logger.info(f"Backtesting scalping strategy for {pair}...")

            try:
                candles = self.fetch_candles(
                    pair,
                    interval=self.config.candle_interval,
                    since=since
                )

                if len(candles) < self.config.lookback_candles + 10:
                    logger.warning(f"Insufficient data for {pair}: {len(candles)} candles")
                    results[pair] = PairBacktestResult(
                        pair=pair,
                        total_trades=0,
                        wins=0,
                        losses=0,
                        win_rate=0,
                        total_pnl_percent=0,
                        total_pnl_usd=0,
                        avg_pnl_percent=0,
                        avg_hold_time_minutes=0,
                        profit_factor=0,
                        max_drawdown_percent=0,
                        sharpe_ratio=0,
                        is_profitable=False,
                        recommendation="INSUFFICIENT DATA"
                    )
                    continue

                result = self.run_pair_backtest(pair, candles)
                results[pair] = result

                logger.info(
                    f"{pair}: {result.total_trades} trades, "
                    f"Win rate: {result.win_rate:.1%}, "
                    f"P&L: {result.total_pnl_percent:.2f}%, "
                    f"Recommendation: {result.recommendation}"
                )

            except Exception as e:
                logger.error(f"Error backtesting {pair}: {e}")
                results[pair] = PairBacktestResult(
                    pair=pair,
                    total_trades=0,
                    wins=0,
                    losses=0,
                    win_rate=0,
                    total_pnl_percent=0,
                    total_pnl_usd=0,
                    avg_pnl_percent=0,
                    avg_hold_time_minutes=0,
                    profit_factor=0,
                    max_drawdown_percent=0,
                    sharpe_ratio=0,
                    is_profitable=False,
                    recommendation=f"ERROR: {str(e)}"
                )

        return results

    def generate_report(
        self,
        results: Dict[str, PairBacktestResult],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a profitability report.

        Args:
            results: Backtest results by pair.
            output_path: Optional path to save report.

        Returns:
            Report as string.
        """
        lines = [
            "=" * 80,
            "SCALPING STRATEGY BACKTEST REPORT",
            "=" * 80,
            "",
            f"Strategy: {self.strategy.name}",
            f"Take Profit: {self.scalping_config.take_profit_percent}%",
            f"Stop Loss: {self.scalping_config.stop_loss_percent}%",
            f"Fee: {self.config.fee_percent}%",
            f"Candle Interval: {self.config.candle_interval} min",
            "",
            "-" * 80,
            "PAIR SUMMARY",
            "-" * 80,
            "",
            f"{'Pair':<12} {'Trades':>8} {'Win%':>8} {'P&L%':>10} {'PF':>8} {'Sharpe':>8} {'Recommendation'}",
            "-" * 80,
        ]

        # Sort by profitability
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].total_pnl_percent,
            reverse=True
        )

        profitable_pairs = []
        for pair, result in sorted_results:
            pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < float('inf') else "inf"
            lines.append(
                f"{pair:<12} {result.total_trades:>8} {result.win_rate:>7.1%} "
                f"{result.total_pnl_percent:>9.2f}% {pf_str:>8} {result.sharpe_ratio:>8.2f} "
                f"{result.recommendation}"
            )
            if result.is_profitable:
                profitable_pairs.append(pair)

        lines.extend([
            "",
            "-" * 80,
            "RECOMMENDATIONS",
            "-" * 80,
            "",
        ])

        if profitable_pairs:
            lines.append(f"Profitable pairs for scalping: {', '.join(profitable_pairs)}")
        else:
            lines.append("No pairs recommended for scalping with current settings.")

        lines.extend([
            "",
            "Note: Results are based on historical data and do not guarantee future performance.",
            "Consider paper trading before live deployment.",
            "",
        ])

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def get_profitable_pairs(
        self,
        results: Dict[str, PairBacktestResult],
        min_trades: int = 10,
        min_win_rate: float = 0.45,
        min_profit_factor: float = 1.0
    ) -> List[str]:
        """
        Get list of pairs that meet profitability criteria.

        Args:
            results: Backtest results.
            min_trades: Minimum number of trades.
            min_win_rate: Minimum win rate.
            min_profit_factor: Minimum profit factor.

        Returns:
            List of profitable pair symbols.
        """
        profitable = []
        for pair, result in results.items():
            if (
                result.total_trades >= min_trades
                and result.win_rate >= min_win_rate
                and result.profit_factor >= min_profit_factor
                and result.total_pnl_usd > 0
            ):
                profitable.append(pair)

        return sorted(profitable, key=lambda p: results[p].total_pnl_percent, reverse=True)

    def run_walk_forward(
        self,
        pair: str,
        candles: List[OHLC],
        train_candles: int = 2000,
        test_candles: int = 500,
        step_candles: int = 500
    ) -> Dict[str, object]:
        """
        Walk-forward validation: train on N candles, test on M, slide forward.

        Detects overfitting by comparing in-sample vs out-of-sample performance.

        Args:
            pair: Trading pair.
            candles: Full OHLC dataset.
            train_candles: Number of candles for training window.
            test_candles: Number of candles for testing window.
            step_candles: How far to slide the window forward each iteration.

        Returns:
            Dictionary with walk-forward results and degradation metrics.
        """
        if len(candles) < train_candles + test_candles:
            return {
                "pair": pair,
                "error": f"Need {train_candles + test_candles} candles, have {len(candles)}",
                "windows": [],
            }

        windows = []
        start = 0

        while start + train_candles + test_candles <= len(candles):
            train_data = candles[start:start + train_candles]
            test_data = candles[start + train_candles:start + train_candles + test_candles]

            # Run backtest on both windows
            train_result = self.run_pair_backtest(pair, train_data)
            test_result = self.run_pair_backtest(pair, test_data)

            windows.append({
                "window_start": start,
                "train_trades": train_result.total_trades,
                "train_win_rate": train_result.win_rate,
                "train_pnl_pct": train_result.total_pnl_percent,
                "train_profit_factor": train_result.profit_factor,
                "test_trades": test_result.total_trades,
                "test_win_rate": test_result.win_rate,
                "test_pnl_pct": test_result.total_pnl_percent,
                "test_profit_factor": test_result.profit_factor,
            })

            start += step_candles

        # Calculate degradation between train and test
        if windows:
            avg_train_wr = sum(w["train_win_rate"] for w in windows) / len(windows)
            avg_test_wr = sum(w["test_win_rate"] for w in windows) / len(windows)
            avg_train_pnl = sum(w["train_pnl_pct"] for w in windows) / len(windows)
            avg_test_pnl = sum(w["test_pnl_pct"] for w in windows) / len(windows)

            wr_degradation = avg_train_wr - avg_test_wr
            pnl_degradation = avg_train_pnl - avg_test_pnl

            # If test performance is within 20% of train, strategy is robust
            is_robust = (
                avg_test_pnl > 0
                and wr_degradation < 0.15
                and (avg_test_pnl > avg_train_pnl * 0.5 if avg_train_pnl > 0 else True)
            )
        else:
            avg_train_wr = avg_test_wr = 0
            avg_train_pnl = avg_test_pnl = 0
            wr_degradation = pnl_degradation = 0
            is_robust = False

        return {
            "pair": pair,
            "total_windows": len(windows),
            "avg_train_win_rate": avg_train_wr,
            "avg_test_win_rate": avg_test_wr,
            "win_rate_degradation": wr_degradation,
            "avg_train_pnl_pct": avg_train_pnl,
            "avg_test_pnl_pct": avg_test_pnl,
            "pnl_degradation_pct": pnl_degradation,
            "is_robust": is_robust,
            "windows": windows,
        }
