"""
Discord webhook notifications for trade events and daily summaries.
"""

import requests
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from ..observability.logger import get_logger

logger = get_logger(__name__)


class NotificationType(Enum):
    """Types of notifications."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    MARTINGALE_ADD = "martingale_add"
    ERROR = "error"
    DAILY_SUMMARY = "daily_summary"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


@dataclass
class TradeNotification:
    """Data for trade notifications."""
    pair: str
    side: str
    price: float
    size: float
    size_usd: float
    reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    entry_num: Optional[int] = None  # For martingale
    avg_entry: Optional[float] = None  # For martingale


@dataclass
class DailySummaryData:
    """Data for daily summary notifications."""
    date: str
    total_trades: int
    wins: int
    losses: int
    daily_pnl: float
    total_pnl: float
    win_rate: float
    current_capital: float
    open_position: Optional[dict] = None
    sharpe_ratio: float = 0.0
    max_drawdown_percent: float = 0.0


class DiscordNotifier:
    """
    Discord webhook notifier for trade events.

    Sends formatted embed messages for:
    - Trade entries
    - Trade exits (with P&L)
    - Martingale add-ons
    - Errors
    - Daily summaries
    """

    # Color codes for embeds
    COLORS = {
        NotificationType.TRADE_ENTRY: 0x00FF00,      # Green
        NotificationType.TRADE_EXIT: 0x0099FF,       # Blue
        NotificationType.MARTINGALE_ADD: 0xFFAA00,   # Orange
        NotificationType.ERROR: 0xFF0000,            # Red
        NotificationType.DAILY_SUMMARY: 0x9933FF,    # Purple
        NotificationType.STARTUP: 0x00FFFF,          # Cyan
        NotificationType.SHUTDOWN: 0x808080,         # Gray
    }

    def __init__(self, webhook_url: str, enabled: bool = True):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL.
            enabled: Whether notifications are enabled.
        """
        self.webhook_url = webhook_url
        self.enabled = enabled
        self._session = requests.Session()

        if enabled and webhook_url:
            logger.info("Discord notifications enabled")
        else:
            logger.info("Discord notifications disabled")

    def _send_webhook(self, payload: dict) -> bool:
        """
        Send a webhook request to Discord.

        Args:
            payload: The webhook payload.

        Returns:
            True if successful.
        """
        if not self.enabled or not self.webhook_url:
            return False

        try:
            response = self._session.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 204:
                logger.debug("Discord notification sent successfully")
                return True
            else:
                logger.warning(
                    f"Discord webhook returned {response.status_code}: {response.text}"
                )
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def _create_embed(
        self,
        title: str,
        description: str,
        color: int,
        fields: list[dict],
        footer: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> dict:
        """
        Create a Discord embed object.

        Args:
            title: Embed title.
            description: Embed description.
            color: Embed color (hex).
            fields: List of field dicts with name, value, inline.
            footer: Optional footer text.
            timestamp: Optional timestamp.

        Returns:
            Embed dictionary.
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
        }

        if footer:
            embed["footer"] = {"text": footer}

        if timestamp:
            embed["timestamp"] = timestamp.isoformat()
        else:
            embed["timestamp"] = datetime.now(timezone.utc).isoformat()

        return embed

    def notify_trade_entry(self, trade: TradeNotification, paper_trading: bool = True) -> bool:
        """
        Send notification for trade entry.

        Args:
            trade: Trade notification data.
            paper_trading: Whether this is paper trading.

        Returns:
            True if sent successfully.
        """
        mode_indicator = " [PAPER]" if paper_trading else ""

        embed = self._create_embed(
            title=f"{'BUY' if trade.side == 'buy' else 'SELL'} {trade.pair}{mode_indicator}",
            description=f"New position opened",
            color=self.COLORS[NotificationType.TRADE_ENTRY],
            fields=[
                {"name": "Price", "value": f"${trade.price:,.2f}", "inline": True},
                {"name": "Size", "value": f"{trade.size:.6f}", "inline": True},
                {"name": "Value", "value": f"${trade.size_usd:,.2f}", "inline": True},
                {"name": "Reason", "value": trade.reason or "Signal triggered", "inline": False},
            ],
            footer="Kraken Momentum Trader"
        )

        return self._send_webhook({"embeds": [embed]})

    def notify_trade_exit(self, trade: TradeNotification, paper_trading: bool = True) -> bool:
        """
        Send notification for trade exit.

        Args:
            trade: Trade notification data.
            paper_trading: Whether this is paper trading.

        Returns:
            True if sent successfully.
        """
        mode_indicator = " [PAPER]" if paper_trading else ""
        pnl_emoji = "" if trade.pnl >= 0 else ""
        pnl_color = 0x00FF00 if trade.pnl >= 0 else 0xFF0000

        embed = self._create_embed(
            title=f"{pnl_emoji} CLOSE {trade.pair}{mode_indicator}",
            description=f"Position closed",
            color=pnl_color,
            fields=[
                {"name": "Exit Price", "value": f"${trade.price:,.2f}", "inline": True},
                {"name": "Size", "value": f"{trade.size:.6f}", "inline": True},
                {"name": "P&L", "value": f"${trade.pnl:+,.2f} ({trade.pnl_percent:+.2f}%)", "inline": True},
                {"name": "Reason", "value": trade.reason or "Exit signal", "inline": False},
            ],
            footer="Kraken Momentum Trader"
        )

        return self._send_webhook({"embeds": [embed]})

    def notify_martingale_add(self, trade: TradeNotification, paper_trading: bool = True) -> bool:
        """
        Send notification for martingale add-on.

        Args:
            trade: Trade notification data.
            paper_trading: Whether this is paper trading.

        Returns:
            True if sent successfully.
        """
        mode_indicator = " [PAPER]" if paper_trading else ""

        embed = self._create_embed(
            title=f"MARTINGALE ADD #{trade.entry_num} {trade.pair}{mode_indicator}",
            description=f"Adding to position (averaging down)",
            color=self.COLORS[NotificationType.MARTINGALE_ADD],
            fields=[
                {"name": "Add Price", "value": f"${trade.price:,.2f}", "inline": True},
                {"name": "Add Size", "value": f"{trade.size:.6f}", "inline": True},
                {"name": "Add Value", "value": f"${trade.size_usd:,.2f}", "inline": True},
                {"name": "New Avg Entry", "value": f"${trade.avg_entry:,.2f}", "inline": True},
                {"name": "Entry #", "value": f"{trade.entry_num} of 4", "inline": True},
                {"name": "Reason", "value": trade.reason or "Price drop triggered", "inline": False},
            ],
            footer="Kraken Momentum Trader"
        )

        return self._send_webhook({"embeds": [embed]})

    def notify_error(self, error_type: str, message: str, details: Optional[str] = None) -> bool:
        """
        Send notification for errors.

        Args:
            error_type: Type of error.
            message: Error message.
            details: Optional additional details.

        Returns:
            True if sent successfully.
        """
        fields = [
            {"name": "Error Type", "value": error_type, "inline": True},
            {"name": "Message", "value": message[:1000], "inline": False},
        ]

        if details:
            fields.append({"name": "Details", "value": details[:1000], "inline": False})

        embed = self._create_embed(
            title="ERROR",
            description="An error occurred in the trading bot",
            color=self.COLORS[NotificationType.ERROR],
            fields=fields,
            footer="Kraken Momentum Trader"
        )

        return self._send_webhook({"embeds": [embed]})

    def notify_daily_summary(self, summary: DailySummaryData, paper_trading: bool = True) -> bool:
        """
        Send daily performance summary.

        Args:
            summary: Daily summary data.
            paper_trading: Whether this is paper trading.

        Returns:
            True if sent successfully.
        """
        mode_indicator = " [PAPER]" if paper_trading else ""
        pnl_emoji = "" if summary.daily_pnl >= 0 else ""

        fields = [
            {"name": "Date", "value": summary.date, "inline": True},
            {"name": "Trades Today", "value": str(summary.total_trades), "inline": True},
            {"name": "Win/Loss", "value": f"{summary.wins}W / {summary.losses}L", "inline": True},
            {"name": f"{pnl_emoji} Daily P&L", "value": f"${summary.daily_pnl:+,.2f}", "inline": True},
            {"name": "Total P&L", "value": f"${summary.total_pnl:+,.2f}", "inline": True},
            {"name": "Win Rate", "value": f"{summary.win_rate:.1f}%", "inline": True},
            {"name": "Capital", "value": f"${summary.current_capital:,.2f}", "inline": True},
            {"name": "Sharpe Ratio", "value": f"{summary.sharpe_ratio:.2f}", "inline": True},
            {"name": "Max Drawdown", "value": f"{summary.max_drawdown_percent:.1f}%", "inline": True},
        ]

        # Add open position info if exists
        if summary.open_position:
            pos = summary.open_position
            fields.append({
                "name": "Open Position",
                "value": f"{pos['pair']}: {pos['size']:.4f} @ ${pos['entry_price']:,.2f} (entries: {pos['num_entries']})",
                "inline": False
            })

        embed = self._create_embed(
            title=f"Daily Summary{mode_indicator}",
            description=f"Performance report for {summary.date}",
            color=self.COLORS[NotificationType.DAILY_SUMMARY],
            fields=fields,
            footer="Kraken Momentum Trader - 5:00 PM ET"
        )

        return self._send_webhook({"embeds": [embed]})

    def notify_startup(self, pairs: list[str], paper_trading: bool = True) -> bool:
        """
        Send notification when bot starts.

        Args:
            pairs: List of trading pairs.
            paper_trading: Whether this is paper trading.

        Returns:
            True if sent successfully.
        """
        mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"

        embed = self._create_embed(
            title=f"Bot Started - {mode}",
            description="Kraken Momentum Trader is now running",
            color=self.COLORS[NotificationType.STARTUP],
            fields=[
                {"name": "Mode", "value": mode, "inline": True},
                {"name": "Pairs", "value": ", ".join(pairs), "inline": False},
            ],
            footer="Kraken Momentum Trader"
        )

        return self._send_webhook({"embeds": [embed]})

    def notify_shutdown(self, reason: str = "Manual shutdown") -> bool:
        """
        Send notification when bot shuts down.

        Args:
            reason: Shutdown reason.

        Returns:
            True if sent successfully.
        """
        embed = self._create_embed(
            title="Bot Stopped",
            description="Kraken Momentum Trader has stopped",
            color=self.COLORS[NotificationType.SHUTDOWN],
            fields=[
                {"name": "Reason", "value": reason, "inline": False},
            ],
            footer="Kraken Momentum Trader"
        )

        return self._send_webhook({"embeds": [embed]})

    def close(self) -> None:
        """Close the session."""
        self._session.close()
