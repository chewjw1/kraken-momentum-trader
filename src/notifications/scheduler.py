"""
Scheduler for daily summary notifications.
Sends summary at 5:00 PM ET each day.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from ..observability.logger import get_logger

logger = get_logger(__name__)

# Eastern Time zone
ET = ZoneInfo("America/New_York")


class DailySummaryScheduler:
    """
    Scheduler that triggers daily summary at a specific time.

    Runs in a background thread and calls the provided callback
    at 5:00 PM ET each day.
    """

    def __init__(
        self,
        callback: Callable[[], None],
        target_hour: int = 17,  # 5 PM
        target_minute: int = 0,
        timezone: ZoneInfo = ET
    ):
        """
        Initialize the scheduler.

        Args:
            callback: Function to call when summary should be sent.
            target_hour: Hour to trigger (24-hour format, default 17 = 5 PM).
            target_minute: Minute to trigger (default 0).
            timezone: Timezone for scheduling (default Eastern Time).
        """
        self.callback = callback
        self.target_hour = target_hour
        self.target_minute = target_minute
        self.timezone = timezone
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_run_date: Optional[str] = None

    def start(self) -> None:
        """Start the scheduler in a background thread."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"Daily summary scheduler started - will run at "
            f"{self.target_hour:02d}:{self.target_minute:02d} {self.timezone}"
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("Daily summary scheduler stopped")

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running and not self._stop_event.is_set():
            try:
                now = datetime.now(self.timezone)
                today_str = now.strftime("%Y-%m-%d")

                # Check if we should run
                if self._should_run(now, today_str):
                    logger.info("Triggering daily summary")
                    self._last_run_date = today_str

                    try:
                        self.callback()
                    except Exception as e:
                        logger.error(f"Daily summary callback failed: {e}")

                # Calculate sleep time until next check
                # Check every minute, but sleep in short intervals for responsiveness
                sleep_time = self._get_sleep_time(now)
                self._stop_event.wait(timeout=sleep_time)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                self._stop_event.wait(timeout=60)

    def _should_run(self, now: datetime, today_str: str) -> bool:
        """
        Check if the summary should run now.

        Args:
            now: Current datetime.
            today_str: Today's date string.

        Returns:
            True if should run.
        """
        # Don't run if already ran today
        if self._last_run_date == today_str:
            return False

        # Check if we're at or past the target time
        if now.hour == self.target_hour and now.minute >= self.target_minute:
            return True

        # Also trigger if we're past the target hour (in case bot was down)
        if now.hour > self.target_hour:
            return True

        return False

    def _get_sleep_time(self, now: datetime) -> float:
        """
        Calculate how long to sleep until next check.

        Args:
            now: Current datetime.

        Returns:
            Sleep time in seconds.
        """
        # If we're close to the target time, check more frequently
        target_today = now.replace(
            hour=self.target_hour,
            minute=self.target_minute,
            second=0,
            microsecond=0
        )

        if now < target_today:
            time_until = (target_today - now).total_seconds()
            if time_until < 120:  # Within 2 minutes
                return 10  # Check every 10 seconds
            elif time_until < 3600:  # Within 1 hour
                return 60  # Check every minute
            else:
                return 300  # Check every 5 minutes
        else:
            # Already past target, check every 5 minutes
            return 300

    def get_next_run_time(self) -> datetime:
        """
        Get the next scheduled run time.

        Returns:
            Next run datetime.
        """
        now = datetime.now(self.timezone)
        target_today = now.replace(
            hour=self.target_hour,
            minute=self.target_minute,
            second=0,
            microsecond=0
        )

        # If already past today's target or already ran today
        if now >= target_today or self._last_run_date == now.strftime("%Y-%m-%d"):
            return target_today + timedelta(days=1)

        return target_today

    def trigger_now(self) -> None:
        """
        Manually trigger the daily summary immediately.
        Useful for testing or manual reports.
        """
        logger.info("Manual daily summary trigger")
        try:
            self.callback()
        except Exception as e:
            logger.error(f"Manual summary trigger failed: {e}")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running
