#!/usr/bin/env python3
"""
Kraken Momentum Trading Bot
Main entry point for the automated trading application
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trader import MomentumTrader
from src.logger import setup_logger

def main():
      """Main function to run the trading bot"""

    # Load environment variables
      load_dotenv()

    # Setup logging
      logger = setup_logger()
      logger.info("Starting Kraken Momentum Trading Bot...")

    # Validate environment variables
      required_vars = ['KRAKEN_API_KEY', 'KRAKEN_API_SECRET']
      missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
              logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
              logger.error("Please set these variables in your .env file")
              sys.exit(1)

    try:
              # Initialize trader
              trader = MomentumTrader()

        # Start trading loop
              logger.info("Trader initialized. Starting trading loop...")
              trader.run()

except KeyboardInterrupt:
          logger.info("Received shutdown signal. Stopping trader...")
          sys.exit(0)
except Exception as e:
          logger.error(f"Fatal error: {str(e)}", exc_info=True)
          sys.exit(1)

if __name__ == "__main__":
      main()
