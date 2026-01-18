# Kraken Momentum Trader

Automated cryptocurrency momentum trading bot for the Kraken exchange. This containerized application monitors market signals and executes momentum trades with built-in risk management features including stop-loss and take-profit mechanisms.

## Features

- **Momentum Trading**: Automatically detects and captures momentum trends using RSI, EMA, and volume indicators
- **Risk Management**: Built-in trailing stop, circuit breaker, and exposure limits
- **Martingale System**: Optional position building when price drops (with safeguards)
- **Kraken Integration**: Direct integration with Kraken's REST API
- **Containerized**: Docker-ready for easy deployment on any platform
- **Cloud Ready**: Designed for deployment on Replit, AWS, GCP, or any container platform
- **Configurable**: Extensive configuration options via YAML and environment variables
- **Paper Trading**: Safe simulation mode for strategy validation
- **State Persistence**: Crash recovery with automatic state saving

## Project Structure

```
kraken-momentum-trader/
├── src/
│   ├── core/
│   │   ├── trader.py           # Main trading orchestrator
│   │   └── state_machine.py    # Trading state management
│   ├── exchange/
│   │   ├── kraken_client.py    # Kraken API wrapper
│   │   └── rate_limiter.py     # API rate limiting
│   ├── strategy/
│   │   ├── base_strategy.py    # Strategy interface
│   │   ├── momentum_strategy.py # Momentum implementation
│   │   └── signals/
│   │       ├── rsi.py          # RSI indicator
│   │       ├── ema.py          # EMA crossover
│   │       └── volume.py       # Volume signals
│   ├── risk/
│   │   ├── risk_manager.py     # Risk orchestrator
│   │   ├── position_sizer.py   # Position sizing
│   │   └── circuit_breaker.py  # Emergency stops
│   ├── persistence/
│   │   └── file_store.py       # File-based storage
│   ├── config/
│   │   ├── settings.py         # Config loading
│   │   └── platform.py         # Platform detection
│   └── observability/
│       ├── logger.py           # Structured logging
│       └── metrics.py          # Performance tracking
├── tests/
│   ├── unit/
│   │   ├── test_signals.py
│   │   ├── test_risk.py
│   │   └── test_state_machine.py
│   └── backtest/
├── config/
│   └── config.yaml             # Base configuration
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── DESIGN.md                   # Design documentation
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Kraken account with API keys

### 1. Clone the Repository

```bash
git clone https://github.com/chewjw1/kraken-momentum-trader.git
cd kraken-momentum-trader
```

### 2. Configure Environment

Copy the example environment file and add your Kraken API credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
KRAKEN_API_KEY=your_actual_api_key_here
KRAKEN_API_SECRET=your_actual_api_secret_here
```

### 3. Run with Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 4. Run Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the bot
python main.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KRAKEN_API_KEY` | Your Kraken API key | Required |
| `KRAKEN_API_SECRET` | Your Kraken API secret | Required |
| `PAPER_TRADING` | Enable paper trading mode | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Configuration File (config/config.yaml)

```yaml
trading:
  pairs: ["BTC/USD", "ETH/USD"]
  check_interval_seconds: 60
  paper_trading: true

strategy:
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
  ema_short_period: 12
  ema_long_period: 26

risk:
  max_position_size_usd: 5000
  max_position_percent: 40.0
  max_total_exposure_percent: 200.0  # Increased for Martingale
  use_trailing_stop: true
  trailing_stop_percent: 5.0
  trailing_stop_activation_percent: 5.0
  circuit_breaker_consecutive_losses: 5
  circuit_breaker_cooldown_hours: 2

  martingale:
    enabled: true
    max_entries: 4              # Initial + 3 add-ons
    size_multiplier: 1.25       # Each add-on is 1.25x previous
    add_on_drop_percent: 5.0    # Add when price drops 5% from avg
    require_rsi_oversold: true  # Must meet RSI condition
```

## CLI Options

```bash
python main.py                    # Run the trader
python main.py --config config.yaml  # Use specific config
python main.py --validate         # Validate strategy metrics
python main.py --show-config      # Show current config
```

## Trading Strategy

### Entry Conditions (ALL must be met)

- RSI < 30 (oversold condition)
- Price above long-term EMA (uptrend confirmation)
- Volume above average (momentum confirmation)

### Exit Conditions (ANY triggers exit)

- Trailing stop hit (price drops 5% from peak after 5% profit reached)
- RSI > 70 (overbought condition)
- Bearish EMA crossover

## Risk Management

| Safeguard | Default | Description |
|-----------|---------|-------------|
| Max Position Size | 40% / $5,000 | Maximum per trade |
| Max Total Exposure | 200% | Maximum capital in positions (for Martingale) |
| Trailing Stop Activation | 5% profit | Trailing stop activates after this gain |
| Trailing Stop | 5% from peak | Exit when price drops this much from peak |
| Circuit Breaker | 5 losses | 2-hour cooldown after consecutive losses |

**Note:** No fixed stop loss - positions are held until profitable, then protected by trailing stop.

### Martingale Position Building

When enabled, the bot can add to losing positions with the following safeguards:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Add-on Trigger | 5% drop | Add when price drops 5% from average entry |
| Size Multiplier | 1.25x | Each add-on is 1.25x the previous entry |
| Max Entries | 4 | Maximum entries (initial + 3 add-ons) |
| RSI Requirement | Oversold | Only add when RSI indicates oversold |

The 200% exposure limit ensures Martingale cannot grow unbounded.

## Deployment

### Replit

1. Import the GitHub repository into Replit
2. Add environment variables in Replit Secrets
3. Run with `python main.py`

### AWS/GCP

1. **Build Docker image**:
   ```bash
   docker build -t kraken-momentum-trader .
   ```

2. **Push to container registry** (AWS ECR or GCP Container Registry)

3. **Deploy** to:
   - AWS: ECS Fargate, ECS EC2, or Lambda
   - GCP: Cloud Run, GKE, or Compute Engine

### Docker Compose (Local/VPS)

```bash
docker-compose up -d
```

## Security Best Practices

- Never commit your `.env` file or API keys
- Use API keys with minimal required permissions
- For production, use secrets management (AWS Secrets Manager, GCP Secret Manager)
- Regularly rotate your API keys
- Monitor your trading activity regularly

## Getting Kraken API Keys

1. Log in to your Kraken account
2. Navigate to Settings > API
3. Create a new API key with these permissions:
   - Query Funds
   - Query Open Orders & Trades
   - Query Closed Orders & Trades
   - Create & Modify Orders
4. Save your key and secret securely

## Disclaimer

**Trading Risk Warning**:

Cryptocurrency trading involves substantial risk of loss. This bot is provided for educational purposes. Use at your own risk. The authors are not responsible for any financial losses.

- Paper trade for at least 2 weeks before using real capital
- Never invest more than you can afford to lose
- Monitor your bot regularly
- Past performance does not guarantee future results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Add backtesting framework
- [ ] Implement multiple trading strategies
- [ ] Add web dashboard for monitoring
- [ ] Support for additional exchanges
- [ ] Machine learning-based signal generation
- [ ] Telegram/Discord notifications
- [ ] DynamoDB persistence for AWS

## Support

If you encounter any issues or have questions:

- Open an issue on GitHub
- Check existing issues for solutions
- Review the [DESIGN.md](DESIGN.md) for architecture details
