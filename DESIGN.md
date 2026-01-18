# Kraken Momentum Trader - Design Document

## Overview

A cryptocurrency momentum trading application that connects to the Kraken exchange for automated trading using a multi-indicator momentum strategy with comprehensive risk management.

**CRITICAL**: Start with paper trading for 2+ weeks before using real capital. Cryptocurrency trading involves significant risk of loss.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Trader    │  │  Strategy   │  │   Risk Manager      │  │
│  │   Engine    │  │  (Momentum) │  │   (Safeguards)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   INTEGRATION LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Kraken    │  │   Config    │  │   Persistence       │  │
│  │   Client    │  │   Loader    │  │   (State)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   PLATFORM LAYER                             │
│  ┌──────────────────────┐  ┌─────────────────────────────┐  │
│  │  Replit (Dev/Test)   │  │  AWS (Production)           │  │
│  │  - Env vars secrets  │  │  - Secrets Manager          │  │
│  │  - File-based state  │  │  - DynamoDB state           │  │
│  │  - Console logging   │  │  - CloudWatch logs          │  │
│  └──────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

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
│   └── unit/
│       ├── test_signals.py
│       ├── test_risk.py
│       └── test_state_machine.py
├── config/
│   └── config.yaml             # Base configuration
├── main.py                     # Entry point
├── requirements.txt
└── DESIGN.md
```

## Key Components

### 1. Trader Engine (`src/core/trader.py`)

The main orchestrator that coordinates all components:

- Initializes and manages exchange client, strategy, and risk manager
- Runs the main trading loop
- Handles state persistence for crash recovery
- Manages graceful shutdown

### 2. State Machine (`src/core/state_machine.py`)

Manages trading lifecycle states:

```
INITIALIZING → IDLE ⟷ ANALYZING → ENTERING → IN_POSITION → EXITING
                ↓                     ↓            ↓           ↓
              ERROR ←─────────────────┴────────────┴───────────┘
                ↓
             STOPPED
```

### 3. Kraken Client (`src/exchange/kraken_client.py`)

- REST API wrapper with retry logic
- Rate limiting (15-20 calls/min for private endpoints)
- Order placement, balance queries, price data
- Paper trading simulation mode

### 4. Momentum Strategy (`src/strategy/momentum_strategy.py`)

Multi-indicator confirmation using:

**Entry Conditions (ALL must be met):**
- RSI < 30 (oversold)
- Price above long-term EMA (uptrend)
- Volume above average (confirmation)

**Exit Conditions (ANY):**
- Trailing stop triggered (price drops 5% from peak after 5% profit)
- RSI > 70 (overbought)
- Bearish EMA crossover detected

### 5. Risk Manager (`src/risk/risk_manager.py`)

**Risk safeguards:**

| Rule | Limit | Action |
|------|-------|--------|
| Max position size | 40% of capital / $5,000 | Reduce position |
| Max total exposure | 200% of capital | Block new trades |
| Consecutive losses | 5 losses | 2-hour cooldown |

**Trailing Stop Strategy:**
- No initial stop loss (position held until profitable)
- Trailing stop activates after 5% profit
- Exit when price drops 5% from peak

**Martingale Position Building:**
- If price drops 5% from average entry, add to position
- Each add-on is 1.25x the size of the previous entry
- Maximum 4 total entries (initial + 3 add-ons)
- Add-ons require RSI to be oversold (< 30)
- Exposure limit (200%) prevents unlimited pyramiding

### 6. Circuit Breaker (`src/risk/circuit_breaker.py`)

States:
- **CLOSED**: Normal operation
- **OPEN**: Cooldown period after consecutive losses
- **EMERGENCY**: Critical condition, exit all positions

## Configuration

See `config/config.yaml` for all configurable parameters:

```yaml
trading:
  pairs: ["BTC/USD", "ETH/USD"]
  check_interval_seconds: 60
  paper_trading: true  # CRITICAL: Keep true until validated

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

  # Martingale configuration
  martingale:
    enabled: true
    max_entries: 4              # Initial + 3 add-ons
    size_multiplier: 1.25       # Each add-on is 1.25x previous
    add_on_drop_percent: 5.0    # Add when price drops 5% from avg
    require_rsi_oversold: true  # Must meet RSI condition
    require_ema_trend: false    # EMA check optional
```

## Running the Application

### Development / Paper Trading

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (or use .env file)
export KRAKEN_API_KEY=your_key
export KRAKEN_API_SECRET=your_secret

# Run with paper trading (default)
python main.py

# Show configuration
python main.py --show-config

# Validate strategy metrics
python main.py --validate
```

### Running Tests

```bash
pytest tests/unit/ -v
```

## Risk Warnings

### Strategy Viability Concerns

| Risk | Impact | Mitigation |
|------|--------|------------|
| Transaction fees | 0.32-0.52% round-trip | Only trade when expected gain > 1.5% |
| Whipsaws | 40-60% signal failure in choppy markets | Multi-indicator confirmation |
| Flash crashes | Sudden large losses | Strict stop-losses, circuit breaker |
| Crypto volatility | Extreme price swings | Position limits, max exposure caps |

### Validation Requirements

Before considering live trading:

1. **Paper trade for minimum 2 weeks**
2. **Achieve these metrics:**
   - Sharpe ratio > 1.0
   - Max drawdown < 20%
   - Win rate > 40% with positive expectancy
   - Minimum 20 trades

Use `python main.py --validate` to check current metrics.

## AWS Production Deployment

| Component | Service | Purpose |
|-----------|---------|---------|
| Compute | ECS Fargate | Always-on container |
| Secrets | Secrets Manager | API keys |
| State | DynamoDB | Trade history, positions |
| Logging | CloudWatch | Logs + metrics |
| Alerts | SNS | Error notifications |

## Future Enhancements

- [ ] DynamoDB persistence implementation
- [ ] CloudFormation/Terraform templates
- [ ] Backtesting framework
- [ ] Multiple strategy support
- [ ] Web dashboard for monitoring
- [ ] Telegram/Discord notifications

## License

MIT License - See LICENSE file for details.
