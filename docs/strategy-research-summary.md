# Strategy Research Summary

## Date: 2026-02-19

## Objective
Evaluate approaches to generate consistent, meaningful returns (15%+) on ~$10K capital for a US-based crypto trader across sideways, bull, and bear markets.

---

## Phase 1: Current Strategy Review

### Existing System (Scalping Strategy)
- **Architecture:** Multi-indicator mean-reversion scalper (RSI, Bollinger Bands, VWAP, Volume)
- **Direction:** Long-only
- **Timeframe:** 60-minute candles
- **Risk:** 5% TP / 2.5% SL (optimized config), 3+ confirmations required
- **Execution:** Maker orders on Kraken (0.16% fee)

### Key Issues Found
1. **Dead code:** `check_trailing_stop()` exists but is never called by the live runner
2. **Unused indicator:** EMA is computed but never used in entry/exit logic
3. **VWAP never resets:** Unconventional — standard VWAP resets per session
4. **No compounding:** Position sizing uses `initial_capital`, not current equity
5. **Overfitting risk:** 10 parameters optimized on 118 days of data (Lopez de Prado framework recommends max 3-5)
6. **Low trade frequency:** ~1 trade/week per pair

### Backtest Results (118 days, Oct 2025 - Jan 2026)
| Pair | Win Rate | Return | Trades |
|------|----------|--------|--------|
| BTC/USD | 58.8% | +2.90% | 17 |
| ETH/USD | 41.7% | -2.59% | 24 |
| SOL/USD | 60.0% | +1.29% | 40 |
| XRP/USD | 50.0% | +3.00% | 32 |

---

## Phase 2: Strategy Theses Evaluated

### Thesis 1: Regime-Adaptive Dual Strategy
- EMA circuit breaker + separate trending/ranging strategies
- **Verdict:** Good as an overlay, not standalone

### Thesis 2: Volatility Harvesting Grid
- Grid trading with fixed buy/sell levels
- **Real data result:** -22.83% over 22 months on BTC ($10K capital)
- Grid sold out during Nov 2024 rally, accumulated underwater inventory during dips
- $732 in fees on 472 fills
- **Verdict:** Mathematically proven zero expected value before fees (Chen et al. 2025). Does not work.

### Thesis 3: Time-of-Day Momentum
- Exploit UTC timezone patterns in crypto
- **Verdict:** Academic evidence exists but edge is thin, needs careful implementation

### Thesis 4: TTM Squeeze Breakout
- Bollinger/Keltner channel squeeze detection
- **Verdict:** Low parameter count = less overfitting risk, but needs 3 new indicators

### Thesis 5: Funding Rate Arbitrage
- Long spot + short perpetual = delta-neutral, collect funding
- Ethena at $15B validates at scale
- BIS: crypto carry is 10x larger than equity carry
- **US reality at 1x leverage:** 1.3-5.0% net on Coinbase CFM
- **At 2x:** 5-16% (liquidates at +50% spike)
- **Verdict:** Best documented risk-adjusted return across all asset classes, but US access limitations severely cap returns

---

## Phase 3: Exchange Fee Analysis

| Exchange | Maker | Taker | US Access |
|----------|-------|-------|-----------|
| Bitfinex | 0% | 0% | No |
| MEXC | 0% | 0.02% | No |
| Coinbase (perps) | 0% promo | 0.02-0.03% | Yes |
| Kraken | 0.14% | 0.24% | Yes |
| Kraken+ ($4.99/mo) | 0% | 0% | Yes |

**Key finding:** Kraken is one of the most expensive exchanges for active trading. Switching to Kraken+ ($4.99/mo) would eliminate fees entirely.

---

## Phase 4: Traditional Markets Comparison

| Market | Strategy | Expected Return | Notes |
|--------|----------|----------------|-------|
| FX Carry (G10) | Interest rate differential | 2-5% | "Almost ceased" post-2008 |
| FX Grid | Grid trading | 0% documented | No academic support |
| FX Vol Selling | Strangles/condors | 3-5% unlevered | Tail risk |
| Commodity Roll | Calendar spread | 3-6% | Lumpy, volatile |
| CTA Trend | Managed futures | 4.5-7.6% | Needs $50K+ |

**Conclusion:** No traditional market strategy reliably generates 15%+. Crypto funding rate arb has the highest documented carry across all asset classes (BIS research).

---

## Phase 5: Options & Derivatives Analysis

### US-Accessible Crypto Options Platforms
- IBIT ETF options (via Schwab/Fidelity/IBKR)
- CME micro futures/options (via futures brokers, ~$3,267 margin/contract)
- Crypto.com (limited liquidity)
- Bitnomial (nascent, CFTC-regulated)
- Deribit: **BLOCKED for US users** (85% of global crypto options volume)

### Why Options Don't Help at $10K
1. **Selling volatility has negative expected value** over a full crypto cycle (Oct 2025 crash: -47%)
2. **CME basis trade is dead:** Went from 20%+ annualized (2024) to -2.35% (Dec 2025)
3. **Options + funding arb combo breaks delta-neutrality** — covered calls on spot leg add short gamma
4. **Protective puts cost more than funding income** at small scale
5. **Structured products ("dual investment")** are just repackaged options selling — US blocked anyway

### Strategy Comparison (US Retail, $10K)

| Strategy | Annual Return (net) | Max DD | US Access | Key Risk |
|---|---|---|---|---|
| IBIT covered calls | 8-15% premium | 50%+ | Yes | Full BTC downside |
| CME basis trade | 0-5% currently | 10-15% | Yes | Negative basis |
| Funding rate arb (1x) | 1.3-5% | 5-15% | Yes | Exchange risk |
| Selling BTC strangles | -20% to +15% | 100%+ | Limited | Ruin risk |
| Refined scalping | 5-20%+ | 20-30% | Yes | Alpha decay |

---

## Final Recommendation

### Priority Order for $10K US-Based Trader

1. **Refine the existing scalping strategy**
   - Fix dead code (trailing stops, EMA usage)
   - Add EMA circuit breaker (~50 lines)
   - Subscribe to Kraken+ ($4.99/mo) for zero fees
   - Improve signal quality > adding new strategies
   - This is where marginal time investment yields the highest return

2. **Conservative IBIT covered calls (supplement)**
   - Buy ~$5K IBIT, sell monthly calls 10-15% OTM
   - 2-4%/month premium income
   - Full BTC downside exposure remains

3. **Build monitoring infrastructure for future opportunities**
   - CME basis spread monitor
   - Funding rate tracker across exchanges
   - Alert when arb opportunities exceed 10% annualized
   - Deploy capital when conditions improve (next bull phase)

### What NOT to Do
- Sell naked crypto options (ruin risk from fat tails)
- Use offshore platforms as a US person
- Chase structured product APY labels
- Overcomplicate a $10K account with multi-leg derivative strategies
- Over-optimize on historical data (10 params on 118 days = overfitting)

---

## Key Research Sources
- Lopez de Prado: Overfitting framework (3-5 params max)
- Chen et al. 2025: Grid trading zero expected value proof
- BIS: Crypto carry 10x larger than equity carry, uncorrelated
- Ethena: $15B funding rate arb validation
- CME Group: Micro BTC futures margin ~$3,267/contract
- Coinbase-Deribit acquisition ($2.9B) may change US options landscape
