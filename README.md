# Kraken Momentum Trader

Automated cryptocurrency momentum trading bot for the Kraken exchange. This containerized application monitors market signals and executes momentum trades with built-in risk management features including stop-loss and take-profit mechanisms.

## Features

- **Momentum Trading**: Automatically detects and captures momentum trends
- - **Risk Management**: Built-in stop-loss and take-profit mechanisms
  - - **Kraken Integration**: Direct integration with Kraken's REST API
    - - **Containerized**: Docker-ready for easy deployment on any platform
      - - **Cloud Ready**: Designed for deployment on Replit, AWS, GCP, or any container platform
        - - **Configurable**: Extensive configuration options via environment variables
          - - **Logging**: Comprehensive logging for monitoring and debugging
           
            - ## Project Structure
           
            - ```
              kraken-momentum-trader/
              ├── src/                    # Source code directory
              │   ├── __init__.py        # Package initializer
              │   ├── trader.py          # Main trading logic
              │   ├── kraken_client.py   # Kraken API client
              │   ├── signals.py         # Technical indicators and signals
              │   ├── risk_manager.py    # Risk management logic
              │   └── logger.py          # Logging configuration
              ├── logs/                   # Log files (gitignored)
              ├── data/                   # Trading data and state (gitignored)
              ├── .env                    # Environment variables (gitignored)
              ├── .env.example            # Example environment configuration
              ├── .gitignore              # Git ignore file
              ├── Dockerfile              # Docker configuration
              ├── docker-compose.yml      # Docker Compose configuration
              ├── main.py                 # Application entry point
              ├── requirements.txt        # Python dependencies
              ├── LICENSE                 # MIT License
              └── README.md               # This file
              ```

              ## Quick Start

              ### Prerequisites

              - Python 3.11+
              - - Docker (optional, for containerized deployment)
                - - Kraken account with API keys
                 
                  - ### 1. Clone the Repository
                 
                  - ```bash
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
                    # Install dependencies
                    pip install -r requirements.txt

                    # Run the bot
                    python main.py
                    ```

                    ## Configuration

                    All configuration is done via environment variables in the `.env` file:

                    | Variable | Description | Default |
                    |----------|-------------|---------|
                    | `KRAKEN_API_KEY` | Your Kraken API key | Required |
                    | `KRAKEN_API_SECRET` | Your Kraken API secret | Required |
                    | `TRADING_PAIRS` | Comma-separated trading pairs | `BTC/USD,ETH/USD` |
                    | `CHECK_INTERVAL_SECONDS` | How often to check signals | `60` |
                    | `MOMENTUM_THRESHOLD` | Momentum signal threshold | `0.02` (2%) |
                    | `STOP_LOSS_PERCENTAGE` | Stop-loss percentage | `0.05` (5%) |
                    | `TAKE_PROFIT_PERCENTAGE` | Take-profit percentage | `0.10` (10%) |
                    | `MAX_POSITION_SIZE_USD` | Maximum position size | `1000` |
                    | `MAX_DAILY_TRADES` | Maximum trades per day | `10` |
                    | `LOG_LEVEL` | Logging level | `INFO` |

                    ## Development with Claude Code

                    This project is designed to work seamlessly with Claude Code:

                    1. **Clone the Repository**: Use the Git integration in your IDE
                    2. 2. **Connect Claude Code**: Point Claude Code to the repository directory
                       3. 3. **Start Coding**: Claude Code will understand the project structure and help you develop
                         
                          4. ### Suggested Development Tasks
                         
                          5. - Implement technical indicators (RSI, MACD, Bollinger Bands)
                             - - Add backtesting capabilities
                               - - Implement additional exchange connectors
                                 - - Create a web dashboard for monitoring
                                   - - Add unit tests and integration tests
                                    
                                     - ## Deployment
                                    
                                     - ### Replit
                                    
                                     - 1. Import the GitHub repository into Replit
                                       2. 2. Add environment variables in Replit Secrets
                                          3. 3. Run with `python main.py` or use Docker
                                            
                                             4. ### AWS/GCP
                                            
                                             5. 1. **Build Docker image**:
                                                2.    ```bash
                                                         docker build -t kraken-momentum-trader .
                                                         ```

                                                      2. **Push to container registry** (AWS ECR or GCP Container Registry)
                                                  
                                                      3. 3. **Deploy** to:
                                                         4.    - AWS: ECS, Fargate, or EC2
                                                               -    - GCP: Cloud Run, GKE, or Compute Engine
                                                                
                                                                    - ### Local Development
                                                                
                                                                    - ```bash
                                                                      # Create virtual environment
                                                                      python -m venv venv
                                                                      source venv/bin/activate  # On Windows: venv\Scripts\activate

                                                                      # Install dependencies
                                                                      pip install -r requirements.txt

                                                                      # Run
                                                                      python main.py
                                                                      ```

                                                                      ## Security Best Practices

                                                                      ⚠️ **Important Security Notes**:

                                                                      - Never commit your `.env` file or API keys
                                                                      - - Use API keys with minimal required permissions
                                                                        - - For production, consider using secrets management services (AWS Secrets Manager, GCP Secret Manager)
                                                                          - - Regularly rotate your API keys
                                                                            - - Monitor your trading activity regularly
                                                                             
                                                                              - ## Getting Kraken API Keys
                                                                             
                                                                              - 1. Log in to your Kraken account
                                                                                2. 2. Navigate to Settings → API
                                                                                   3. 3. Create a new API key with these permissions:
                                                                                      4.    - Query Funds
                                                                                            -    - Query Open Orders & Trades
                                                                                                 -    - Query Closed Orders & Trades
                                                                                                      -    - Create & Modify Orders
                                                                                                           - 4. Save your key and secret securely
                                                                                                            
                                                                                                             5. ## Disclaimer
                                                                                                            
                                                                                                             6. ⚠️ **Trading Risk Warning**:
                                                                                                            
                                                                                                             7. Cryptocurrency trading involves substantial risk of loss. This bot is provided for educational purposes. Use at your own risk. The authors are not responsible for any financial losses.
                                                                                                            
                                                                                                             8. - Always start with small amounts
                                                                                                                - - Test thoroughly in a paper trading environment first
                                                                                                                  - - Never invest more than you can afford to lose
                                                                                                                    - - Monitor your bot regularly
                                                                                                                     
                                                                                                                      - ## Contributing
                                                                                                                     
                                                                                                                      - Contributions are welcome! Please feel free to submit a Pull Request.
                                                                                                                     
                                                                                                                      - 1. Fork the repository
                                                                                                                        2. 2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
                                                                                                                           3. 3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
                                                                                                                              4. 4. Push to the branch (`git push origin feature/AmazingFeature`)
                                                                                                                                 5. 5. Open a Pull Request
                                                                                                                                   
                                                                                                                                    6. ## License
                                                                                                                                   
                                                                                                                                    7. This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
                                                                                                                                   
                                                                                                                                    8. ## Support
                                                                                                                                   
                                                                                                                                    9. If you encounter any issues or have questions:
                                                                                                                                   
                                                                                                                                    10. - Open an issue on GitHub
                                                                                                                                        - - Check existing issues for solutions
                                                                                                                                          - - Review the documentation
                                                                                                                                           
                                                                                                                                            - ## Roadmap
                                                                                                                                           
                                                                                                                                            - - [ ] Add backtesting framework
                                                                                                                                              - [ ] - [ ] Implement multiple trading strategies
                                                                                                                                              - [ ] - [ ] Add web dashboard for monitoring
                                                                                                                                              - [ ] - [ ] Support for additional exchanges
                                                                                                                                              - [ ] - [ ] Machine learning-based signal generation
                                                                                                                                              - [ ] - [ ] Telegram/Discord notifications
                                                                                                                                              - [ ] - [ ] Advanced risk management features
                                                                                                                                             
                                                                                                                                              - [ ] ## Acknowledgments
                                                                                                                                             
                                                                                                                                              - [ ] - Kraken Exchange for their comprehensive API
                                                                                                                                              - [ ] - The Python trading community
                                                                                                                                              - [ ] - All contributors to this project
                                                                                                                                             
                                                                                                                                              - [ ] ---
                                                                                                                                             
                                                                                                                                              - [ ] **Happy Trading! 🚀📈**
