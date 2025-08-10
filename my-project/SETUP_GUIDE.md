# Complete Setup Guide for Algo-Trading System

This guide will walk you through setting up the complete algo-trading system with all integrations.

## 📋 Prerequisites

- Python 3.8+
- Git
- Internet connection for API access

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd my-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
nano .env  # or use your preferred editor
```

## 🔑 API Configuration

### Alpha Vantage API (Required)

The system needs Alpha Vantage for stock data. The free tier provides:
- 5 requests per minute
- 500 requests per day
- Real-time and historical data

**Setup Steps:**

1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. Get your API key
4. Add it to your `.env` file:
   ```
   ALPHA_VANTAGE_API_KEY=your_actual_api_key_here
   ```

### Google Sheets Integration (Optional)

For automated trade logging and performance tracking.

**Setup Steps:**

1. **Create Google Cloud Project:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Google Sheets API:**
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

3. **Create Service Account:**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in the details and create

4. **Download Credentials:**
   - Click on your service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create New Key" > "JSON"
   - Download the JSON file
   - Rename it to `google_sheets_credentials.json`
   - Place it in the `credentials/` directory

5. **Create Google Sheet:**
   - Create a new Google Sheet
   - Copy the spreadsheet ID from the URL
   - Share the sheet with your service account email (found in the JSON file)
   - Give "Editor" permissions

6. **Update .env file:**
   ```
   GOOGLE_SHEETS_CREDENTIALS_FILE=credentials/google_sheets_credentials.json
   GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id_here
   ```

### Telegram Bot Integration (Optional)

For real-time trading alerts and notifications.

**Setup Steps:**

1. **Create Telegram Bot:**
   - Open Telegram and search for @BotFather
   - Send `/newbot` command
   - Follow instructions to create your bot
   - Save the bot token

2. **Get Your Chat ID:**
   - Search for @userinfobot on Telegram
   - Send `/start` command
   - Copy your chat ID

3. **Update .env file:**
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

## 🧪 Testing Your Setup

### 1. Check Configuration
```bash
python main.py config
```
Should show all your settings and API status.

### 2. Test API Connections
```bash
python main.py test
```
Tests Alpha Vantage, Google Sheets, and Telegram connections.

### 3. Run Backtest
```bash
python main.py backtest
```
Executes a 6-month strategy backtest.

### 4. Train ML Model
```bash
python main.py train-ml
```
Trains the machine learning prediction model.

### 5. Market Scan
```bash
python main.py scan
```
Performs a single market scan for trading signals.

## 🎯 Trading Configuration

Customize trading parameters in your `.env` file:

```env
# Capital and Risk Management
INITIAL_CAPITAL=100000          # Starting capital (₹1,00,000)
RISK_PER_TRADE=0.02            # Risk 2% per trade
MAX_POSITIONS=5                # Maximum 5 concurrent positions

# Technical Indicators (in config.py)
RSI_PERIOD=14                  # 14-day RSI
RSI_OVERSOLD=30               # Buy when RSI < 30
RSI_OVERBOUGHT=70             # Sell when RSI > 70
SHORT_MA=20                   # 20-day moving average
LONG_MA=50                    # 50-day moving average
```

## 📊 Understanding the Output

### Backtest Results
- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### ML Model Metrics
- **Test Accuracy**: Model's prediction accuracy on unseen data
- **Cross-Validation**: Average accuracy across multiple data splits
- **Features**: Number of technical indicators used

### Market Scan Output
- **Signal Strength**: WEAK/MODERATE/STRONG signals
- **Price & RSI**: Current stock price and RSI value
- **Action**: BUY/SELL/HOLD recommendations

## 🔧 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Alpha Vantage free tier: 5 requests/minute
   - System implements automatic rate limiting
   - Cached data is used when available

2. **Google Sheets Not Working**
   - Verify service account has edit access to sheet
   - Check credentials file path
   - Ensure Google Sheets API is enabled

3. **Telegram Alerts Not Sending**
   - Verify bot token and chat ID
   - Check if bot is blocked/muted
   - Test with `/start` command to your bot

4. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt` again
   - Check Python version (3.8+ required)

## 🚨 Important Notes

### Risk Disclaimer
- This is a prototype for educational purposes
- Not financial advice - use at your own risk
- Always test thoroughly before live trading
- Past performance doesn't guarantee future results

### Data Accuracy
- Market data may have delays
- Technical indicators are calculated from available data
- ML predictions are probabilistic, not guaranteed

### Production Considerations
- Consider Alpha Vantage premium for higher API limits
- Implement proper error handling and monitoring
- Use secure credential storage
- Test extensively before live deployment

## 📈 Next Steps

1. **Customize Strategy**: Modify parameters in `src/config.py`
2. **Add Indicators**: Extend `technical_indicators.py`
3. **Improve ML Model**: Experiment with different algorithms
4. **Add More Stocks**: Extend the NIFTY_50_STOCKS list
5. **Live Trading**: Integrate with broker APIs for actual trading

## 📞 Support

For issues or questions:
1. Check this guide
2. Review the main README.md
3. Check log files in `logs/` directory
4. Verify your configuration with `python main.py config`

---

**Happy Trading! 🚀📈**

*Remember: This is educational software. Always conduct thorough testing and risk assessment before any live trading.*