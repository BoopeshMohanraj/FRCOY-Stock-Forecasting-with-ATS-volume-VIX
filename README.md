# FRCOY-Stock-Forecasting-with-ATS-volume-VIX

# Overview

This project aims to predict short-term stock price movements for Fast Retailing ADR (FRCOY) using Alternative Trading System (ATS) volume, VIX (Volatility Index), and Machine Learning (ML) models. By analyzing institutional trading activity, market volatility, and technical indicators, this model generates actionable trading signals for better decision-making.

# Libraries & Packages Used

Python Libraries:

Data Processing: pandas, numpy

Machine Learning: scikit-learn, xgboost, tensorflow, optuna

Time Series Analysis: fbprophet, statsmodels

Visualization: matplotlib, seaborn, plotly

Portfolio Analysis: pyfolio, backtrader

Financial Data Retrieval: yfinance, alpaca-trade-api, FINRA API

# Goal

1. Detect Institutional Activity: ATS volume spikes indicate institutional buying/selling.

2. Improve Trade Timing: Combining ATS volume with VIX enhances market entry/exit points.

3. Optimize Portfolio Allocation: Machine Learning models are used for risk-adjusted portfolio optimization.

4. Provide Actionable Insights: Generate BUY/SELL signals before market participants react.
   
# Introduction
Institutional investors often use ATS (dark pools) to execute large trades without impacting public markets. Traditional models rely on historical prices, while this approach incorporates ATS volume, VIX, and Machine Learning to enhance predictive accuracy.

Unlike fundamental analysis, this project applies technical portfolio optimization to maximize returns for a given level of risk using quantitative trading techniques.

# Methodology

1️⃣ Data Collection

ATS Data: Retrieved from FINRA API.

Stock Price Data (OHLCV & VIX): Collected from Yahoo Finance.

Volatility Index (VIX): Used as a risk metric.


2️⃣ Data Preprocessing & Feature Engineering

Dimensionality Reduction: PCA (Principal Component Analysis) was applied to optimize feature selection.

Target Variable: Predicting UP (1) or DOWN (0) stock movement.

Time-Series Feature Engineering: Lag-based predictors and rolling statistics were extracted.


3️⃣ Model Training & Evaluation

We implemented and compared three ML models:

LSTM (Long Short-Term Memory): Captures sequential stock price trends.

XGBoost: Tree-based model for tabular stock data.

Reinforcement Learning (Q-Learning): Trained an agent to optimize portfolio allocation dynamically.


4️⃣ Hyperparameter Tuning

Used Optuna for automated hyperparameter tuning.

Improved model accuracy and risk-adjusted returns.


5️⃣ Forecasting & Backtesting

Time Series Forecasting: Implemented using Facebook Prophet.

Sharpe Ratio Analysis: Evaluated risk-adjusted performance.

Trading Strategy Backtesting: Conducted using pyfolio and backtrader.


# Results
Model Performance Overview:
The model achieved a real-time accuracy of 64%, demonstrating its ability to capture short-term stock movements.

![image](https://github.com/user-attachments/assets/252269dd-6ad7-48e2-8925-4c54ca21c9af)

1. LSTM demonstrated better generalization ability, but XGBoost performed well on recall.

2. Reinforcement Learning showed a balance between precision and recall, offering robust predictions.

3. ATS volume spikes provided strong signals for institutional trading influence.

4. VIX volatility significantly influenced stock movement predictions, enhancing trade execution timing.

5. Backtesting with Sharpe Ratio analysis confirmed the model’s ability to improve risk-adjusted returns.

![image](https://github.com/user-attachments/assets/a93dfa79-eb80-4422-adac-691e3d1833ac)

# Observations from Forecasting Graph

1. Historical price trends show volatility peaks in early 2021, followed by a period of relative stabilization.

2. The forecasted prices (red line) closely follow historical trends initially, validating the model's reliability in short-term predictions.

3. The confidence interval (shaded region) widens significantly in later years, reflecting increased uncertainty in long-term predictions.

4. Stock price movement remains relatively stable in the projected period, with minor fluctuations indicating expected market trends.

5. Potential breakouts are visible when forecasted prices diverge slightly from historical trends, which may signal institutional movement or macroeconomic influences.

# Value & Business Impact

1. This project demonstrates how ATS, VIX, and machine learning can be leveraged for financial decision-making:

2. Data-Driven Investment Strategies: Helps traders understand how institutional trading affects stock trends.

Enhanced Risk Management: Uses volatility signals (VIX) and ATS flows to anticipate market fluctuations.

3. AI-Powered Trading Models: Machine learning optimizes trade timing and improves decision-making.

4. Cross-Asset Class Expansion: This method can be applied to stocks, ETFs, and other financial instruments.

# Potential Use Cases

1. Institutional Investors: Detecting hidden liquidity and market shifts.

2. Retail Traders: Leveraging ATS & VIX-based signals for better trade execution.

3. Risk Management Teams: Measuring volatility effects on portfolio performance.

4. Algorithmic Trading Firms: Using machine learning-driven trade execution models.

# Next Steps

1. Enhance Sentiment Analysis: Integrate news & social media sentiment data.
   
2. Advanced Hyperparameter Tuning: Expand Optuna & RayTune search space.
   
3. Factor-Based Portfolio Optimization: Incorporate momentum & value factors.
   
4. Deploy as a Real-Time Trading Bot via QuantConnect or Interactive Brokers API.
