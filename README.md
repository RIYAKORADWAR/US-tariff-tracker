# US Tariff Market Reaction Tracker

## Overview

The US Tariff Market Reaction Tracker is a data-driven analytics platform designed to analyze the impact of US tariff-related events on global equity markets. The project processes historical financial market data, identifies trends and anomalies, and provides interactive visualizations and predictive insights through a web-based dashboard.

The system combines data engineering, exploratory data analysis (EDA), machine learning, and visualization techniques to help users understand how tariff announcements influence market behavior across sectors and regions.

---

## Features

* Automated data pipeline for collecting and processing financial datasets
* Analysis of 10,000+ market data points across multiple global indices
* Event-based market impact tracking for 24+ tariff events
* Exploratory Data Analysis (EDA) for identifying trends and anomalies
* Interactive dashboard built with Streamlit
* Machine Learning integration using XGBoost
* Real-time insights and automated reporting modules
* Sector-level and region-wise market comparisons

---

## Tech Stack

### Programming & Data Processing

* Python
* Pandas
* NumPy
* SQL

### Machine Learning

* XGBoost
* Scikit-learn

### Visualization & Dashboard

* Streamlit
* Matplotlib
* Seaborn
* Plotly

### Database & Storage

* SQL Database

---

## Project Architecture

1. Data Collection

   * Financial market datasets collected from multiple sources
   * Tariff event datasets compiled and structured

2. Data Processing

   * Cleaning and preprocessing using Pandas and SQL
   * Missing value handling and transformation
   * Data validation and normalization

3. Exploratory Data Analysis

   * Trend analysis
   * Volatility analysis
   * Sector-wise comparisons
   * Event impact visualization

4. Machine Learning

   * Predictive modeling using XGBoost
   * Market movement forecasting
   * Pattern recognition

5. Dashboard & Reporting

   * Interactive Streamlit dashboard
   * Real-time charts and filters
   * Automated reporting modules

---

## Dataset Information

The project analyzes:

* 14 global equity indices
* 24 tariff-related events
* Historical market data from 2018–2025
* 10,000+ processed data points

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/us-tariff-market-reaction-tracker.git
cd us-tariff-market-reaction-tracker
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Start the Streamlit Application

```bash
streamlit run app.py
```

---

## Project Structure

```bash
US-Tariff-Market-Reaction-Tracker/
│
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks for analysis
├── models/                  # Trained ML models
├── dashboard/               # Streamlit dashboard files
├── scripts/                 # Data processing scripts
├── app.py                   # Main application file
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── assets/                  # Images and visual assets
```

---

## Key Insights

* Identified market volatility patterns during major tariff announcements
* Compared sector-level reactions across different global markets
* Analyzed short-term vs long-term market behavior
* Automated insight generation through dashboard modules

---

## Future Improvements

* Real-time API integration for live market updates
* Advanced forecasting models using deep learning
* Sentiment analysis on financial news articles
* Expanded global market coverage
* Enhanced dashboard analytics and filtering

---

## Skills Demonstrated

* SQL & Database Management
* Data Cleaning & Transformation
* ETL Pipeline Development
* Exploratory Data Analysis
* Machine Learning
* Dashboard Development
* Data Visualization
* Business Analytics

---

## Author

Riya Koradwar

* LinkedIn: [https://linkedin.com](https://linkedin.com)
* GitHub: [https://github.com](https://github.com)

---

## License

This project is for educational and portf
