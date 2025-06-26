# Behavioural Finance Simulations

<p align="center">
  <a href="https://behavioural-finance-simulator-xnxirrqvttgw7cicbyduzi.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" style="height:60px;">
  </a>
</p>

<h1 align="center" style="font-size:2.5em;">Behavioural Finance Simulator</h1>

<p align="center">
  <img src="streamlit_app/screenshot.png" alt="App Screenshot" width="800"/>
</p>

## 🚀 Overview

This project enables me and whoever is interested to:
- Simulate a variety of portfolio strategies (e.g., buy & hold, periodic rebalancing, recency bias, panic selling).
- Explore the impact of behavioural biases (like recency bias and panic selling) on long-term investment outcomes.
- Visualize and compare the performance of classic and behavioural strategies interactively.

All core logic is implemented in the `simulation_helpers` module, while the user-friendly interface is provided by a Streamlit app.

---

## 🏗️ Project Structure

```
behavioural_finance_simulations/           # Main Python package
    simulation_helpers.py                  # Core simulation functions
streamlit_app/
    streamlit_interactive.py               # Streamlit app UI
data/
    clean_data_SPY_AGG_IJS.csv             # Example data (S&P 500, Bonds, Small Cap Value)
notebooks/
    ...                                    # Notebooks for exploration (optional)
setup.py
requirements.txt
README.md
```

---

## 📊 Main Features

- **Asset Allocation:**  
  Choose portfolio weights for assets like S&P 500 (SPY), World Aggregate Bonds (AGG), and Small Cap Value (IJS).

- **Behavioural Strategies:**  
  - **Buy & Hold:** No rebalancing.
  - **Periodic Rebalancing:** Monthly, quarterly, yearly.
  - **Recency Bias:** Shift allocation toward recent winners using softmax over past returns (momentum effect).
  - **Panic Selling:** Exit after a large drawdown; optionally re-enter after recovery.

- **Interactive Visualization:**  
  - Compare strategies on cumulative wealth, normalized asset performance, and risk/return statistics.
  - Adjust behavioural parameters and see real-time changes.

- **Educational Tips:**  
  - Explanations for each asset and parameter.
  - Insights into behavioural finance effects.

---

## 🖥️ Quick Start

### 1. **Installation**

Clone the repo and install dependencies:

```bash
git clone https://github.com/adrianocosi0/behavioural_finance_simulations.git
cd behavioural_finance_simulations
pip install .
```

Or, for development:

```bash
pip install -e .
```

### 2. **Run the Streamlit App**

```bash
streamlit run streamlit_app/streamlit_interactive.py
```

This will launch your browser with the interactive simulator.

---

## 🧑‍💻 Usage

- **As a Streamlit app:**  
  Use the UI to set allocations, select strategies, and adjust behavioural parameters.  
  See live plots and summaries.

- **As a Python library:**  
  Import simulation logic for your own scripts or notebooks:

```python
from behavioural_finance_simulations import simulation_helpers as sh

# Example usage
result = sh.simulate_recency_bias(prices, weights, lookback_window=60, recency_strength=0.8, temperature=0.1)
```

---

## 📂 Data

- Example datasets are provided in the `data/` folder.
- Format: CSV with daily prices for SPY, AGG, and IJS.
- You can replace with your own data (format: columns=assets, index=dates).

---

## 📝 Notebooks

See the `notebooks/` folder for examples on using the simulation helpers in research and analysis.

---

## 🧠 Behavioural Finance Concepts

- **Recency Bias:** Overweighting assets that have recently performed well (momentum).
- **Panic Selling:** Selling after a large loss, then re-entering only after partial recovery.
- **Periodic Rebalancing:** Returning portfolio to target weights at regular intervals.
- **Buy & Hold:** No changes after initial allocation.

The simulator helps you understand how these behaviours affect long-term wealth and risk.

---

## 📈 Example Strategies

- **Recency bias (40–60 day window)**: Often outperforms static rebalancing due to momentum.
- **Panic selling:** It's bad!
- **Mix and match:** Experiment with all parameters in the app!

---

## 🛠️ Development

- All simulation logic: `behavioural_finance_simulations/simulation_helpers.py`
- Streamlit app UI: `streamlit_app/streamlit_interactive.py`
- To add new strategies, contribute to `simulation_helpers.py` and update the app accordingly.

---

## 📜 License

MIT License (see LICENSE file).

---

## 🙏 Acknowledgements

- Data: Yahoo Finance, publicly available indices.
- Inspired by classic and recent research in behavioural finance and asset allocation.

---

## 💡 Contributions

Pull requests, suggestions, and issues are welcome! Please use the GitHub Issues page.

---

## 📬 Contact

For questions or suggestions, open an issue or contact [Adriano Cosi](mailto:adrianocosi0@gmail.com).