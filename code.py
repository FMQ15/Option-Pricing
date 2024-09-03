import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Black-Scholes formula
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

# Binomial model for European options
def binomial_tree(S, K, T, r, sigma, steps, option_type):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    option_values = np.zeros((steps + 1, steps + 1))
    
    # Calculate option value at each final node
    for i in range(steps + 1):
        ST = S * (u ** (steps - i)) * (d ** i)
        if option_type == 'call':
            option_values[i, steps] = max(0, ST - K)
        elif option_type == 'put':
            option_values[i, steps] = max(0, K - ST)
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    # Backward induction
    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            option_values[i, j] = np.exp(-r * dt) * (p * option_values[i, j + 1] + (1 - p) * option_values[i + 1, j + 1])

    return option_values[0, 0]

# Function to generate heatmap data
def generate_heatmap_data(func, K, T, r, sigma_range, S_range, steps, option_type, model, num_cells=10):
    spot_prices = np.linspace(S_range[0], S_range[1], num_cells)
    volatilities = np.linspace(sigma_range[0], sigma_range[1], num_cells)
    prices = np.zeros((len(spot_prices), len(volatilities)))

    for i, S in enumerate(spot_prices):
        for j, sigma in enumerate(volatilities):
            if model == "Black-Scholes":
                prices[i, j] = func(S, K, T, r, sigma, option_type)
            else:
                prices[i, j] = func(S, K, T, r, sigma, steps, option_type)

    return prices, spot_prices, volatilities

# Function to plot heatmap
def plot_heatmap(data, x_labels, y_labels, title, xlabel, ylabel, cmap='RdYlGn'):
    plt.figure(figsize=(14, 7))  # Larger figure size
    sns.heatmap(
        data,
        xticklabels=[f"{x * 100:.1f}%" for x in x_labels],  # Convert to percentage
        yticklabels=[f"${y:.1f}" for y in y_labels],        # Format y_labels with dollar sign and one decimal place
        annot=True,
        fmt=".2f",
        cbar=False,  # Hide the color bar
        linewidths=0.5,
        linecolor='gray',
        cmap=cmap
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()  # Ensure that everything fits nicely
    st.pyplot(plt)

# Function to calculate profitability
def calculate_profitability(option_price, K, spot_prices, volatilities, option_type, amount_paid, pricing_model):
    profitability = np.zeros((len(spot_prices), len(volatilities)))
    for i, S in enumerate(spot_prices):
        for j, sigma in enumerate(volatilities):
            # Compute option price for current spot price and volatility
            if option_type == 'call':
                price = black_scholes(S, K, T, r, sigma, 'call') if pricing_model == 'Black-Scholes' else binomial_tree(S, K, T, r, sigma, steps, 'call')
            elif option_type == 'put':
                price = black_scholes(S, K, T, r, sigma, 'put') if pricing_model == 'Black-Scholes' else binomial_tree(S, K, T, r, sigma, steps, 'put')
            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")
            profitability[i, j] = price - amount_paid
    return profitability

# Streamlit app
st.title("Option Pricing")

# Header section with author info and LinkedIn logo
st.markdown("""
    <div style="display: flex; align-items: center;">
        <h3>Created by Filip Matic</h3>
        <a href="https://www.linkedin.com/in/filip-matic15/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn Logo" style="height: 24px; margin-left: 10px;">
        </a>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for model selection and profitability check
if 'pricing_model' not in st.session_state:
    st.session_state.pricing_model = 'Black-Scholes'
if 'option_type_profit' not in st.session_state:
    st.session_state.option_type_profit = None
if 'amount_paid' not in st.session_state:
    st.session_state.amount_paid = None
if 'check_profitability_clicked' not in st.session_state:
    st.session_state.check_profitability_clicked = False

# Input parameters in sidebar
with st.sidebar:
    S = st.number_input("Spot Price", value=100.0, step=1.0)
    K = st.number_input("Strike Price", value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (years)", value=1.0, step=0.01)
    r = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.1) / 100
    sigma = st.number_input("Volatility (%)", value=20.0, step=0.1) / 100

    st.sidebar.header("Profitability Inputs")
    option_type_profit = st.sidebar.radio("Option Type for Profitability", ("call", "put"))
    amount_paid = st.sidebar.number_input("Amount Paid for Option", value=0.0, step=0.01)
    
    if st.sidebar.button("Check Profitability"):
        st.session_state.option_type_profit = option_type_profit
        st.session_state.amount_paid = amount_paid
        st.session_state.check_profitability_clicked = True
    else:
        st.session_state.check_profitability_clicked = False

# Option type and pricing model selection
st.header("Options")
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    option_type = st.radio("Option Type", ("European", "American"))
with col2:
    if option_type == "European":
        pricing_model = st.radio("Pricing Model", ("Black-Scholes", "Binomial Tree"))
    else:
        pricing_model = "Binomial Tree"
with col3:
    st.sidebar.write("The profitability section is updated automatically based on button click.")

# Handle American options with Black-Scholes
if option_type == "American" and pricing_model == "Black-Scholes":
    st.error("The Black-Scholes model does not support American options. Switching to Binomial Tree model.")
    st.session_state.pricing_model = "Binomial Tree"
    pricing_model = st.session_state.pricing_model

# Steps for binomial model
steps = st.number_input("Number of Steps in Binomial Tree", value=100, step=1) if pricing_model == "Binomial Tree" else None

# Calculate option prices
if option_type == "European":
    if pricing_model == "Black-Scholes":
        call_price = black_scholes(S, K, T, r, sigma, "call")
        put_price = black_scholes(S, K, T, r, sigma, "put")
    else:
        call_price = binomial_tree(S, K, T, r, sigma, steps, "call")
        put_price = binomial_tree(S, K, T, r, sigma, steps, "put")
else:
    if pricing_model == "Binomial Tree":
        call_price = binomial_tree(S, K, T, r, sigma, steps, "call")
        put_price = binomial_tree(S, K, T, r, sigma, steps, "put")
    else:
        call_price = put_price = None

# Display option prices
if call_price is not None and put_price is not None:
    st.header("Option Prices")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Call Option")
        st.markdown(f"<h2 style='color: #4CAF50;'>${call_price:.2f}</h2>", unsafe_allow_html=True)
    with col2:
        st.subheader("Put Option")
        st.markdown(f"<h2 style='color: #F44336;'>${put_price:.2f}</h2>", unsafe_allow_html=True)

    # Generate heatmap data
    st.header("Heatmaps")
    spot_price_range = (S * 0.8, S * 1.2)
    volatility_range = (sigma * 0.8, sigma * 1.2)
    if volatility_range[0] <= 0:
        volatility_range = (0.01, volatility_range[1])

    # Heatmap for call options
    heatmap_data_call, spot_prices, volatilities = generate_heatmap_data(
        binomial_tree if pricing_model == "Binomial Tree" else black_scholes,
        K, T, r, volatility_range, spot_price_range, steps, "call", pricing_model
    )
    
    # Heatmap for put options
    heatmap_data_put, spot_prices, volatilities = generate_heatmap_data(
        binomial_tree if pricing_model == "Binomial Tree" else black_scholes,
        K, T, r, volatility_range, spot_price_range, steps, "put", pricing_model
    )
    
    # Plot heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # Larger figure size and two subplots
    sns.heatmap(
        heatmap_data_call,
        xticklabels=[f"{x * 100:.1f}%" for x in volatilities],  # Convert to percentage
        yticklabels=[f"${y:.1f}" for y in spot_prices],        # Format y_labels with dollar sign and one decimal place
        annot=True,
        fmt=".2f",
        cbar=False,  # Hide the color bar
        linewidths=0.5,
        linecolor='gray',
        cmap='RdYlGn',
        ax=axes[0]
    )
    axes[0].set_xlabel("Volatility (%)")
    axes[0].set_ylabel("Spot Price")
    axes[0].set_title("Call Option Price Heatmap")

    sns.heatmap(
        heatmap_data_put,
        xticklabels=[f"{x * 100:.1f}%" for x in volatilities],  # Convert to percentage
        yticklabels=[f"${y:.1f}" for y in spot_prices],        # Format y_labels with dollar sign and one decimal place
        annot=True,
        fmt=".2f",
        cbar=False,  # Hide the color bar
        linewidths=0.5,
        linecolor='gray',
        cmap='RdYlGn',
        ax=axes[1]
    )
    axes[1].set_xlabel("Volatility (%)")
    axes[1].set_ylabel("Spot Price")
    axes[1].set_title("Put Option Price Heatmap")

    plt.tight_layout()  # Ensure that everything fits nicely
    st.pyplot(fig)

    # Check profitability
    if st.session_state.check_profitability_clicked:
        st.header("Profitability Heatmaps")
        
        # Profitability heatmap for call options
        if st.session_state.option_type_profit == 'call':
            profitability_data_call = calculate_profitability(call_price, K, spot_prices, volatilities, "call", st.session_state.amount_paid, pricing_model)
            plot_heatmap(
                profitability_data_call,
                volatilities,
                spot_prices,
                "Call Option Profitability Heatmap",
                "Volatility (%)",
                "Spot Price",
                cmap='RdYlGn'
            )
        
        # Profitability heatmap for put options
        if st.session_state.option_type_profit == 'put':
            profitability_data_put = calculate_profitability(put_price, K, spot_prices, volatilities, "put", st.session_state.amount_paid, pricing_model)
            plot_heatmap(
                profitability_data_put,
                volatilities,
                spot_prices,
                "Put Option Profitability Heatmap",
                "Volatility (%)",
                "Spot Price",
                cmap='RdYlGn'
            )
