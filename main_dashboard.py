import streamlit as st
import requests
import pandas as pd
import altair as alt
from time import sleep
import ast
import numpy as np
from scipy.stats import percentileofscore

# Initialize dataframe for storing values
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        'timestamp', 'bid', 'ask', 'spread_price',
        'spread_buy', 'spread_sell', 'difference'
    ])

# Widgets for parameters
st.sidebar.header("Parameters")
slug = st.sidebar.text_input("Polymarket Slug", "bitcoin-above-97000-on-february-14")
lower_strike = st.sidebar.number_input("Lower Strike (K0)", value=96000)
higher_strike = st.sidebar.number_input("Higher Strike (K1)", value=97000)
btc_symbol = st.sidebar.text_input("Symbol", "BTC")
option_date = st.sidebar.text_input("Option Date", "250214")
smoothing_window = st.sidebar.number_input("Smoothing Window (data points)", min_value=1, value=30, step=5)

# Function to fetch prices
def get_prices():
    # Get Polymarket data
    poly_url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
    market = requests.get(poly_url).json()

    url = "https://clob.polymarket.com/book"
    response = requests.get(url, params={'token_id': ast.literal_eval(market['markets'][0]['clobTokenIds'])[0]})
    bestBid = float(response.json()['bids'][-1]['price'])
    bestAsk = float(response.json()['asks'][-1]['price'])

    # Get Binance spread price
    binance_url = "https://eapi.binance.com/eapi/v1/mark"
    lower_option = requests.get(binance_url, params={
        'symbol': f'{btc_symbol}-{option_date}-{lower_strike}-C'
    }).json()
    print(lower_option)
    higher_option = requests.get(binance_url, params={
        'symbol': f'{btc_symbol}-{option_date}-{higher_strike}-C'
    }).json()
    spread_price = (float(lower_option[0]['markPrice']) -
                    float(higher_option[0]['markPrice'])) / (higher_strike - lower_strike)

    # Binance spread bid/ask prices
    binance_depth_url = "https://eapi.binance.com/eapi/v1/depth"
    lower_strike_option_bid = float(
        requests.get(binance_depth_url, params={'symbol': f'{btc_symbol}-{option_date}-{lower_strike}-C'}).json()['bids'][0][0])
    lower_strike_option_ask = float(
        requests.get(binance_depth_url, params={'symbol': f'{btc_symbol}-{option_date}-{lower_strike}-C'}).json()['asks'][0][0])
    # Note: Adjusting the symbol for higher strike options if needed.
    higher_strike_option_bid = float(
        requests.get(binance_depth_url, params={'symbol': f'{btc_symbol}-{option_date}-{higher_strike}-C'}).json()['bids'][0][0])
    higher_strike_option_ask = float(
        requests.get(binance_depth_url, params={'symbol': f'{btc_symbol}-{option_date}-{higher_strike}-C'}).json()['asks'][0][0])

    spread_buy = (lower_strike_option_ask - higher_strike_option_bid) / (higher_strike - lower_strike)
    spread_sell = (lower_strike_option_bid - higher_strike_option_ask) / (higher_strike - lower_strike)
    return bestBid, bestAsk, spread_price, spread_buy, spread_sell

# Create chart placeholders
st.title("Pricing Polymarket Crypto Price Markets with Vertical Spreads")
st.markdown("""
Thesis: You can mimic the payoff of an European digital option with a Binance vertical spread, with the higher spread
being the strike price of the digital option. Theoretically, if the strike prices of the Binance options are
infinitely close, it will become a digital option. In practice, this is not the case, as the strike prices
of the Binance options are different, and they have a slightly different expiry time in the same day than 
the polymarket digital option. However, both structures should have a highly correlated price, and Binance options
have leading information compared to polymarket, which is relatively less reactive and liquid.
""")
st.markdown(r"""
### Digital Call Payoff
$$
\text{Payoff}_{\text{digital}} = 
\begin{cases} 
1 & \text{if } S_T \geq K \\
0 & \text{otherwise}
\end{cases}
$$

### Vertical Spread Replication
$$
\text{Payoff}_{\text{spread}} = \frac{1}{\Delta} \left[ \max(S_T - K, 0) - \max(S_T - (K + \Delta), 0) \right]
$$

**Theoretical limit as $\Delta \to 0$:**
$$
\text{Digital Call} \approx \lim_{\Delta \to 0} \frac{\text{Call}(K) - \text{Call}(K+\Delta)}{\Delta}
$$
""")
st.latex(r"""
P(S_T > K) \approx \frac{P_{\text{vertical spread}}}{\Delta}
""")
st.divider()

# Chart placeholders
chart1 = st.empty()
st.divider()
st.markdown("Bid asks of polymarket (blue) vs binance (red)")
chart2 = st.empty()
data_placeholder = st.empty()

while True:
    # Fetch prices
    bid, ask, spread_price, spread_buy, spread_sell = get_prices()
    if bid is not None and ask is not None and spread_price is not None:
        # Update dataframe with the new row
        new_row = pd.DataFrame([{
            'timestamp': pd.Timestamp.now(),
            'bid': bid,
            'ask': ask,
            'spread_price': spread_price,
            'spread_buy': spread_buy,
            'spread_sell': spread_sell,
            'difference': (bid + ask) / 2 - spread_price
        }])
        st.session_state.data = pd.concat([st.session_state.data, new_row]).tail(3*60*60)  # keep data for the last 3 hours

        # Calculate rolling statistics for the spread price if needed (for the first chart)
        window = smoothing_window
        st.session_state.data['smoothed_spread'] = st.session_state.data['spread_price'].rolling(window=window).mean()
        st.session_state.data['percentile'] = st.session_state.data['smoothed_spread'].rolling(window=window).rank(pct=True)

        # --- Chart 1: Price and Spread Over Time with Smoothing ---
        base = alt.Chart(st.session_state.data).encode(x='timestamp:T')

        line_bid = base.mark_line(color='blue').encode(
            y='bid:Q',
            tooltip=['bid']
        )

        line_ask = base.mark_line(color='blue', strokeDash=[5,1]).encode(
            y='ask:Q',
            tooltip=['ask']
        )

        line_spread_raw = base.mark_line(color='red').encode(
            y='spread_price:Q',
            tooltip=['spread_price']
        )

        # Smoothed spread price line created with a window transform for smoother visualization
        line_spread_smoothed = base.transform_window(
            rolling_mean='mean(spread_price)',
            frame=[-smoothing_window, 0]
        ).mark_line(color='green', strokeDash=[5,3]).encode(
            y='rolling_mean:Q',
            tooltip=[alt.Tooltip('rolling_mean:Q', title='smoothed_spread')]
        )

        chart_components = [line_bid, line_ask, line_spread_raw, line_spread_smoothed]

        # Check for threshold crossing and add a vertical rule if needed
        if not st.session_state.data.empty:
            latest_percentile = st.session_state.data['percentile'].iloc[-1]
            latest_timestamp = st.session_state.data['timestamp'].iloc[-1]
            if not pd.isna(latest_percentile):
                if latest_percentile < threshold or latest_percentile > (1 - threshold):
                    color_rule = 'red' if latest_percentile < threshold else 'green'
                    rule = alt.Chart(pd.DataFrame({'timestamp': [latest_timestamp]})).mark_rule(
                        color=color_rule,
                        strokeWidth=2
                    ).encode(x='timestamp:T')
                    chart_components.append(rule)

        chart1_alt = alt.layer(*chart_components).properties(
            title='Price and Spread Over Time (with Smoothing)'
        ).configure_legend(
            orient='right',
            titleFontSize=12,
            labelFontSize=10
        )
        chart1.altair_chart(chart1_alt, use_container_width=True)

        # --- Chart 2: Live Streaming of Polymarket and Binance Spread Bid/Ask ---
        base2 = alt.Chart(st.session_state.data).encode(x='timestamp:T')

        line_bid_stream = base2.mark_line(color='blue').encode(
            y=alt.Y('bid:Q', title='Polymarket Bid/Ask & Binance Spread Bid/Ask'),
            tooltip=[alt.Tooltip('bid:Q', title='Polymarket Bid')]
        )

        line_ask_stream = base2.mark_line(color='blue', strokeDash=[5,1]).encode(
            y=alt.Y('ask:Q', title='Polymarket Bid/Ask & Binance Spread Bid/Ask'),
            tooltip=[alt.Tooltip('ask:Q', title='Polymarket Ask')]
        )

        line_spread_buy = base2.mark_line(color='red').encode(
            y=alt.Y('spread_buy:Q', title='Polymarket Bid/Ask & Binance Spread Bid/Ask'),
            tooltip=[alt.Tooltip('spread_buy:Q', title='Binance Spread Buy')]
        )

        line_spread_sell = base2.mark_line(color='red', strokeDash=[5,1]).encode(
            y=alt.Y('spread_sell:Q', title='Polymarket Bid/Ask & Binance Spread Bid/Ask'),
            tooltip=[alt.Tooltip('spread_sell:Q', title='Binance Spread Sell')]
        )

        chart2_alt = alt.layer(line_bid_stream, line_ask_stream, line_spread_buy, line_spread_sell).properties(
        ).configure_legend(
            orient='right',
            titleFontSize=12,
            labelFontSize=10
        )
        chart2.altair_chart(chart2_alt, use_container_width=True)

        # --- Show raw data ---
        data_placeholder.dataframe(st.session_state.data.tail(10), use_container_width=True)

    sleep(5)  # Update every 5 seconds
