import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- 1. KONFIGURASI APLIKASI DAN ASET ---
st.set_page_config(
    page_title="Cryptocurrency Price Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling modern
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1a202c;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 2rem 0;
        color: white;
    }
    
    .select-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .slider-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        margin-top: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .generate-btn {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-top: 2.5rem;
    }

    
    .stButton > button {
        background: linear-gradient(45deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.3);
    }
    
    .info-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #667eea;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Kamus (Dictionary) untuk menyimpan konfigurasi setiap aset
ASSETS = {
    "BTC-USD": {
        "name": "Bitcoin",
        "ticker": "BTC-USD",
        "model_file": "models/btc/btc_model_2a.h5",
        "scaler_file": "models/btc/btc_scaler_2a.pkl",
        "window_file": "models/btc/btc_last_window_2a.npy",
        "color": "#f7931a"
    },
    "ETH-USD": {
        "name": "Ethereum",
        "ticker": "ETH-USD",
        "model_file": "models/eth/eth_model_2c.h5",
        "scaler_file": "models/eth/eth_scaler_2c.pkl",
        "window_file": "models/eth/eth_last_window_2c.npy",
        "color": "#627eea"
    }
}

# --- 2. FUNGSI-FUNGSI UTAMA ---

@st.cache_resource
def load_artifacts(asset_ticker):
    """Memuat model, scaler, dan data window berdasarkan aset yang dipilih."""
    try:
        config = ASSETS[asset_ticker]
        model = tf.keras.models.load_model(config["model_file"], compile=False)
        scaler = joblib.load(config["scaler_file"])
        last_window = np.load(config["window_file"])
        return model, scaler, last_window
    except Exception as e:
        st.error(f"❌ Error saat memuat artefak untuk {config['name']}: {e}")
        st.info("📁 Pastikan file model (.h5), scaler (.pkl), dan window (.npy) tersedia di direktori yang sama.")
        return None, None, None

@st.cache_data(ttl=1800)  # Cache untuk 30 menit
def get_live_data(asset_ticker):
    """Mengambil data harga berdasarkan ticker aset yang dipilih."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        data = yf.download(asset_ticker, start=start_date, end=end_date, interval='1h', progress=False)
        return data
    except Exception as e:
        st.error(f"❌ Error mengambil data untuk {asset_ticker}: {e}")
        return pd.DataFrame()

def predict_future(model, scaler, initial_window, hours_to_predict):
    """Fungsi prediksi masa depan."""
    current_window = initial_window.copy().reshape(1, initial_window.shape[0], 1)
    future_predictions_scaled = []
    
    for _ in range(hours_to_predict):
        next_pred_scaled = model.predict(current_window, verbose=0)
        future_predictions_scaled.append(next_pred_scaled[0, 0])
        new_window_entry = next_pred_scaled.reshape(1, 1, 1)
        current_window = np.append(current_window[:, 1:, :], new_window_entry, axis=1)
    
    future_predictions_actual = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    return future_predictions_actual

def create_enhanced_plot(x_data, y_data, title, color, ylabel="Price (USD)"):
    """Membuat plot yang enhanced dengan styling yang lebih baik."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    ax.plot(x_data, y_data, color=color, linewidth=2.5, alpha=0.9)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#2d3748')
    ax.set_ylabel(ylabel, fontsize=12, color='#4a5568')
    ax.set_xlabel('Time', fontsize=12, color='#4a5568')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# --- 3. TAMPILAN UTAMA APLIKASI ---

# Header utama
st.markdown('<h1 class="main-header">🚀 Cryptocurrency Price Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hybrid BiLSTM-GRU Neural Network Model</p>', unsafe_allow_html=True)

# Informasi penting tentang delay data
st.markdown("""
<div class="info-card">
    <strong>📊 Data Information:</strong><br>
    • Data diambil dari Yahoo Finance dengan interval 1 jam<br>
    • Terdapat delay sekitar 9 jam dari waktu real-time<br>
    • Data historis menampilkan 90 hari terakhir untuk analisis tren<br>
    • Model menggunakan arsitektur Hybrid BiLSTM-GRU untuk prediksi yang akurat
</div>
""", unsafe_allow_html=True)

# Container untuk form prediksi
with st.container():    
    # Dropdown untuk pilihan cryptocurrency
    st.markdown('<p class="select-label">📈 Select Cryptocurrency</p>', unsafe_allow_html=True)
    
    # Membuat options yang menampilkan nama lengkap tapi menggunakan ticker sebagai value
    crypto_options = {f"{ASSETS[ticker]['name']} ({ticker})": ticker for ticker in ASSETS.keys()}
    
    selected_crypto_display = st.selectbox(
        "",
        options=list(crypto_options.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    selected_crypto_ticker = crypto_options[selected_crypto_display]
    selected_crypto_name = ASSETS[selected_crypto_ticker]['name']
    selected_crypto_color = ASSETS[selected_crypto_ticker]['color']
    
    # Slider untuk jam prediksi
    st.markdown('<p class="slider-label">⏰ Prediction Hours</p>', unsafe_allow_html=True)
    
    hours_to_predict = st.slider(
        "",
        min_value=1,
        max_value=168,
        value=24,
        step=1,
        label_visibility="collapsed"
    )
    
    # Tampilkan informasi slider
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown('<p style="color: white; font-size: 0.9rem;">1 hour</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p style="color: white; text-align: center; font-weight: 600;">Selected: {hours_to_predict} hours ({hours_to_predict/24:.1f} days)</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p style="color: white; text-align: right; font-size: 0.9rem;">168 hours (7 days)</p>', unsafe_allow_html=True)
    
    # Tombol Generate Model
    # st.markdown('<div class="generate-btn">', unsafe_allow_html=True)
    # generate_button = st.button("GENERATE PREDICTION", type="primary")
    # st.markdown('</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button("GENERATE PREDICTION")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. LOGIKA PREDIKSI DAN HASIL ---

if generate_button:
    # Memuat artefak berdasarkan aset yang dipilih
    model, scaler, last_window = load_artifacts(selected_crypto_ticker)
    
    if model is not None and scaler is not None and last_window is not None:
        with st.spinner(f"🤖 Generating prediction for {selected_crypto_name}... Please wait."):
            # Mengambil data live
            live_data = get_live_data(selected_crypto_ticker)
            
            if not live_data.empty:
                # Melakukan prediksi
                future_preds = predict_future(model, scaler, last_window, hours_to_predict)
                last_timestamp = live_data.index[-1]
                future_timeline = pd.date_range(
                    start=last_timestamp + timedelta(hours=1), 
                    periods=hours_to_predict, 
                    freq='h'
                )
                
                # Menyimpan hasil prediksi
                results_df = pd.DataFrame({
                    'Waktu Prediksi': future_timeline,
                    'Prediksi Harga (USD)': future_preds.flatten()
                })
                
                # Simpan ke session state
                st.session_state['prediction_results'] = results_df
                st.session_state['live_data_at_prediction_time'] = live_data
                st.session_state['predicted_crypto_name'] = selected_crypto_name
                st.session_state['predicted_crypto_ticker'] = selected_crypto_ticker
                st.session_state['predicted_crypto_color'] = selected_crypto_color
                
                st.success(f"✅ Prediction completed successfully for {selected_crypto_name}!")
            else:
                st.error(f"❌ Failed to fetch data for {selected_crypto_ticker}. Please try again later.")
    else:
        st.error(f"❌ Failed to load model artifacts for {selected_crypto_name}. Please check if all required files are available.")

# Tampilkan hasil prediksi jika ada
if 'prediction_results' in st.session_state:
    results_df = st.session_state['prediction_results']
    live_data_for_plot = st.session_state['live_data_at_prediction_time']
    predicted_crypto_name = st.session_state['predicted_crypto_name']
    predicted_crypto_ticker = st.session_state['predicted_crypto_ticker']
    predicted_crypto_color = st.session_state['predicted_crypto_color']
    
    # Tampilkan data historis terkini
    st.markdown("### 📉 Recent Historical Data")
    st.markdown(f"*Showing last 90 days of {predicted_crypto_name} price data (with ~9 hours delay from real-time)*")
    
    if not live_data_for_plot.empty:
        fig_hist = create_enhanced_plot(
            live_data_for_plot.index,
            live_data_for_plot['Close'].values, 
            f'{predicted_crypto_ticker} Historical Price - Last 90 Days',
            predicted_crypto_color
        )
        st.pyplot(fig_hist)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabel data terkini
        st.markdown("#### 📋 Recent Price Data (Last 10 Hours)")
        recent_data = live_data_for_plot[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        recent_data = recent_data.round(2)
        recent_data.index = recent_data.index.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_data, use_container_width=True)
    
    # Tampilkan hasil prediksi
    st.markdown("### 🔮 Prediction Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Predicted Prices Table")
        display_df = results_df.copy()
        display_df['Prediksi Harga (USD)'] = display_df['Prediksi Harga (USD)'].apply(lambda x: f"${x:.2f}")
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "Waktu Prediksi": st.column_config.DatetimeColumn(
                    "🕐 Prediction Time",
                    format="DD/MM/YY HH:mm"
                ),
                "Prediksi Harga (USD)": "💵 Predicted Price"
            }
        )
    
    with col2:
        st.markdown("#### 📈 Prediction Chart")
        fig_pred = create_enhanced_plot(
            results_df['Waktu Prediksi'],
            results_df['Prediksi Harga (USD)'].values,
            f'{predicted_crypto_name} Price Prediction - Next {len(results_df)} Hours',
            '#ff6b6b'
        )
        st.pyplot(fig_pred)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Grafik kombinasi
    st.markdown("### 🔗 Combined Analysis: Historical vs Predicted Prices")    
    fig_combined, ax_combined = plt.subplots(figsize=(16, 8))
    fig_combined.patch.set_facecolor('white')
    
    # Plot data historis (48 jam terakhir untuk clarity)
    recent_hours = min(48, len(live_data_for_plot))
    recent_data = live_data_for_plot.tail(recent_hours)
    
    ax_combined.plot(recent_data.index, recent_data['Close'], 
                    color=predicted_crypto_color, label='📊 Historical Price', 
                    linewidth=3, alpha=0.8)
    ax_combined.plot(results_df['Waktu Prediksi'], results_df['Prediksi Harga (USD)'], 
                    color='#ff6b6b', linestyle='--', marker='o', 
                    label='Predicted Price', linewidth=2.5, markersize=4, alpha=0.9)
    
    # Tambahkan garis vertikal untuk memisahkan historical dan prediction
    if len(recent_data) > 0:
        ax_combined.axvline(x=recent_data.index[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax_combined.text(recent_data.index[-1], ax_combined.get_ylim()[1]*0.95, 
                        'Current Time', rotation=90, ha='right', va='top', 
                        color='gray', fontsize=10, alpha=0.8)
    
    ax_combined.set_title(f'🔍 {predicted_crypto_name} - Historical vs Predicted Prices', 
                         fontsize=18, fontweight='bold', pad=20, color='#2d3748')
    ax_combined.set_xlabel('Time', fontsize=14, color='#4a5568')
    ax_combined.set_ylabel('Price (USD)', fontsize=14, color='#4a5568')
    ax_combined.legend(fontsize=12, loc='upper left')
    ax_combined.grid(True, alpha=0.3, linestyle='--')
    
    # Styling untuk axes
    ax_combined.spines['top'].set_visible(False)
    ax_combined.spines['right'].set_visible(False)
    ax_combined.spines['left'].set_color('#e2e8f0')
    ax_combined.spines['bottom'].set_color('#e2e8f0')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_combined)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tambahan informasi
    st.markdown("---")
    st.markdown("### ℹ️ Additional Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **📊 Model Information:**
        - Architecture: Hybrid BiLSTM-GRU
        - Prediction Range: {len(results_df)} hours
        - Data Source: Yahoo Finance
        - Update Frequency: Hourly
        """)
    
    with col2:
        st.warning(f"""
        **⚠️ Important Notes:**
        - Data has ~9 hours delay from real-time
        - Predictions are for educational purposes
        - Consider multiple factors before trading
        - Past performance doesn't guarantee future results
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>🚀 Cryptocurrency Price Prediction App</strong></p>
    <p>Powered by Hybrid BiLSTM-GRU Neural Networks | Data from Yahoo Finance</p>
    <p><em>⚠️ For educational and research purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)