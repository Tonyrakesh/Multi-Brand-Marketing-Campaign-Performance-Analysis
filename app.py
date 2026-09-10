from datetime import date
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. APP CONFIGURATION

st.set_page_config(
    page_title='Campaign Performance & ROI Predictor',
    layout='wide',
)

st.title('Multi-Brand Marketing Campaign Predictor')
st.markdown(
    'Estimate projected **Revenue** and **Profit/Loss** status:'
)

# 2. LOAD TRAINED MODELS & COLUMN METADATA

@st.cache_resource
def load_artifacts():
  reg = joblib.load('reg_model.joblib')
  cls = joblib.load('cls_model.joblib')
  reg_cols = joblib.load('reg_columns.joblib')
  cls_cols = joblib.load('cls_columns.joblib')
  return reg, cls, reg_cols, cls_cols

try:
  reg_model, cls_model, reg_columns, cls_columns = load_artifacts()
except Exception as e:
  st.error(
      '⚠️ Model files not found! Please run `train_and_save_models.py` first.'
  )
  st.stop()

# 3. INPUT FORM (SIDEBAR & MAIN DASHBOARD)

with st.sidebar:
  st.header('🏷️ Campaign Descriptors')
  brand = st.selectbox('Select Brand', ['Nykaa', 'Purplle', 'Tira'])
  campaign_type = st.selectbox(
      'Campaign Type', ['Social Media', 'Paid Ads', 'Influencer', 'Email', 'SEO']
  )
  target_audience = st.selectbox(
      'Target Audience',
      [
          'College Students',
          'Tier 2 City Customers',
          'Youth',
          'Working Women',
          'Premium Shoppers',
      ],
  )
  customer_segment = st.selectbox(
      'Customer Segment',
      [
          'College Students',
          'Tier 2 City Customers',
          'Premium Shoppers',
          'Youth',
          'Working Women',
      ],
  )
  language = st.selectbox(
      'Primary Language', ['Tamil', 'Hindi', 'English', 'Bengali']
  )
  campaign_date = st.date_input('Campaign Launch Date', value=date.today())

  st.header('Channels Utilized')
  col_ch1, col_ch2 = st.columns(2)
  with col_ch1:
    ch_email = st.checkbox('Email', value=True)
    ch_fb = st.checkbox('Facebook', value=True)
    ch_google = st.checkbox('Google Ads', value=False)
  with col_ch2:
    ch_insta = st.checkbox('Instagram', value=True)
    ch_whatsapp = st.checkbox('WhatsApp', value=False)
    ch_yt = st.checkbox('YouTube', value=True)

st.subheader('Operational Metrics')
c1, c2, c3, c4 = st.columns(4)
with c1:
  duration = st.number_input(
      'Duration (Days)', min_value=1, max_value=180, value=21, step=1
  )
  impressions = st.number_input(
      'Total Impressions',
      min_value=1000,
      max_value=10000000,
      value=58000,
      step=1000,
  )
with c2:
  clicks = st.number_input(
      'Total Clicks', min_value=10, max_value=1000000, value=6150, step=50
  )
  leads = st.number_input(
      'Generated Leads', min_value=5, max_value=500000, value=3600, step=25
  )
with c3:
  conversions = st.number_input(
      'Conversions (Sales)', min_value=1, max_value=200000, value=2350, step=10
  )
  acquisition_cost = st.number_input(
      'Acquisition Cost (₹)',
      min_value=1.0,
      max_value=5000.0,
      value=207.2,
      step=5.0,
  )
with c4:
  engagement_score = st.slider(
      'Engagement Score (0 - 50)',
      min_value=0.0,
      max_value=50.0,
      value=21.0,
      step=0.01,
  )
  est_roi_input = st.number_input(
      'Estimated / Historical ROI Multiplier',
      min_value=-1.0,
      max_value=20.0,
      value=2.83,
      step=0.1,
  )

# 4. PREDICTION LOGIC & INFERENCE ENGINE

if st.button('Predict Campaign Performance', type='primary', use_container_width=True):
  # Computed Spend
  total_spend = conversions * acquisition_cost

  # Build Raw Record Dictionary
  input_record = {
      'Duration': duration,
      'Impressions': impressions,
      'Clicks': clicks,
      'Leads': leads,
      'Conversions': conversions,
      'Acquisition_Cost': acquisition_cost,
      'ROI': est_roi_input,
      'Engagement_Score': engagement_score,
      'Profit_Loss_Flag': 1 if est_roi_input > 0 else 0,
      'Year': campaign_date.year,
      'Month': campaign_date.month,
      'Day': campaign_date.day,
      'DayOfWeek': campaign_date.weekday(),
      'Channel_Email': int(ch_email),
      'Channel_Facebook': int(ch_fb),
      'Channel_Google': int(ch_google),
      'Channel_Instagram': int(ch_insta),
      'Channel_WhatsApp': int(ch_whatsapp),
      'Brand': brand,
      'Campaign_Type': campaign_type,
      'Target_Audience': target_audience,
      'Language': language,
      'Customer_Segment': customer_segment,
  }

  input_df = pd.DataFrame([input_record])

  # One-hot encode matching training schema
  encoded_df = pd.get_dummies(input_df, drop_first=True, dtype=int)

  # Align with Regression Columns
  X_reg_live = pd.DataFrame(0, index=[0], columns=reg_columns)
  for col in reg_columns:
    if col in encoded_df.columns:
      X_reg_live[col] = encoded_df[col].values[0]

  # 1. Predict Revenue (Regression)
  pred_revenue = float(reg_model.predict(X_reg_live)[0])
  pred_profit = pred_revenue - total_spend
  computed_roi = (
      (pred_profit / total_spend) if total_spend > 0 else 0
  )

  # 2. Predict Profit/Loss (Classification)
  X_cls_live = pd.DataFrame(0, index=[0], columns=cls_columns)
  encoded_df['Revenue'] = pred_revenue
  for col in cls_columns:
    if col in encoded_df.columns:
      X_cls_live[col] = encoded_df[col].values[0]

  pred_cls_flag = int(cls_model.predict(X_cls_live)[0])
  pred_cls_prob = float(cls_model.predict_proba(X_cls_live)[0][1])

  # 5. VISUALIZATION OF RESULTS & KEY PERFORMANCE INDICATORS
  
  st.divider()
  st.subheader('📊 Predictions & Financial Health')

  m1, m2, m3, m4 = st.columns(4)
  m1.metric(
      'Predicted Revenue',
      f'₹{pred_revenue:,.2f}',
      delta=f'₹{pred_profit:,.2f} Net',
  )
  m2.metric('Total Campaign Spend', f'₹{total_spend:,.2f}')
  m3.metric('Projected ROI', f'{computed_roi:.2f}x')

  if pred_cls_flag == 1:
    m4.success(f'✅ **PROFIT** ({pred_cls_prob * 100:.1f}%)')
  else:
    m4.error(f'⚠️ **LOSS** ({(1 - pred_cls_prob) * 100:.1f}%)')

  st.divider()
  # Comparative Visualizations
  st.subheader('📈 Financial Breakdown & Conversion Funnel')
  col_left, col_right = st.columns(2)

  with col_left:
    # Waterfall / Bar of Spend vs Revenue vs Profit
    fig_fin = go.Figure(
        data=[
            go.Bar(
                name='Financials (₹)',
                x=['Total Spend (Cost)', 'Predicted Revenue', 'Net Profit'],
                y=[total_spend, pred_revenue, pred_profit],
                marker_color=['#EF553B','#636EFA','#00CC96' 
                              if pred_profit > 0 else '#AB63FA',
                ],
                text=[
                    f'₹{total_spend:,.0f}',
                    f'₹{pred_revenue:,.0f}',
                    f'₹{pred_profit:,.0f}',
                ],
                textposition='auto',
            )
        ]
    )
    fig_fin.update_layout(
        title='Spend vs. Projected Revenue Comparison',
        yaxis_title='Amount (₹)',
        template='plotly_white',
    )
    st.plotly_chart(fig_fin, use_container_width=True)

  with col_right:
    # Conversion Funnel Diagram
    fig_funnel = go.Figure(
        go.Funnel(
            y=['Impressions', 'Clicks', 'Leads', 'Conversions'],
            x=[impressions, clicks, leads, conversions],
            textinfo='value+percent initial',
            marker={
                'color': ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c']
            },
        )
    )
    fig_funnel.update_layout(
        title='Audience Conversion Funnel', template='plotly_white'
    )
    st.plotly_chart(fig_funnel, use_container_width=True)