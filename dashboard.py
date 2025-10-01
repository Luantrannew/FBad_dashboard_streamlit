import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Ad Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    .stMetric label {font-size: 14px !important; font-weight: 600 !important; color: #1f1f1f !important;}
    .stMetric [data-testid="stMetricValue"] {font-size: 28px !important; font-weight: 700 !important; color: #1f1f1f !important;}
    .stMetric [data-testid="stMetricDelta"] {color: #1f1f1f !important;}
    h1 {color: #2E86AB; padding-bottom: 10px; border-bottom: 3px solid #2E86AB;}
    h3 {color: #444; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'male': '#3498DB',
    'female': '#E91E63',
    'age_65': '#F1C40F',
    'age_55': '#27AE60',
    'age_45': '#5DADE2',
    'age_25': '#E67E22'
}

# Product pricing data
@st.cache_data
def load_product_pricing():
    pricing_data = {
        '3H': 34.95, '3T': 25.95, 'AAP': 29.95, 'ACA': 16.95, 'ACO': 17.95, 'ACO_': 15.95,
        'AKEY': 19.95, 'ALCA': 19.95, 'ALIBO': 34.95, 'AMUG': 24.95, 'APLA': 24.95,
        'AREEL': 24.95, 'ASHAKE': 24.95, 'ASUN': 34.95, 'BBAS': 24.95, 'BBAS_': 42.95,
        'BBB': 19.95, 'BBL': 36.95, 'BCB': 25.95, 'BETO': 34.95, 'BLK': 39.95,
        'BMUG': 26.95, 'BNIE': 21.95, 'BP': 55.95, 'BRG': 25.95, 'BROCK': 22.95,
        'BTL': 39.95, 'BTLWBXXX': 49.95, 'BUN': 32.95, 'CAHOC': 29.95, 'CAHOW': 34.95,
        'CAP': 29.95, 'CAUMUG': 34.95, 'CB': 44.95, 'CCAP': 29.95, 'CDOLL': 19.95,
        'CEP': 29.95, 'CEV': 32.95, 'CEW': 34.95, 'CGOR': 17.95, 'CH': 49.95,
        'CHE': 39.95, 'CMUG': 26.95, 'CPLA': 29.95, 'CSOCK': 18.95, 'CSUN': 22.95,
        'CTOT': 34.95, 'CUDOR': 27.95, 'DKB': 49.95, 'DRM': 29.95, 'DTNECK': 24.95,
        'DWO': 21.95, 'EBE': 32.95, 'ELEA': 34.95, 'ELK': 29.95, 'ENWA': 59.95,
        'EWG': 26.95, 'FLAMP': 39.95, 'FOFLA': 39.95, 'GAF': 29.95, 'GAFXXX': 47.95,
        'GLACU': 29.95, 'GLM': 29.95, 'GOLFB': 19.95, 'GOSICA': 22.95, 'HABES': 49.95,
        'HAP': 29.95, 'HB': 49.95, 'HBTAG': 19.95, 'HCE': 19.95, 'HEGOR': 22.95,
        'HESHA': 34.95, 'HGO': 19.95, 'HPOS': 27.95, 'HW': 29.95, 'HWII': 32.95,
        'JARL': 29.95, 'JRNL': 29.95, 'KAP': 24.95, 'KCH': 22.95, 'KCX': 21.95,
        'KHO': 29.95, 'KIBP': 39.95, 'KIBS': 29.95, 'KILUN': 29.95, 'LANT': 22.95,
        'LAPBAG': 29.95, 'LBK': 22.95, 'LECA': 22.95, 'LECOO': 22.95, 'LEJAR': 30.95,
        'LEWO': 39.95, 'LEREC': 39.95, 'LGT': 19.95, 'LTB': 32.95, 'LTRAY': 29.95,
        'MAGN': 19.95, 'MEBES': 29.95, 'MENB': 29.95, 'MG': 22.95, 'MG_other': 22.95,
        'MSIGN': 29.95, 'MW': 34.95, 'MZB': 29.95, 'MZBPRB': 29.95, 'MZX': 29.95,
        'ONES': 24.95, 'PAHA': 32.95, 'PECHA': 26.95, 'PENH': 24.95, 'PHC': 24.95,
        'PIMUG': 32.95, 'PLEV': 29.95, 'PLUBO': 26.95, 'PLW': 25.95, 'PMAM': 26.95,
        'PMGB': 29.95, 'PMGO': 29.95, 'PMGW': 29.95, 'PMGY': 29.95, 'POLOS': 32.95,
        'PPH': 24.95, 'PW': 34.95, 'RCE': 19.95, 'RCRF': 27.95, 'RECA': 34.95,
        'ROBAG': 36.95, 'ROCO': 22.95, 'RPLA': 33.95, 'RSUN': 22.95, 'RWG': 23.95,
        'SCRD': 26.95, 'SCRF': 29.95, 'SECA': 31.95, 'SHEEPA': 39.95, 'STIK': 17.95,
        'SUNVIS': 21.95, 'SWATCH': 29.95, 'TAG': 21.95, 'TB': 32.95, 'TBS': 44.95,
        'TECU': 34.95, 'TED': 36.95, 'TIE': 29.95, 'TMS': 24.95, 'TRIGRA': 29.95,
        'TWS': 23.95, 'VIS': 21.95, 'VPOS': 27.95, 'WATRA': 29.95, 'WBAS': 24.95,
        'WBAS_': 42.95, 'WCA': 17.95, 'WDCAP': 28.95, 'WDO': 18.95, 'WEBES': 29.95,
        'WHIBO': 39.95, 'WIGL': 27.95, 'WINDS': 29.95, 'WLEG': 29.95, 'WOBOE': 29.95,
        'WOBOP': 29.95, 'WOMUG': 34.95, 'WPLA': 24.95, 'WSIGN': 35.95, 'WSO': 19.95,
        'WW': 36.95, 'WYOU': 24.95
    }
    return pricing_data

# Load data function
@st.cache_data
def load_data():
    # Load Excel file
    file_path = r'C:\gems\data\preprocessed_data_01102025.xlsx'
    
    try:
        df = pd.read_excel(file_path)
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Amount Spent', 'Impressions', 'Unique outbound clicks', 
                       'Adds to cart', 'Checkouts initiated', 'Purchases', 
                       'CPM', 'Cost per Click', 'Cost per ATC', 'Frequency', 'Post comments']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Extract Ad Manager from Campaign name
        def extract_ad_manager(campaign_name):
            if pd.isna(campaign_name):
                return 'Other'
            
            campaign_str = str(campaign_name).upper()
            
            # Search for "AD-" pattern
            if 'AD-' in campaign_str:
                # Find the position after "AD-"
                start_idx = campaign_str.find('AD-') + 3
                
                # Find the next "," after "AD-" (priority)
                comma_idx = campaign_str.find(',', start_idx)
                # Find the next "*" after "AD-"
                star_idx = campaign_str.find('*', start_idx)
                
                # Determine end index: comma takes priority, then star
                if comma_idx != -1:
                    end_idx = comma_idx
                elif star_idx != -1:
                    end_idx = star_idx
                else:
                    end_idx = -1
                
                if end_idx != -1:
                    # Extract the name between "AD-" and the delimiter
                    ad_manager = campaign_str[start_idx:end_idx].strip()
                    return ad_manager if ad_manager else 'Other'
            
            return 'Other'
        
        df['Ad_Manager'] = df['Campaign name'].apply(extract_ad_manager)
        
        # Extract designID from Campaign name
        # designID = nicheID + số + productID (lấy từ nicheID đến productID)
        def extract_design_id(campaign_name, niche_id, product_id):
            if pd.isna(campaign_name) or pd.isna(niche_id) or pd.isna(product_id):
                return 'Unknown'
            
            campaign_str = str(campaign_name)
            niche_str = str(niche_id)
            product_str = str(product_id)
            
            # Pattern: NICHE + digits + _ + PRODUCT
            # Example: "WDO123_WDO" -> designID = "WDO123"
            import re
            pattern = f"{re.escape(niche_str)}\\d+(?=_{re.escape(product_str)})"
            match = re.search(pattern, campaign_str)
            
            if match:
                return match.group(0)
            
            return 'Unknown'
        
        df['designID'] = df.apply(
            lambda x: extract_design_id(x['Campaign name'], x['nicheID'], x['productID']), 
            axis=1
        )
        
        # Calculate metrics
        df['CTR'] = (df['Unique outbound clicks'] / df['Impressions'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        df['ATC_Rate'] = (df['Adds to cart'] / df['Unique outbound clicks'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        df['Checkout_Rate'] = (df['Checkouts initiated'] / df['Adds to cart'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        df['Purchase_Rate'] = (df['Purchases'] / df['Checkouts initiated'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        df['CVR'] = (df['Purchases'] / df['Unique outbound clicks'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        df['CPA'] = (df['Amount Spent'] / df['Purchases']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        
        # Load pricing and calculate revenue/profit
        pricing = load_product_pricing()
        df['Selling_Price'] = df['productID'].map(pricing).fillna(0)
        df['Revenue'] = (df['Purchases'] * df['Selling_Price']).round(2)
        df['Profit'] = (df['Revenue'] - df['Amount Spent']).round(2)
        df['Profit_Margin'] = ((df['Profit'] / df['Revenue']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        df['ROAS'] = (df['Revenue'] / df['Amount Spent']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        
        return df
        
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load data
df = load_data()

# Title
st.markdown("# 📊 Ad Performance Dashboard")
st.markdown(f"**Date Range:** Sep 19 - Sep 30, 2025")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("🎚️ Filters")

# Ad Manager filter (NEW)
ad_manager_options = ['All'] + sorted(df['Ad_Manager'].unique().tolist())
selected_ad_manager = st.sidebar.multiselect('👤 Ad Manager', ad_manager_options, default=['All'])

# Age filter
age_options = ['All'] + sorted(df['Age'].unique().tolist())
selected_age = st.sidebar.multiselect('👥 Age Group', age_options, default=['All'])

# Gender filter
gender_options = ['All', 'male', 'female']
selected_gender = st.sidebar.radio('⚥ Gender', gender_options)

# Niche filter
niche_options = ['All'] + sorted(df['nicheID'].unique().tolist())
selected_niche = st.sidebar.multiselect('🎯 Niche', niche_options, default=['All'])

# Product filter
product_options = ['All'] + sorted(df['productID'].unique().tolist())
selected_product = st.sidebar.multiselect('📦 Product', product_options, default=['All'])

# Apply filters
filtered_df = df.copy()

if 'All' not in selected_ad_manager and selected_ad_manager:
    filtered_df = filtered_df[filtered_df['Ad_Manager'].isin(selected_ad_manager)]

if 'All' not in selected_age and selected_age:
    filtered_df = filtered_df[filtered_df['Age'].isin(selected_age)]

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

if 'All' not in selected_niche and selected_niche:
    filtered_df = filtered_df[filtered_df['nicheID'].isin(selected_niche)]

if 'All' not in selected_product and selected_product:
    filtered_df = filtered_df[filtered_df['productID'].isin(selected_product)]

# KPI Cards
st.markdown("### 📈 Key Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_spend = filtered_df['Amount Spent'].sum()
    st.metric("💰 Total Spend", f"${total_spend:,.2f}")

with col2:
    total_purchases = filtered_df['Purchases'].sum()
    st.metric("🛒 Purchases", f"{int(total_purchases)}")

with col3:
    total_revenue = filtered_df['Revenue'].sum()
    st.metric("💵 Revenue", f"${total_revenue:,.2f}")

with col4:
    total_profit = filtered_df['Profit'].sum()
    profit_color = "normal" if total_profit >= 0 else "inverse"
    st.metric("📊 Profit", f"${total_profit:,.2f}", delta=None if total_profit >= 0 else f"Loss: ${abs(total_profit):,.2f}")

with col5:
    avg_roas = filtered_df['ROAS'].mean() if total_spend > 0 else 0
    roas_color = "normal" if avg_roas >= 2 else "inverse"
    st.metric("🎯 ROAS", f"{avg_roas:.2f}x")

with col6:
    avg_margin = filtered_df['Profit_Margin'].mean()
    st.metric("📈 Profit Margin", f"{avg_margin:.1f}%")

st.markdown("---")

# Add a second row of detailed metrics
st.markdown("### 💡 Performance Insights")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_impressions = filtered_df['Impressions'].sum()
    st.metric("👁️ Impressions", f"{total_impressions:,}")

with col2:
    total_clicks = filtered_df['Unique outbound clicks'].sum()
    st.metric("🖱️ Clicks", f"{int(total_clicks):,}")

with col3:
    avg_cpa = filtered_df['CPA'].mean() if total_purchases > 0 else 0
    st.metric("💵 Avg CPA", f"${avg_cpa:,.2f}")

with col4:
    avg_cvr = filtered_df['CVR'].mean()
    st.metric("📈 Avg CVR", f"{avg_cvr:.2f}%")

with col5:
    avg_cpm = filtered_df['CPM'].mean()
    st.metric("📊 Avg CPM", f"${avg_cpm:.2f}")

with col6:
    total_atc = filtered_df['Adds to cart'].sum()
    st.metric("🛒 Add to Cart", f"{int(total_atc)}")

st.markdown("---")

# Row 1: Niche & Product Performance
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Top Niches by Purchases")
    niche_perf = filtered_df.groupby(['nicheID', 'niche_name']).agg({
        'Purchases': 'sum',
        'Amount Spent': 'sum',
        'Revenue': 'sum',
        'Profit': 'sum',
        'CVR': 'mean',
        'ROAS': 'mean'
    }).reset_index().sort_values('Purchases', ascending=True).tail(10)
    
    fig_niche = go.Figure(go.Bar(
        x=niche_perf['Purchases'],
        y=niche_perf['niche_name'],
        orientation='h',
        marker=dict(
            color=niche_perf['Profit'],
            colorscale='Blues',  # Light to dark blue gradient
            showscale=True,
            colorbar=dict(title="Profit ($)")
        ),
        text=niche_perf['Purchases'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Purchases: %{x}<br>Spend: $%{customdata[0]:.2f}<br>Revenue: $%{customdata[1]:.2f}<br>Profit: $%{customdata[2]:.2f}<br>ROAS: %{customdata[3]:.2f}x<extra></extra>',
        customdata=niche_perf[['Amount Spent', 'Revenue', 'Profit', 'ROAS']].values
    ))
    fig_niche.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Purchases", yaxis_title="")
    st.plotly_chart(fig_niche, use_container_width=True)

with col2:
    st.markdown("### 📦 Top Products by Purchases")
    # Group by productID only to avoid missing data
    product_perf = filtered_df.groupby('productID').agg({
        'Purchases': 'sum',
        'Amount Spent': 'sum',
        'Revenue': 'sum',
        'Profit': 'sum',
        'CVR': 'mean',
        'ROAS': 'mean',
        'product_name': 'first'  # Get first product_name for reference
    }).reset_index().sort_values('Purchases', ascending=True).tail(10)
    
    # Create display name: show product_name if available, otherwise just productID
    product_perf['display_name'] = product_perf.apply(
        lambda x: f"{x['productID']} ({x['product_name']})" if pd.notna(x['product_name']) and x['product_name'] != 'Null' else x['productID'], 
        axis=1
    )
    
    fig_product = go.Figure(go.Bar(
        x=product_perf['Purchases'],
        y=product_perf['display_name'],
        orientation='h',
        marker=dict(
            color=product_perf['Profit'],
            colorscale='Teal',  # Light to dark teal gradient
            showscale=True,
            colorbar=dict(title="Profit ($)")
        ),
        text=product_perf['Purchases'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Purchases: %{x}<br>Spend: $%{customdata[0]:.2f}<br>Revenue: $%{customdata[1]:.2f}<br>Profit: $%{customdata[2]:.2f}<br>ROAS: %{customdata[3]:.2f}x<extra></extra>',
        customdata=product_perf[['Amount Spent', 'Revenue', 'Profit', 'ROAS']].values
    ))
    fig_product.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Purchases", yaxis_title="")
    st.plotly_chart(fig_product, use_container_width=True)

# Row 3: Demographics
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚥ Gender Split")
    gender_data = filtered_df.groupby('Gender')['Purchases'].sum().reset_index()
    
    fig_gender = go.Figure(go.Pie(
        labels=gender_data['Gender'],
        values=gender_data['Purchases'],
        hole=0.4,
        marker=dict(colors=[COLORS['male'], COLORS['female']]),
        textinfo='label+percent',
        textposition='outside'
    ))
    fig_gender.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    st.markdown("### 👥 Age Group Performance")
    age_gender_data = filtered_df.groupby(['Age', 'Gender'])['Purchases'].sum().reset_index()
    
    fig_age = px.bar(
        age_gender_data,
        x='Purchases',
        y='Age',
        color='Gender',
        orientation='h',
        color_discrete_map={'male': COLORS['male'], 'female': COLORS['female']},
        text='Purchases'
    )
    fig_age.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Purchases", yaxis_title="Age")
    fig_age.update_traces(textposition='outside')
    st.plotly_chart(fig_age, use_container_width=True)

st.markdown("---")

# Drill-Down Section
st.markdown("## 🔍 Drill-Down Analysis by Design")

# Get list of products with purchases
products_with_purchases = filtered_df[filtered_df['Purchases'] > 0].groupby('productID').agg({
    'Purchases': 'sum',
    'Profit': 'sum',
    'product_name': 'first'
}).reset_index().sort_values('Purchases', ascending=False)

# Create options for selectbox
product_options_drilldown = ['Select a product...'] + [
    f"{row['productID']} - {row['Purchases']:.0f} purchases (${row['Profit']:.2f} profit)"
    for _, row in products_with_purchases.iterrows()
]

selected_product_display = st.selectbox(
    "Select a product to see design breakdown:",
    options=product_options_drilldown,
    key='product_drilldown'
)

if selected_product_display != 'Select a product...':
    # Extract productID from selection
    selected_product_id = selected_product_display.split(' - ')[0]
    
    st.markdown(f"### 📊 Design Breakdown for {selected_product_id}")
    
    # Filter data for selected product
    product_designs = filtered_df[filtered_df['productID'] == selected_product_id].copy()
    
    # Aggregate by designID
    design_perf = product_designs.groupby('designID').agg({
        'Purchases': 'sum',
        'Amount Spent': 'sum',
        'Revenue': 'sum',
        'Profit': 'sum',
        'ROAS': 'mean',
        'CVR': 'mean',
        'Unique outbound clicks': 'sum',
        'Adds to cart': 'sum'
    }).reset_index()
    
    # Remove Unknown designs
    design_perf = design_perf[design_perf['designID'] != 'Unknown']
    
    if len(design_perf) > 0:
        design_perf = design_perf.sort_values('Purchases', ascending=False)
        
        # Show metrics for this product
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Designs", len(design_perf))
        with col2:
            st.metric("Total Purchases", f"{int(design_perf['Purchases'].sum())}")
        with col3:
            st.metric("Total Profit", f"${design_perf['Profit'].sum():,.2f}")
        with col4:
            st.metric("Avg ROAS", f"{design_perf['ROAS'].mean():.2f}x")
        
        # Design performance chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Designs by Purchases")
            top_designs = design_perf.head(15).sort_values('Purchases', ascending=True)
            
            fig_design_purchases = go.Figure(go.Bar(
                x=top_designs['Purchases'],
                y=top_designs['designID'],
                orientation='h',
                marker=dict(
                    color=top_designs['Purchases'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=top_designs['Purchases'],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Purchases: %{x}<br>Profit: $%{customdata[0]:.2f}<br>ROAS: %{customdata[1]:.2f}x<extra></extra>',
                customdata=top_designs[['Profit', 'ROAS']].values
            ))
            fig_design_purchases.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Purchases", yaxis_title="")
            st.plotly_chart(fig_design_purchases, use_container_width=True)
        
        with col2:
            st.markdown("#### Top Designs by Profit")
            top_designs_profit = design_perf.head(15).sort_values('Profit', ascending=True)
            
            fig_design_profit = go.Figure(go.Bar(
                x=top_designs_profit['Profit'],
                y=top_designs_profit['designID'],
                orientation='h',
                marker=dict(
                    color=top_designs_profit['Profit'],
                    colorscale='Teal',
                    showscale=False
                ),
                text=[f"${x:.0f}" for x in top_designs_profit['Profit']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Profit: $%{x:.2f}<br>Purchases: %{customdata[0]}<br>ROAS: %{customdata[1]:.2f}x<extra></extra>',
                customdata=top_designs_profit[['Purchases', 'ROAS']].values
            ))
            fig_design_profit.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Profit ($)", yaxis_title="")
            st.plotly_chart(fig_design_profit, use_container_width=True)
        
        # Detailed table for designs
        st.markdown("#### Design Details")
        design_table = design_perf[['designID', 'Purchases', 'Amount Spent', 'Revenue', 'Profit', 'ROAS', 'CVR', 'Unique outbound clicks', 'Adds to cart']].copy()
        design_table = design_table.sort_values('Profit', ascending=False)
        
        # Format columns
        design_table['Amount Spent'] = design_table['Amount Spent'].apply(lambda x: f"${x:,.2f}")
        design_table['Revenue'] = design_table['Revenue'].apply(lambda x: f"${x:,.2f}")
        design_table['Profit'] = design_table['Profit'].apply(lambda x: f"${x:,.2f}")
        design_table['ROAS'] = design_table['ROAS'].apply(lambda x: f"{x:.2f}x")
        design_table['CVR'] = design_table['CVR'].apply(lambda x: f"{x:.2f}%")
        
        design_table.columns = ['Design ID', 'Purchases', 'Spend', 'Revenue', 'Profit', 'ROAS', 'CVR', 'Clicks', 'ATC']
        
        st.dataframe(design_table, use_container_width=True, height=400, hide_index=True)
    else:
        st.warning(f"No design data found for {selected_product_id}. This might be due to missing designID in campaign names.")
else:
    st.info("👆 Select a product above to see detailed design breakdown and performance metrics.")

# Row 4: Detailed Table
st.markdown("### 📋 Campaign Details")

# Prepare table data
table_df = filtered_df[['Campaign name', 'nicheID', 'niche_name', 'productID', 'product_name', 'Age', 'Gender', 
                         'Amount Spent', 'Revenue', 'Profit', 'ROAS', 
                         'Unique outbound clicks', 'Adds to cart', 'Purchases', 
                         'CPA', 'CVR']].copy()
table_df = table_df.sort_values('Profit', ascending=False)

# Format columns
table_df['Amount Spent'] = table_df['Amount Spent'].apply(lambda x: f"${x:,.2f}")
table_df['Revenue'] = table_df['Revenue'].apply(lambda x: f"${x:,.2f}")
table_df['Profit'] = table_df['Profit'].apply(lambda x: f"${x:,.2f}")
table_df['ROAS'] = table_df['ROAS'].apply(lambda x: f"{x:.2f}x")
table_df['CPA'] = table_df['CPA'].apply(lambda x: f"${x:,.2f}")
table_df['CVR'] = table_df['CVR'].apply(lambda x: f"{x:.2f}%")

# Rename columns for display
table_df.columns = ['Campaign Name', 'Niche ID', 'Niche', 'Product ID', 'Product', 'Age', 'Gender', 
                    'Spend', 'Revenue', 'Profit', 'ROAS',
                    'Clicks', 'ATC', 'Purchases', 'CPA', 'CVR']

st.dataframe(
    table_df,
    use_container_width=True,
    height=400,
    hide_index=True
)

# Download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name='ad_performance_data.csv',
    mime='text/csv',
)

# Footer
st.markdown("---")
st.markdown("**💡 Insights:** Use filters to drill down into specific segments. Click on charts for more details.")