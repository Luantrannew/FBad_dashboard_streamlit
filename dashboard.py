import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

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
    .upload-section {background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
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

# Preprocessing function
@st.cache_data
def preprocess_data(df_raw):
    """Preprocess raw Facebook Ads data with flexible column handling"""
    
    # Map common column name variations to standard names
    column_mappings = {
        'Amount spent (USD)': ['Amount spent', 'Spend', 'Cost'],
        'CPM (cost per 1,000 impressions) (USD)': ['CPM', 'Cost per 1000 impressions'],
        'Cost per unique outbound click (USD)': ['Cost per click', 'CPC', 'Cost per outbound click'],
        'Cost per add to cart (USD)': ['Cost per ATC', 'Cost per add to cart'],
        'Unique outbound clicks': ['Outbound clicks', 'Link clicks', 'Clicks'],
        'Adds to cart': ['Add to cart', 'ATC'],
        'Checkouts initiated': ['Checkouts', 'Initiated checkouts'],
    }
    
    # Try to find columns with alternative names
    df_raw_copy = df_raw.copy()
    for standard_name, alternatives in column_mappings.items():
        if standard_name not in df_raw_copy.columns:
            for alt in alternatives:
                if alt in df_raw_copy.columns:
                    df_raw_copy = df_raw_copy.rename(columns={alt: standard_name})
                    break
    
    # Add missing optional columns with defaults
    default_values = {
        'Reporting starts': pd.Timestamp.now(),
        'Reporting ends': pd.Timestamp.now(),
        'Ad set name': 'N/A',
        'Ad name': 'N/A',
        'CPM (cost per 1,000 impressions) (USD)': 0,
        'Frequency': 0,
        'Post comments': 0,
        'Unique outbound clicks': 0,
        'Cost per unique outbound click (USD)': 0,
        'Adds to cart': 0,
        'Cost per add to cart (USD)': 0,
        'Checkouts initiated': 0,
        'Purchases': 0
    }
    
    for col, default_val in default_values.items():
        if col not in df_raw_copy.columns:
            df_raw_copy[col] = default_val
    
    # Define columns we want (in any order)
    desired_columns = [
        'Reporting starts', 'Reporting ends',
        'Campaign name', 'Ad set name', 'Ad name',
        'Age', 'Gender',
        'Amount spent (USD)', 'Impressions',
        'CPM (cost per 1,000 impressions) (USD)', 'Frequency', 'Post comments',
        'Unique outbound clicks', 'Cost per unique outbound click (USD)',
        'Adds to cart', 'Cost per add to cart (USD)',
        'Checkouts initiated', 'Purchases'
    ]
    
    # Select only existing columns (order doesn't matter)
    existing_columns = [col for col in desired_columns if col in df_raw_copy.columns]
    df = df_raw_copy[existing_columns].copy()
    
    # Rename columns
    df = df.rename(columns={
        'CPM (cost per 1,000 impressions) (USD)': 'CPM',
        'Amount spent (USD)': 'Amount Spent',
        'Cost per unique outbound click (USD)': 'Cost per Click',
        'Cost per add to cart (USD)': 'Cost per ATC'
    })
    
    # Niche dictionary
    niche_dict = {
        'MEX': 'Tự hào Mexico', 'LAT': 'Tự hào Latin Countries NOT Mexico', 'HSP': 'Hispanic',
        'DTG': 'Dusted Girl', 'QCUS': 'Quick Custom', 'PSG': 'Plus Size Girl', 'AAA': 'Design trơn',
        'KID': 'For Kids', 'FTH': 'Faith', 'CPL': 'For Couples', '3T': '3D TShirt', 'BST': 'For Besties',
        '3H': '3D Hoodie', 'GDT': 'For Granddaughter', 'DAU': 'For Daughter', 'GSN': 'For Grandson',
        'SON': 'For Son', 'HSB': 'For Husband', 'WIF': 'For Wife', 'DAD': 'For Dad', 'MOM': 'For Mom',
        'GRM': 'For Grandma', 'GRP': 'For Grandpa', 'FAM': 'Family', 'DOG': 'For Dog Lovers',
        'CAT': 'For Cat Lovers', 'PET': 'Other Pet Lovers', 'PME': 'Pet Memorial', 'HME': 'Human Memorial',
        'FTN': 'For Fitness People', 'COL': 'For Colleagues', 'BLV': 'For Book Lovers',
        'GRAD': 'For Graduation', 'DGM': 'For Dog Mom', 'GRK': 'For Grandkids', 'NUR': 'For Nurses',
        'DOC': 'For Doctors', 'FIF': 'For Firefighters', 'OFC': 'For Officers (Police)',
        'OFW': 'For Office Workers', 'TEACH': 'For Teachers', 'READ': 'For Reading Lovers',
        'FISH': 'For Fishing Lovers', 'CAMP': 'For Camping Lovers', 'GARD': 'For Gardening Lovers',
        'YOGA': 'For Yogo Lovers', 'SPORT': 'For Sport Lovers', 'VETRN': 'For veterans',
        'RETIRE': 'For retirement', 'BDAY': 'For birthday', 'GRILL': 'For grilling lovers',
        'GOLF': 'For golf lovers', 'COOK': 'For cooking lovers', 'INDE': 'For independence day/Patriotrism',
        'DILA': 'For Daughter-In-Law', 'SILA': 'For Son-In-Law', 'MOLA': 'For Mother-In-Law',
        'FALA': 'For Father-In-Law', 'BUS': 'For Bus Driver', 'TRUCK': 'For Truck Driver',
        'WEDD': 'Gift for wedding day', 'LAW': 'For Lawyers', 'XMAS': 'For Xmas'
    }
    
    # Product dictionary
    product_dict = {
        'HB': 'Leather Handbag', 'TB': '20oz/30oz Tumbler', 'WW': 'Leather Women Wallet',
        'DKB': 'Backpack Duckbilled', 'SCRD': 'Square Ceramic Ring Dish', 'JARL': 'Mason Jar Light',
        'HGO': 'Plastic Hanging Ornament', 'CGOR': 'Circle Glass Ornament',
        'RSUN': 'Round Stained Glass Window Hanging Suncatcher', 'HRTS': 'Heart Stone',
        'APLA': 'Custom Shape Acrylic Plaque', 'MAGN': 'Fridge Magnet Custom Shape',
        'CEW': 'Classic Engraved Men Wallet', 'EWG': 'Engraved Whiskey Glass', 'JRNL': 'Leather Journal',
        'CSUN': 'Custom Suncatcher', 'HCE': 'Heart Ceramic Ornament', 'RCE': 'Round Ceramic Ornament',
        'TIE': 'Custom Tie', 'BNIE': 'Custom Beanie', 'LANT': 'Christmas Lantern',
        'ACO': 'Acrylic Ornament With Custom Shape (Xmas)', 'TWS': 'Ugly Wool Sweatshirt',
        'LECA': 'Led Candle', 'WDO': 'Wood Ornament Custom Shape (Xmas)', 'BLK': 'Sherpa/Fleece Blanket',
        'PLW': 'Custom Pillow (1-side print)', 'PJA': 'Adult Pajama Pants', 'MG': 'White Edge-to-Edge Mug',
        'DRM': 'Doormat', 'MZB': 'Music Box', 'XSO': 'Xmas Decorative Sock',
        'WCA': 'Wood Car Hanging Ornament With Custom Shape',
        'ACA': 'Acrylic Car Hanging Ornament With Custom Shape', 'NPR': 'No-Print Product',
        'ALCA': 'Aluminum Wallet Card 1 Side Print', 'WIGL': 'Wine Glass 15OZ',
        'RCRF': 'Round Ceramic Ring Dish (Full Printed)', 'SCRF': 'Square Ceramic Ring Dish (Full Printed)',
        'TMS': 'Tape Measure 5M', 'BRG': '16OZ Beer Glass', 'GLM': 'Glass Mug 11OZ',
        'HAP': 'Heart Shaped Acrylic Plaque', 'LGT': 'Luggage Tag', 'PPH': 'Passport Holder/Cover',
        'RWG': 'Round Whiskey Glass, 2 Side Print', 'VIS': 'Car Visor Clip, 1 Side Print, 2 Layers',
        'TAG': 'Dogtag Necklace', 'CEV': '15OZ Ceramic Flower Vase',
        'CEP': 'Ceramic Plant Pot with Bamboo Tray', 'WSO': 'Wooden Slider Ornament',
        'PMGW': '12OZ White Pottery Mug', 'PMGB': '12OZ Blue Pottery Mug',
        'PMGO': '12OZ Orange Pottery Mug', 'PMGY': '12OZ Yellow Pottery Mug',
        'WPLA': '2 Layer Wood Plaque With Flat Base', 'PMAM': 'Pink Marble Mug',
        'BTL': 'Bottle Lamp 2.9x13in'
    }
    
    # Parse campaign name
    def parse_campaign_name(campaign_name):
        if pd.isna(campaign_name):
            return None, None
        parts = str(campaign_name).split(' - ')
        if len(parts) == 0:
            return None, None
        first_part = parts[0].strip()
        if '_' not in first_part:
            return None, None
        niche_product = first_part.split('_')
        if len(niche_product) < 2:
            return None, None
        niche_code = re.sub(r'\d+$', '', niche_product[0])
        product_code = niche_product[1]
        return niche_code, product_code
    
    if 'Campaign name' in df.columns:
        df[['nicheID', 'productID']] = df['Campaign name'].apply(lambda x: pd.Series(parse_campaign_name(x)))
        df['niche_name'] = df['nicheID'].map(niche_dict)
        df['product_name'] = df['productID'].map(product_dict)
    
    # Extract Ad Manager
    def extract_ad_manager(campaign_name):
        if pd.isna(campaign_name):
            return 'Other'
        campaign_str = str(campaign_name).upper()
        if 'AD-' in campaign_str:
            start_idx = campaign_str.find('AD-') + 3
            comma_idx = campaign_str.find(',', start_idx)
            star_idx = campaign_str.find('*', start_idx)
            if comma_idx != -1:
                end_idx = comma_idx
            elif star_idx != -1:
                end_idx = star_idx
            else:
                end_idx = -1
            if end_idx != -1:
                ad_manager = campaign_str[start_idx:end_idx].strip()
                return ad_manager if ad_manager else 'Other'
        return 'Other'
    
    if 'Campaign name' in df.columns:
        df['Ad_Manager'] = df['Campaign name'].apply(extract_ad_manager)
    
    # Extract designID
    def extract_design_id(campaign_name, niche_id, product_id):
        if pd.isna(campaign_name) or pd.isna(niche_id) or pd.isna(product_id):
            return 'Unknown'
        campaign_str = str(campaign_name)
        niche_str = str(niche_id)
        product_str = str(product_id)
        pattern = f"{re.escape(niche_str)}\\d+(?=_{re.escape(product_str)})"
        match = re.search(pattern, campaign_str)
        if match:
            return match.group(0)
        return 'Unknown'
    
    if all(col in df.columns for col in ['Campaign name', 'nicheID', 'productID']):
        df['designID'] = df.apply(lambda x: extract_design_id(x['Campaign name'], x['nicheID'], x['productID']), axis=1)
    
    # Ensure numeric columns
    numeric_cols = ['Amount Spent', 'Impressions', 'Unique outbound clicks', 
                   'Adds to cart', 'Checkouts initiated', 'Purchases', 
                   'CPM', 'Cost per Click', 'Cost per ATC', 'Frequency', 'Post comments']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate metrics
    if 'Unique outbound clicks' in df.columns and 'Impressions' in df.columns:
        df['CTR'] = (df['Unique outbound clicks'] / df['Impressions'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    if 'Adds to cart' in df.columns and 'Unique outbound clicks' in df.columns:
        df['ATC_Rate'] = (df['Adds to cart'] / df['Unique outbound clicks'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    if 'Checkouts initiated' in df.columns and 'Adds to cart' in df.columns:
        df['Checkout_Rate'] = (df['Checkouts initiated'] / df['Adds to cart'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    if 'Purchases' in df.columns and 'Checkouts initiated' in df.columns:
        df['Purchase_Rate'] = (df['Purchases'] / df['Checkouts initiated'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    if 'Purchases' in df.columns and 'Unique outbound clicks' in df.columns:
        df['CVR'] = (df['Purchases'] / df['Unique outbound clicks'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    if 'Amount Spent' in df.columns and 'Purchases' in df.columns:
        df['CPA'] = (df['Amount Spent'] / df['Purchases']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    # Calculate revenue/profit
    if 'productID' in df.columns and 'Purchases' in df.columns:
        pricing = load_product_pricing()
        df['Selling_Price'] = df['productID'].map(pricing).fillna(0)
        df['Revenue'] = (df['Purchases'] * df['Selling_Price']).round(2)
        
        if 'Amount Spent' in df.columns:
            df['Profit'] = (df['Revenue'] - df['Amount Spent']).round(2)
            df['Profit_Margin'] = ((df['Profit'] / df['Revenue']) * 100).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
            df['ROAS'] = (df['Revenue'] / df['Amount Spent']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    return df

# Main App
st.markdown("# Ad Performance Dashboard")

# File Upload Section
# st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("## Data Upload")
st.markdown("""
**Upload Guidelines:**
- Supported formats: Excel (.xlsx, .xls)
- Use age and gender breakdown file from Facebook Ads Manager
- Maximum file size: 200MB
- Recommended: < 1,000,000 rows for optimal performance
- Estimated capacity: ~2-3 million rows at 200MB limit
""")

uploaded_file = st.file_uploader(
    "Upload your Facebook Ads data (Excel file) ", 
    type=['xlsx', 'xls'],
    help="Upload the raw export from Facebook Ads Manager"
)

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df_raw = pd.read_excel(uploaded_file, sheet_name='Worksheet')
        
        # Check for minimum required columns
        required_cols = ['Campaign name', 'Age', 'Gender']
        missing_required = [col for col in required_cols if col not in df_raw.columns]
        
        if missing_required:
            st.error(f"Missing critical columns: {', '.join(missing_required)}")
            st.stop()
        
        # Preprocess data
        with st.spinner('Processing data...'):
            df = preprocess_data(df_raw)
        
        # Get date range from data
        if 'Reporting starts' in df.columns and 'Reporting ends' in df.columns:
            df['Reporting starts'] = pd.to_datetime(df['Reporting starts'], errors='coerce')
            df['Reporting ends'] = pd.to_datetime(df['Reporting ends'], errors='coerce')
            date_start = df['Reporting starts'].min()
            date_end = df['Reporting ends'].max()
            date_range_display = f"{date_start.strftime('%b %d, %Y')} - {date_end.strftime('%b %d, %Y')}"
        else:
            date_range_display = 'N/A'
        
        st.success(f"Data loaded successfully! {len(df)} rows processed.")
        st.info(f"Date Range: {date_range_display}")
        
        # Show data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store data in session state for persistence
        st.session_state['df'] = df
        st.session_state['date_range'] = date_range_display
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()
else:
    st.markdown('</div>', unsafe_allow_html=True)
    st.warning("Please upload a data file to begin analysis")
    st.stop()

# Use data from session state
df = st.session_state.get('df')
date_range_display = st.session_state.get('date_range', 'N/A')

if df is None:
    st.stop()

st.markdown(f"**Date Range:** {date_range_display}")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("Filters")

# Ad Manager filter
if 'Ad_Manager' in df.columns:
    ad_manager_values = df['Ad_Manager'].dropna().unique().tolist()
    ad_manager_options = ['All'] + sorted([x for x in ad_manager_values if x])
    selected_ad_manager = st.sidebar.multiselect('Ad Manager', ad_manager_options, default=['All'])
else:
    selected_ad_manager = ['All']

# Age filter
if 'Age' in df.columns:
    age_values = df['Age'].dropna().unique().tolist()
    age_options = ['All'] + sorted([x for x in age_values if x])
    selected_age = st.sidebar.multiselect('Age Group', age_options, default=['All'])
else:
    selected_age = ['All']

# Gender filter
if 'Gender' in df.columns:
    gender_options = ['All', 'male', 'female']
    selected_gender = st.sidebar.radio('Gender', gender_options)
else:
    selected_gender = 'All'

# Niche filter
if 'nicheID' in df.columns:
    niche_values = df['nicheID'].dropna().unique().tolist()
    niche_options = ['All'] + sorted([x for x in niche_values if x])
    selected_niche = st.sidebar.multiselect('Niche', niche_options, default=['All'])
else:
    selected_niche = ['All']

# Product filter
if 'productID' in df.columns:
    product_values = df['productID'].dropna().unique().tolist()
    product_options = ['All'] + sorted([x for x in product_values if x])
    selected_product = st.sidebar.multiselect('Product', product_options, default=['All'])
else:
    selected_product = ['All']

# Apply filters
filtered_df = df.copy()

if 'Ad_Manager' in df.columns and 'All' not in selected_ad_manager and selected_ad_manager:
    filtered_df = filtered_df[filtered_df['Ad_Manager'].isin(selected_ad_manager)]

if 'Age' in df.columns and 'All' not in selected_age and selected_age:
    filtered_df = filtered_df[filtered_df['Age'].isin(selected_age)]

if 'Gender' in df.columns and selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

if 'nicheID' in df.columns and 'All' not in selected_niche and selected_niche:
    filtered_df = filtered_df[filtered_df['nicheID'].isin(selected_niche)]

if 'productID' in df.columns and 'All' not in selected_product and selected_product:
    filtered_df = filtered_df[filtered_df['productID'].isin(selected_product)]

# KPI Cards
st.markdown("### Key Metrics")

has_spend = 'Amount Spent' in filtered_df.columns
has_purchases = 'Purchases' in filtered_df.columns
has_revenue = 'Revenue' in filtered_df.columns
has_profit = 'Profit' in filtered_df.columns
has_impressions = 'Impressions' in filtered_df.columns
has_clicks = 'Unique outbound clicks' in filtered_df.columns

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if has_spend:
        total_spend = filtered_df['Amount Spent'].sum()
        st.metric("Total Spend", f"${total_spend:,.2f}")
    else:
        st.warning("Missing: Amount Spent")

with col2:
    if has_purchases:
        total_purchases = filtered_df['Purchases'].sum()
        st.metric("Purchases", f"{int(total_purchases)}")
    else:
        st.warning("Missing: Purchases")

with col3:
    if has_revenue:
        total_revenue = filtered_df['Revenue'].sum()
        st.metric("Revenue", f"${total_revenue:,.2f}")
    else:
        st.warning("Missing: productID")

with col4:
    if has_profit:
        total_profit = filtered_df['Profit'].sum()
        st.metric("Profit", f"${total_profit:,.2f}")
    else:
        st.warning("Missing: data")

with col5:
    if has_revenue and has_spend and has_spend and total_spend > 0:
        avg_roas = filtered_df['ROAS'].mean()
        st.metric("ROAS", f"{avg_roas:.2f}x")
    else:
        st.warning("Missing: ROAS data")

with col6:
    if has_profit and has_revenue:
        avg_margin = filtered_df['Profit_Margin'].mean()
        st.metric("Profit Margin", f"{avg_margin:.1f}%")
    else:
        st.warning("Missing: margin data")

st.markdown("---")

# Second row of metrics
st.markdown("### Performance Insights")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if has_impressions:
        total_impressions = filtered_df['Impressions'].sum()
        st.metric("Impressions", f"{total_impressions:,}")
    else:
        st.warning("Missing: Impressions")

with col2:
    if has_clicks:
        total_clicks = filtered_df['Unique outbound clicks'].sum()
        st.metric("Clicks", f"{int(total_clicks):,}")
    else:
        st.warning("Missing: Clicks")

with col3:
    if 'CPA' in filtered_df.columns and has_purchases and total_purchases > 0:
        avg_cpa = filtered_df['CPA'].mean()
        st.metric("Avg CPA", f"${avg_cpa:,.2f}")
    else:
        st.warning("Missing: CPA data")

with col4:
    if 'CVR' in filtered_df.columns:
        avg_cvr = filtered_df['CVR'].mean()
        st.metric("Avg CVR", f"{avg_cvr:.2f}%")
    else:
        st.warning("Missing: CVR data")

with col5:
    if 'CPM' in filtered_df.columns:
        avg_cpm = filtered_df['CPM'].mean()
        st.metric("Avg CPM", f"${avg_cpm:.2f}")
    else:
        st.warning("Missing: CPM")

with col6:
    if 'Adds to cart' in filtered_df.columns:
        total_atc = filtered_df['Adds to cart'].sum()
        st.metric("Add to Cart", f"{int(total_atc)}")
    else:
        st.warning("Missing: ATC")

st.markdown("---")

# Row 1: Niche & Product Performance
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Niches by Purchases")
    
    if 'nicheID' in filtered_df.columns and 'Purchases' in filtered_df.columns:
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
                colorscale='Blues',
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
    else:
        st.warning("Cannot display: Missing Campaign name or Purchases")

with col2:
    st.markdown("### Top Products by Purchases")
    
    if 'productID' in filtered_df.columns and 'Purchases' in filtered_df.columns:
        product_perf = filtered_df.groupby('productID').agg({
            'Purchases': 'sum',
            'Amount Spent': 'sum',
            'Revenue': 'sum',
            'Profit': 'sum',
            'CVR': 'mean',
            'ROAS': 'mean',
            'product_name': 'first'
        }).reset_index().sort_values('Purchases', ascending=True).tail(10)
        
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
                colorscale='Teal',
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
    else:
        st.warning("Cannot display: Missing Campaign name or Purchases")

# Demographics
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Gender Split")
    
    if 'Gender' in filtered_df.columns and 'Purchases' in filtered_df.columns:
        gender_data = filtered_df.groupby('Gender')['Purchases'].sum().reset_index()
        
        fig_gender = go.Figure(go.Pie(
            labels=gender_data['Gender'],
            values=gender_data['Purchases'],
            hole=0.4,
            marker=dict(colors=[COLORS['female'], COLORS['male']]),
            textinfo='label+percent',
            textposition='outside'
        ))
        fig_gender.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
        st.plotly_chart(fig_gender, use_container_width=True)
    else:
        st.warning("Cannot display: Missing Gender or Purchases")

with col2:
    st.markdown("### Age Group Performance")
    
    if 'Age' in filtered_df.columns and 'Gender' in filtered_df.columns and 'Purchases' in filtered_df.columns:
        age_gender_data = filtered_df.groupby(['Age', 'Gender'])['Purchases'].sum().reset_index()
        
        # Add text column with conditional display (hide 0 values)
        age_gender_data['text_display'] = age_gender_data['Purchases'].apply(
            lambda x: str(int(x)) if x > 0 else ''
        )
        
        fig_age = px.bar(
            age_gender_data,
            x='Purchases',
            y='Age',
            color='Gender',
            orientation='h',
            color_discrete_map={'male': COLORS['male'], 'female': COLORS['female']},
            text='text_display'  # Use conditional text instead
        )
        fig_age.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Purchases", yaxis_title="Age")
        fig_age.update_traces(textposition='inside')  # Change to 'inside' for cleaner look
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.warning("Cannot display: Missing Age, Gender or Purchases")

st.markdown("---")

# Drill-Down Section
st.markdown("## Drill-Down Analysis by Design")

if 'productID' in filtered_df.columns and 'Purchases' in filtered_df.columns:
    products_with_purchases = filtered_df[filtered_df['Purchases'] > 0].groupby('productID').agg({
        'Purchases': 'sum',
        'Profit': 'sum',
        'product_name': 'first'
    }).reset_index().sort_values('Purchases', ascending=False)

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
        selected_product_id = selected_product_display.split(' - ')[0]
        
        st.markdown(f"### Design Breakdown for {selected_product_id}")
        
        product_designs = filtered_df[filtered_df['productID'] == selected_product_id].copy()
        
        if 'designID' in product_designs.columns:
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
            
            design_perf = design_perf[design_perf['designID'] != 'Unknown']
            
            if len(design_perf) > 0:
                design_perf = design_perf.sort_values('Purchases', ascending=False)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Designs", len(design_perf))
                with col2:
                    st.metric("Total Purchases", f"{int(design_perf['Purchases'].sum())}")
                with col3:
                    st.metric("Total Profit", f"${design_perf['Profit'].sum():,.2f}")
                with col4:
                    st.metric("Avg ROAS", f"{design_perf['ROAS'].mean():.2f}x")
                
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
                
                st.markdown("#### Design Details")
                design_table = design_perf[['designID', 'Purchases', 'Amount Spent', 'Revenue', 'Profit', 'ROAS', 'CVR', 'Unique outbound clicks', 'Adds to cart']].copy()
                design_table = design_table.sort_values('Profit', ascending=False)
                
                design_table['Amount Spent'] = design_table['Amount Spent'].apply(lambda x: f"${x:,.2f}")
                design_table['Revenue'] = design_table['Revenue'].apply(lambda x: f"${x:,.2f}")
                design_table['Profit'] = design_table['Profit'].apply(lambda x: f"${x:,.2f}")
                design_table['ROAS'] = design_table['ROAS'].apply(lambda x: f"{x:.2f}x")
                design_table['CVR'] = design_table['CVR'].apply(lambda x: f"{x:.2f}%")
                
                design_table.columns = ['Design ID', 'Purchases', 'Spend', 'Revenue', 'Profit', 'ROAS', 'CVR', 'Clicks', 'ATC']
                
                st.dataframe(design_table, use_container_width=True, height=400, hide_index=True)
            else:
                st.warning(f"No design data found for {selected_product_id}.")
        else:
            st.warning("Cannot extract designID from Campaign names")
    else:
        st.info("Select a product above to see detailed design breakdown")
else:
    st.warning("Cannot display drill-down: Missing productID or Purchases")

st.markdown("---")

# Campaign Details Table
st.markdown("### Campaign Details")

if 'Campaign name' in filtered_df.columns:
    display_cols = [col for col in ['Campaign name', 'nicheID', 'niche_name', 'productID', 'product_name', 'Age', 'Gender', 
                             'Amount Spent', 'Revenue', 'Profit', 'ROAS', 
                             'Unique outbound clicks', 'Adds to cart', 'Purchases', 
                             'CPA', 'CVR'] if col in filtered_df.columns]
    
    table_df = filtered_df[display_cols].copy()
    
    if 'Profit' in table_df.columns:
        table_df = table_df.sort_values('Profit', ascending=False)

    # Format numeric columns
    for col in ['Amount Spent', 'Revenue', 'Profit', 'CPA']:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: f"${x:,.2f}")
    
    if 'ROAS' in table_df.columns:
        table_df['ROAS'] = table_df['ROAS'].apply(lambda x: f"{x:.2f}x")
    
    if 'CVR' in table_df.columns:
        table_df['CVR'] = table_df['CVR'].apply(lambda x: f"{x:.2f}%")

    # Rename for display
    rename_map = {
        'Campaign name': 'Campaign Name',
        'nicheID': 'Niche ID',
        'niche_name': 'Niche',
        'productID': 'Product ID',
        'product_name': 'Product',
        'Amount Spent': 'Spend',
        'Unique outbound clicks': 'Clicks',
        'Adds to cart': 'ATC'
    }
    table_df = table_df.rename(columns=rename_map)

    st.dataframe(table_df, use_container_width=True, height=400, hide_index=True)

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='ad_performance_data.csv',
        mime='text/csv',
    )
else:
    st.warning("Cannot display table: Missing Campaign name column")

st.markdown("---")
st.markdown("**Insights:** Use filters to drill down into specific segments.")