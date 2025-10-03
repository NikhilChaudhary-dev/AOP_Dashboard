import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="📊 AOP Dashboard Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0 2rem 0;
    text-align: center;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(45deg, #ffffff, #e3f2fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    font-size: 1.2rem;
    margin: 0.5rem 0 0 0;
    color: rgba(255,255,255,0.9);
}

.metric-container {
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid rgba(102, 126, 234, 0.3);
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    text-align: center;
    transition: all 0.3s ease;
}

.metric-container:hover {
    transform: translateY(-10px);
    box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #4facfe;
    margin: 0;
    text-shadow: 0 0 20px rgba(79, 172, 254, 0.5);
}

.metric-label {
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.5rem;
    font-weight: 500;
}

.section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 15px;
    margin: 2rem 0 1rem 0;
    font-size: 1.8rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.plotly-graph-div {
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border-radius: 20px;
    padding: 1rem;
    border: 1px solid rgba(102, 126, 234, 0.3);
}

.person-tab {
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 AOP Dashboard Analytics</h1>
    <p>Annual Operating Plan - Sales Performance Tracking</p>
</div>
""", unsafe_allow_html=True)

# Global definitions
years = ['2023', '2024', '2025']
regions = ['US', 'EUUK', 'ANZ', 'Others']

# Function to load and process data
@st.cache_data
def load_aop_data(file):
    try:
        # Try reading with different methods
        try:
            # First try: Standard read
            df = pd.read_excel(file, engine='openpyxl', header=None)
        except:
            # Second try: CSV
            file.seek(0)
            df = pd.read_csv(file, header=None, sep=None)  # Auto-detect sep for CSV/TSV
        
        # Clean completely empty rows
        df = df.dropna(how='all')
        
        # Find the header row (should contain "People", "Region", "Year", etc.)
        header_row_idx = 0
        for idx, row in df.iterrows():
            if 'People' in str(row.values).lower() or 'region' in str(row.values).lower():
                header_row_idx = idx
                break
        
        # Skip first row if it's just "People" and set next row as header
        if header_row_idx < len(df) - 1:
            df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
        
        # Manually set column names based on your data structure
        expected_cols = ['People', 'Metric', 'Channel', 'Region',
                         'US_2023', 'EUUK_2023', 'ANZ_2023', 'Others_2023',
                         'blank1',
                         'US_2024', 'EUUK_2024', 'ANZ_2024', 'Others_2024',
                         'blank2',
                         'US_2025', 'EUUK_2025', 'ANZ_2025', 'Others_2025']
        
        # Adjust column count safely
        num_cols = len(df.columns)
        num_expected = len(expected_cols)
        if num_cols >= num_expected:
            df.columns = expected_cols + [f'extra_col_{i}' for i in range(num_cols - num_expected)]
        else:
            df.columns = expected_cols[:num_cols]
        
        # Normalize column names (replace / with nothing for EU/UK -> EUUK)
        df.columns = [col.replace('/', '') for col in df.columns]
        
        # Remove blank columns
        df = df.loc[:, ~df.columns.str.contains('blank')]
        
        # Clean data - remove rows where People is empty or contains header text
        df = df[df.iloc[:, 0].notna()]
        df = df[~df.iloc[:, 0].astype(str).str.lower().str.contains('people|region|year', na=False)]
        
        # Rename first 3 columns
        df.columns.values[0] = 'People'
        if len(df.columns) > 1:
            df.columns.values[1] = 'Metric'
        if len(df.columns) > 2:
            df.columns.values[2] = 'Channel'
        
        # Clean People column - remove leading numbers
        df['People'] = df['People'].astype(str).str.replace(r'^\d+\s*', '', regex=True).str.strip()
        df = df[df['People'] != '']
        df = df[df['People'] != 'nan']
        
        # Fill forward the People and Channel columns (for merged cells)
        df['People'] = df['People'].replace('', None)
        df['People'] = df['People'].ffill()
        
        if 'Channel' in df.columns:
            df['Channel'] = df['Channel'].replace('', None)
            df['Channel'] = df['Channel'].ffill()
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col not in ['People', 'Channel', 'Metric']]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('-', '0'), errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar - File Upload
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin-bottom: 1rem; text-align: center;">
    <h3 style="color: white; margin: 0;">📁 Upload AOP Data</h3>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"], key="aop_file")

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        df = load_aop_data(uploaded_file)
        if df is None or df.empty:
            st.stop()
        st.session_state.df = df
    st.sidebar.success(f"Loaded {len(st.session_state.df)} records!")
elif 'df' in st.session_state:
    df = st.session_state.df
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%); border-radius: 20px; border: 2px dashed #667eea; margin: 2rem 0;">
        <h2 style="color: #4facfe;">📊 Upload Your AOP Data</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">Upload your sales data to unlock insights</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# Sidebar Navigation
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
    <h3 style="color: white; margin: 0;">🧭 Navigation</h3>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox("Select Page", ["🏠 Dashboard Home", "📊 Year Comparison", "👑 Boss Dashboard", "👥 Detailed Performers"], label_visibility="collapsed")

# Filters
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
    <h3 style="color: white; margin: 0;">⚙️ Filters</h3>
</div>
""", unsafe_allow_html=True)

# Get unique values for filters
people_list = sorted(df['People'].unique())
channel_list = sorted(df['Channel'].unique())
metric_list = sorted(df['Metric'].unique())

selected_people = st.sidebar.multiselect("👥 Select People", options=people_list, default=people_list)
selected_channel = st.sidebar.multiselect("📺 Select Channel", options=channel_list, default=channel_list)
selected_metric = st.sidebar.multiselect("📊 Select Metric", options=metric_list, default=metric_list)
selected_regions = st.sidebar.multiselect("🌍 Select Regions", options=regions, default=regions)
selected_years = st.sidebar.multiselect("📅 Select Years", options=years, default=years)

# Apply row filters
filtered_df = df[
    (df['People'].isin(selected_people)) &
    (df['Channel'].isin(selected_channel)) &
    (df['Metric'].isin(selected_metric))
].copy()

# Function to format currency
def format_currency(value):
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.0f}"

def format_number(value):
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.0f}"

def compute_metrics(df_subset, sel_years, sel_regions):
    year_region_cols = [col for col in df_subset.columns if any(year in col for year in sel_years) and any(region in col for region in sel_regions)]
    total_opps = df_subset[df_subset['Metric'] == 'Opps created'][year_region_cols].sum().sum()
    total_pipeline = df_subset[df_subset['Metric'] == 'Pipeline'][year_region_cols].sum().sum()
    total_closure = df_subset[df_subset['Metric'] == 'Closure'][year_region_cols].sum().sum()
    conversion_rate = (total_closure / total_pipeline * 100) if total_pipeline > 0 else 0
    avg_deal_size = total_closure / total_opps if total_opps > 0 else 0
    
    return total_opps, total_pipeline, total_closure, conversion_rate, avg_deal_size

# Dashboard Pages
if page == "🏠 Dashboard Home":
    st.markdown('<div class="section-header">📈 Performance Overview</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    year_region_cols = [col for col in filtered_df.columns if any(year in col for year in selected_years) and any(region in col for region in selected_regions)]
    
    with col1:
        total_opps = filtered_df[filtered_df['Metric'] == 'Opps created'][year_region_cols].sum().sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{format_number(total_opps)}</div>
            <div class="metric-label">🎯 Total Opportunities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_pipeline = filtered_df[filtered_df['Metric'] == 'Pipeline'][year_region_cols].sum().sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{format_currency(total_pipeline)}</div>
            <div class="metric-label">💰 Total Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_closure = filtered_df[filtered_df['Metric'] == 'Closure'][year_region_cols].sum().sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{format_currency(total_closure)}</div>
            <div class="metric-label">✅ Total Closures</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        conversion_rate = (total_closure / total_pipeline * 100) if total_pipeline > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{conversion_rate:.1f}%</div>
            <div class="metric-label">📊 Conversion Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_deal_size = total_closure / total_opps if total_opps > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{format_currency(avg_deal_size)}</div>
            <div class="metric-label">💵 Avg Deal Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance by Region
    st.markdown('<div class="section-header">🌍 Regional Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pipeline by Region
        region_data = []
        for region in selected_regions:
            region_cols = [col for col in filtered_df.columns if region in col and any(year in col for year in selected_years)]
            pipeline_val = filtered_df[filtered_df['Metric'] == 'Pipeline'][region_cols].sum().sum()
            closure_val = filtered_df[filtered_df['Metric'] == 'Closure'][region_cols].sum().sum()
            region_data.append({
                'Region': region,
                'Pipeline': pipeline_val,
                'Closure': closure_val
            })
        
        region_df = pd.DataFrame(region_data)
        
        fig_region = px.bar(
            region_df,
            x='Region',
            y=['Pipeline', 'Closure'],
            title="Pipeline vs Closure by Region",
            barmode='group',
            color_discrete_sequence=['#667eea', '#4facfe']
        )
        fig_region.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig_region.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        # Opportunities by Region
        opp_data = []
        for region in selected_regions:
            region_cols = [col for col in filtered_df.columns if region in col and any(year in col for year in selected_years)]
            opp_val = filtered_df[filtered_df['Metric'] == 'Opps created'][region_cols].sum().sum()
            opp_data.append({'Region': region, 'Opportunities': opp_val})
        
        opp_df = pd.DataFrame(opp_data)
        
        fig_opp = px.pie(
            opp_df,
            values='Opportunities',
            names='Region',
            title="Opportunities Distribution by Region",
            color_discrete_sequence=['#667eea', '#764ba2', '#4facfe', '#00f2fe', '#ff6b6b']
        )
        fig_opp.update_traces(textposition='inside', textinfo='percent+label+value')
        fig_opp.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        st.plotly_chart(fig_opp, use_container_width=True)
    
    # Performance by Channel
    st.markdown('<div class="section-header">📺 Channel Performance</div>', unsafe_allow_html=True)
    
    channel_data = []
    for channel in filtered_df['Channel'].unique():
        channel_df = filtered_df[filtered_df['Channel'] == channel]
        pipeline = channel_df[channel_df['Metric'] == 'Pipeline'][year_region_cols].sum().sum()
        closure = channel_df[channel_df['Metric'] == 'Closure'][year_region_cols].sum().sum()
        opps = channel_df[channel_df['Metric'] == 'Opps created'][year_region_cols].sum().sum()
        
        channel_data.append({
            'Channel': channel,
            'Pipeline': pipeline,
            'Closure': closure,
            'Opportunities': opps
        })
    
    channel_df = pd.DataFrame(channel_data)
    
    fig_channel = px.bar(
        channel_df,
        x='Channel',
        y=['Pipeline', 'Closure'],
        title="Pipeline vs Closure by Channel",
        barmode='group',
        color_discrete_sequence=['#764ba2', '#ff6b6b']
    )
    fig_channel.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
    fig_channel.update_layout(
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font_size=18
    )
    fig_channel.update_xaxes(tickangle=45)
    st.plotly_chart(fig_channel, use_container_width=True)
    
    # Top Performers
    st.markdown('<div class="section-header">🏆 Top Performers</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top by Closure
        people_closure = []
        for person in filtered_df['People'].unique():
            person_df = filtered_df[filtered_df['People'] == person]
            closure = person_df[person_df['Metric'] == 'Closure'][year_region_cols].sum().sum()
            people_closure.append({'Person': person, 'Closure': closure})
        
        closure_df = pd.DataFrame(people_closure).sort_values('Closure', ascending=False).head(10)
        
        fig_top = px.bar(
            closure_df,
            x='Person',
            y='Closure',
            title="Top 10 by Closure",
            color='Closure',
            color_continuous_scale='Viridis'
        )
        fig_top.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig_top.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        fig_top.update_xaxes(tickangle=45)
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Top by Opportunities
        people_opps = []
        for person in filtered_df['People'].unique():
            person_df = filtered_df[filtered_df['People'] == person]
            opps = person_df[person_df['Metric'] == 'Opps created'][year_region_cols].sum().sum()
            people_opps.append({'Person': person, 'Opportunities': opps})
        
        opps_df = pd.DataFrame(people_opps).sort_values('Opportunities', ascending=False).head(10)
        
        fig_top_opps = px.bar(
            opps_df,
            x='Person',
            y='Opportunities',
            title="Top 10 by Opportunities",
            color='Opportunities',
            color_continuous_scale='Plasma'
        )
        fig_top_opps.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig_top_opps.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        fig_top_opps.update_xaxes(tickangle=45)
        st.plotly_chart(fig_top_opps, use_container_width=True)

elif page == "📊 Year Comparison":
    st.markdown('<div class="section-header">📊 Year-wise Performance Comparison</div>', unsafe_allow_html=True)
    
    # Year selector
    comparison_years = st.multiselect(
        "Select Years to Compare",
        options=years,
        default=years
    )
    
    if len(comparison_years) < 2:
        st.warning("Please select at least 2 years for comparison")
    else:
        # Year-wise metrics
        st.markdown("### 📈 Key Metrics Comparison")
        
        year_metrics = []
        for year in comparison_years:
            year_cols = [col for col in filtered_df.columns if year in col and any(region in col for region in selected_regions)]
            
            opps = filtered_df[filtered_df['Metric'] == 'Opps created'][year_cols].sum().sum()
            pipeline = filtered_df[filtered_df['Metric'] == 'Pipeline'][year_cols].sum().sum()
            closure = filtered_df[filtered_df['Metric'] == 'Closure'][year_cols].sum().sum()
            conversion = (closure / pipeline * 100) if pipeline > 0 else 0
            
            year_metrics.append({
                'Year': year,
                'Opportunities': opps,
                'Pipeline': pipeline,
                'Closure': closure,
                'Conversion Rate (%)': conversion
            })
        
        metrics_df = pd.DataFrame(year_metrics)
        st.dataframe(
            metrics_df.style
            .format({
                'Opportunities': format_number,
                'Pipeline': format_currency,
                'Closure': format_currency,
                'Conversion Rate (%)': '{:.2f}%'
            })
            .background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        # Visual Comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            # Opportunities Trend
            fig_opps = px.bar(
                metrics_df,
                x='Year',
                y='Opportunities',
                title="Opportunities Trend",
                color='Year',
                color_discrete_sequence=['#667eea', '#764ba2', '#4facfe']
            )
            fig_opps.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
            fig_opps.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                showlegend=False
            )
            st.plotly_chart(fig_opps, use_container_width=True)
        
        with col2:
            # Pipeline vs Closure
            fig_pipeline = go.Figure()
            fig_pipeline.add_trace(go.Bar(
                x=metrics_df['Year'],
                y=metrics_df['Pipeline'],
                name='Pipeline',
                marker_color='#667eea',
                text=metrics_df['Pipeline'],
                textposition='outside',
                texttemplate='%{text:,.0f}'
            ))
            fig_pipeline.add_trace(go.Bar(
                x=metrics_df['Year'],
                y=metrics_df['Closure'],
                name='Closure',
                marker_color='#4facfe',
                text=metrics_df['Closure'],
                textposition='outside',
                texttemplate='%{text:,.0f}'
            ))
            fig_pipeline.update_layout(
                title="Pipeline vs Closure Trend",
                barmode='group',
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=18
            )
            st.plotly_chart(fig_pipeline, use_container_width=True)
        
        # Conversion Rate Trend
        fig_conversion = px.line(
            metrics_df,
            x='Year',
            y='Conversion Rate (%)',
            title="Conversion Rate Trend",
            markers=True,
            line_shape='spline'
        )
        fig_conversion.update_traces(
            line_color='#00f2fe', 
            marker=dict(size=12, color='#4facfe'),
            texttemplate='%{y:.1f}%',
            textposition="top center"
        )
        fig_conversion.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        st.plotly_chart(fig_conversion, use_container_width=True)
        
        # Regional Comparison
        st.markdown("### 🌍 Regional Performance by Year")
        
        for region in selected_regions:
            region_year_data = []
            for year in comparison_years:
                region_col = f'{region}_{year}'
                if region_col in filtered_df.columns:
                    closure = filtered_df[filtered_df['Metric'] == 'Closure'][region_col].sum()
                    region_year_data.append({
                        'Year': year,
                        'Region': region,
                        'Closure': closure
                    })
            
            if region_year_data:
                region_df = pd.DataFrame(region_year_data)
                if region_df['Closure'].sum() > 0:  # Only show if there's data
                    fig_region = px.bar(
                        region_df,
                        x='Year',
                        y='Closure',
                        title=f"{region} - Closure by Year",
                        color='Year',
                        color_discrete_sequence=['#667eea', '#764ba2', '#4facfe']
                    )
                    fig_region.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                    fig_region.update_layout(
                        font=dict(color='white'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title_font_size=16
                    )
                    st.plotly_chart(fig_region, use_container_width=True)
        
        # Channel Comparison
        st.markdown("### 📺 Channel Performance by Year")
        
        channel_year_data = []
        for channel in filtered_df['Channel'].unique():
            for year in comparison_years:
                year_cols = [col for col in filtered_df.columns if year in col and any(region in col for region in selected_regions)]
                channel_df_temp = filtered_df[filtered_df['Channel'] == channel]
                closure = channel_df_temp[channel_df_temp['Metric'] == 'Closure'][year_cols].sum().sum()
                channel_year_data.append({
                    'Channel': channel,
                    'Year': year,
                    'Closure': closure
                })
        
        channel_year_df = pd.DataFrame(channel_year_data)
        
        fig_channel_year = px.bar(
            channel_year_df,
            x='Channel',
            y='Closure',
            color='Year',
            barmode='group',
            title="Channel Performance Across Years",
            color_discrete_sequence=['#667eea', '#764ba2', '#4facfe']
        )
        fig_channel_year.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig_channel_year.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        fig_channel_year.update_xaxes(tickangle=45)
        st.plotly_chart(fig_channel_year, use_container_width=True)

elif page == "👑 Boss Dashboard":
    st.markdown('<div class="section-header">👑 Executive Summary Dashboard</div>', unsafe_allow_html=True)
    
    # Executive Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    year_region_cols = [col for col in filtered_df.columns if any(year in col for year in selected_years) and any(region in col for region in selected_regions)]
    
    with col1:
        total_opps = filtered_df[filtered_df['Metric'] == 'Opps created'][year_region_cols].sum().sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{format_number(total_opps)}</div>
            <div class="metric-label">🎯 Total Opportunities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_pipeline = filtered_df[filtered_df['Metric'] == 'Pipeline'][year_region_cols].sum().sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{format_currency(total_pipeline)}</div>
            <div class="metric-label">💰 Total Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_closure = filtered_df[filtered_df['Metric'] == 'Closure'][year_region_cols].sum().sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{format_currency(total_closure)}</div>
            <div class="metric-label">✅ Total Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        conversion = (total_closure / total_pipeline * 100) if total_pipeline > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{conversion:.1f}%</div>
            <div class="metric-label">📊 Win Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Growth Metrics
    st.markdown("### 📈 Year-over-Year Growth")
    
    growth_data = []
    for year in selected_years:
        year_cols = [col for col in filtered_df.columns if year in col and any(region in col for region in selected_regions)]
        closure = filtered_df[filtered_df['Metric'] == 'Closure'][year_cols].sum().sum()
        growth_data.append({'Year': year, 'Revenue': closure})
    
    growth_df = pd.DataFrame(growth_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_growth = px.line(
            growth_df,
            x='Year',
            y='Revenue',
            title="Revenue Growth Trend",
            markers=True,
            line_shape='spline'
        )
        fig_growth.update_traces(
            line_color='#00f2fe', 
            marker=dict(size=15, color='#4facfe'),
            texttemplate='%{y:,.0f}',
            textposition="top center"
        )
        fig_growth.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    
    with col2:
        # Revenue by Region
        region_revenue = []
        for region in selected_regions:
            region_cols = [col for col in filtered_df.columns if region in col and any(year in col for year in selected_years)]
            revenue = filtered_df[filtered_df['Metric'] == 'Closure'][region_cols].sum().sum()
            region_revenue.append({'Region': region, 'Revenue': revenue})
        
        region_rev_df = pd.DataFrame(region_revenue)
        
        fig_region_rev = px.pie(
            region_rev_df,
            values='Revenue',
            names='Region',
            title="Revenue Distribution by Region",
            color_discrete_sequence=['#667eea', '#764ba2', '#4facfe', '#00f2fe', '#ff6b6b']
        )
        fig_region_rev.update_traces(textposition='inside', textinfo='percent+label+value')
        fig_region_rev.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18
        )
        st.plotly_chart(fig_region_rev, use_container_width=True)
    
    # Top Performers Summary
    st.markdown("### 🏆 Top Performers")
    
    people_summary = []
    for person in filtered_df['People'].unique():
        person_df = filtered_df[filtered_df['People'] == person]
        opps = person_df[person_df['Metric'] == 'Opps created'][year_region_cols].sum().sum()
        pipeline = person_df[person_df['Metric'] == 'Pipeline'][year_region_cols].sum().sum()
        closure = person_df[person_df['Metric'] == 'Closure'][year_region_cols].sum().sum()
        conversion = (closure / pipeline * 100) if pipeline > 0 else 0
        
        people_summary.append({
            'Person': person,
            'Opportunities': opps,
            'Pipeline': pipeline,
            'Revenue': closure,
            'Win Rate (%)': conversion
        })
    
    summary_df = pd.DataFrame(people_summary).sort_values('Revenue', ascending=False).head(10)
    st.dataframe(
        summary_df.style
        .format({
            'Opportunities': format_number,
            'Pipeline': format_currency,
            'Revenue': format_currency,
            'Win Rate (%)': '{:.2f}%'
        })
        .background_gradient(subset=['Revenue', 'Pipeline'], cmap='Blues'),
        use_container_width=True
    )
    
    # Channel Performance
    st.markdown("### 📺 Business Channel Performance")
    
    channel_summary = []
    for channel in filtered_df['Channel'].unique():
        channel_df = filtered_df[filtered_df['Channel'] == channel]
        opps = channel_df[channel_df['Metric'] == 'Opps created'][year_region_cols].sum().sum()
        pipeline = channel_df[channel_df['Metric'] == 'Pipeline'][year_region_cols].sum().sum()
        closure = channel_df[channel_df['Metric'] == 'Closure'][year_region_cols].sum().sum()
        
        channel_summary.append({
            'Channel': channel,
            'Opportunities': opps,
            'Pipeline': pipeline,
            'Revenue': closure
        })
    
    channel_sum_df = pd.DataFrame(channel_summary).sort_values('Revenue', ascending=False)
    
    fig_channel_sum = px.bar(
        channel_sum_df,
        x='Channel',
        y='Revenue',
        title="Revenue by Business Channel",
        color='Revenue',
        color_continuous_scale='Viridis'
    )
    fig_channel_sum.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
    fig_channel_sum.update_layout(
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font_size=18
    )
    fig_channel_sum.update_xaxes(tickangle=45)
    st.plotly_chart(fig_channel_sum, use_container_width=True)
    
    # Key Recommendations
    st.markdown("### 📝 Executive Recommendations")
    
    recommendations = [
        "🎯 **Focus on High-Performing Regions**: Allocate more resources to regions with highest conversion rates",
        "💰 **Pipeline Health**: Current pipeline indicates strong future revenue potential - maintain momentum",
        "👥 **Team Performance**: Top performers driving majority of revenue - consider scaling their best practices",
        "📈 **Growth Trajectory**: Year-over-year growth shows positive trend - continue investment in successful channels",
        "🌍 **Market Expansion**: Underperforming regions present opportunity for strategic initiatives"
    ]
    
    for rec in recommendations:
        st.markdown(f'<div style="background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #4facfe;">{rec}</div>', unsafe_allow_html=True)

elif page == "👥 Detailed Performers":
    st.markdown('<div class="section-header">👥 Detailed Performers by Channel</div>', unsafe_allow_html=True)
    
    # Detailed data preparation for channels
    detailed_data = []
    year_region_cols_global = [col for col in filtered_df.columns if any(year in col for year in selected_years) and any(region in col for region in selected_regions)]
    for person in sorted(filtered_df['People'].unique()):
        for channel in sorted(filtered_df['Channel'].unique()):
            person_channel_df = filtered_df[(filtered_df['People'] == person) & (filtered_df['Channel'] == channel)]
            if not person_channel_df.empty:
                opps = person_channel_df[person_channel_df['Metric'] == 'Opps created'][year_region_cols_global].sum().sum()
                pipeline = person_channel_df[person_channel_df['Metric'] == 'Pipeline'][year_region_cols_global].sum().sum()
                closure = person_channel_df[person_channel_df['Metric'] == 'Closure'][year_region_cols_global].sum().sum()
                win_rate = (closure / pipeline * 100) if pipeline > 0 else 0
                deal_size = closure / opps if opps > 0 else 0
                
                detailed_data.append({
                    'Person': person,
                    'Channel': channel,
                    'Opportunities': opps,
                    'Pipeline': pipeline,
                    'Revenue': closure,
                    'Win Rate (%)': win_rate,
                    'Deal Size': deal_size
                })
    
    channel_detailed_df = pd.DataFrame(detailed_data)
    
    # New: Detailed data preparation for Person-Year-Region
    person_year_region_data = []
    for person in sorted(filtered_df['People'].unique()):
        for year in selected_years:
            for region in selected_regions:
                person_year_region_df = filtered_df[(filtered_df['People'] == person)]
                year_region_col = f'{region}_{year}'
                if year_region_col in person_year_region_df.columns:
                    opps = person_year_region_df[person_year_region_df['Metric'] == 'Opps created'][year_region_col].sum()
                    pipeline = person_year_region_df[person_year_region_df['Metric'] == 'Pipeline'][year_region_col].sum()
                    closure = person_year_region_df[person_year_region_df['Metric'] == 'Closure'][year_region_col].sum()
                    win_rate = (closure / pipeline * 100) if pipeline > 0 else 0
                    deal_size = closure / opps if opps > 0 else 0
                    
                    person_year_region_data.append({
                        'Person': person,
                        'Year': year,
                        'Region': region,
                        'Opportunities': opps,
                        'Pipeline': pipeline,
                        'Revenue': closure,
                        'Win Rate (%)': win_rate,
                        'Deal Size': deal_size
                    })
    
    person_year_region_df = pd.DataFrame(person_year_region_data)
    
    # New: Detailed data preparation for Region-Year (Overall)
    region_year_data = []
    for year in selected_years:
        for region in selected_regions:
            year_region_col = f'{region}_{year}'
            if year_region_col in filtered_df.columns:
                opps = filtered_df[filtered_df['Metric'] == 'Opps created'][year_region_col].sum()
                pipeline = filtered_df[filtered_df['Metric'] == 'Pipeline'][year_region_col].sum()
                closure = filtered_df[filtered_df['Metric'] == 'Closure'][year_region_col].sum()
                win_rate = (closure / pipeline * 100) if pipeline > 0 else 0
                deal_size = closure / opps if opps > 0 else 0
                
                region_year_data.append({
                    'Region': region,
                    'Year': year,
                    'Opportunities': opps,
                    'Pipeline': pipeline,
                    'Revenue': closure,
                    'Win Rate (%)': win_rate,
                    'Deal Size': deal_size
                })
    
    region_year_df = pd.DataFrame(region_year_data)
    
    if not channel_detailed_df.empty or not person_year_region_df.empty or not region_year_df.empty:
        # Person selector
        selected_person = st.selectbox("Select a Person for Detailed Channel Breakdown:", options=['Overall'] + sorted(filtered_df['People'].unique()))
        
        # Dynamic key metrics based on selection
        if selected_person == 'Overall':
            df_subset = filtered_df
            title = "Overall Metrics"
            total_opps, total_pipeline, total_closure, conversion_rate, avg_deal_size = compute_metrics(df_subset, selected_years, selected_regions)
        else:
            df_subset = filtered_df[filtered_df['People'] == selected_person]
            title = f"{selected_person} Metrics"
            total_opps, total_pipeline, total_closure, conversion_rate, avg_deal_size = compute_metrics(df_subset, selected_years, selected_regions)
        
        st.markdown(f"### {title}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{format_number(total_opps)}</div>
                <div class="metric-label">🎯 Total Opportunities</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{format_currency(total_pipeline)}</div>
                <div class="metric-label">💰 Total Pipeline</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{format_currency(total_closure)}</div>
                <div class="metric-label">✅ Total Closures</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{conversion_rate:.1f}%</div>
                <div class="metric-label">📊 Conversion Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{format_currency(avg_deal_size)}</div>
                <div class="metric-label">💵 Avg Deal Size</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Channel Breakdown (existing)
        if selected_person != 'Overall':
            # Detailed view for selected person
            person_data = channel_detailed_df[channel_detailed_df['Person'] == selected_person].sort_values('Channel')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart for Revenue by Channel
                fig_bar = px.bar(
                    person_data,
                    x='Channel',
                    y='Revenue',
                    title=f"{selected_person} - Revenue by Channel",
                    color='Revenue',
                    color_continuous_scale='Blues'
                )
                fig_bar.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                fig_bar.update_layout(
                    font=dict(color='white'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title_font_size=16
                )
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Table for detailed metrics (channels)
                st.markdown(f"### 📋 {selected_person} - Channel Details")
                st.dataframe(
                    person_data[['Channel', 'Opportunities', 'Pipeline', 'Revenue', 'Win Rate (%)', 'Deal Size']].style
                    .format({
                        'Opportunities': format_number,
                        'Pipeline': format_currency,
                        'Revenue': format_currency,
                        'Win Rate (%)': '{:.2f}%',
                        'Deal Size': format_currency
                    })
                    .background_gradient(subset=['Revenue'], cmap='Blues'),
                    use_container_width=True,
                    hide_index=True
                )
        
        # New: Year-wise Breakdown for Selected Person
        st.markdown('<div class="section-header">📅 Year-wise Performance</div>', unsafe_allow_html=True)
        person_year_data = person_year_region_df[person_year_region_df['Person'] == selected_person].groupby('Year').agg({
            'Opportunities': 'sum',
            'Pipeline': 'sum',
            'Revenue': 'sum',
            'Win Rate (%)': 'mean',
            'Deal Size': 'mean'
        }).reset_index()
        
        if not person_year_data.empty:
            st.markdown("### 📋 Detailed Metrics by Year")
            st.dataframe(
                person_year_data.style
                .format({
                    'Opportunities': format_number,
                    'Pipeline': format_currency,
                    'Revenue': format_currency,
                    'Win Rate (%)': '{:.2f}%',
                    'Deal Size': format_currency
                })
                .background_gradient(subset=['Revenue', 'Pipeline'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )
            
            # Year-wise Chart
            fig_year = px.bar(
                person_year_data,
                x='Year',
                y=['Pipeline', 'Revenue'],
                title=f"{selected_person} - Revenue vs Pipeline by Year",
                barmode='group',
                color_discrete_sequence=['#667eea', '#4facfe']
            )
            fig_year.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
            fig_year.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=16
            )
            st.plotly_chart(fig_year, use_container_width=True)
        
        # New: Region-wise Breakdown for Selected Person
        st.markdown('<div class="section-header">🌍 Region-wise Performance</div>', unsafe_allow_html=True)
        person_region_data = person_year_region_df[person_year_region_df['Person'] == selected_person].groupby('Region').agg({
            'Opportunities': 'sum',
            'Pipeline': 'sum',
            'Revenue': 'sum',
            'Win Rate (%)': 'mean',
            'Deal Size': 'mean'
        }).reset_index()
        
        if not person_region_data.empty:
            st.markdown("### 📋 Detailed Metrics by Region")
            st.dataframe(
                person_region_data.style
                .format({
                    'Opportunities': format_number,
                    'Pipeline': format_currency,
                    'Revenue': format_currency,
                    'Win Rate (%)': '{:.2f}%',
                    'Deal Size': format_currency
                })
                .background_gradient(subset=['Revenue', 'Pipeline'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )
            
            # Region-wise Chart
            fig_region = px.pie(
                person_region_data,
                values='Revenue',
                names='Region',
                title=f"{selected_person} - Revenue by Region",
                color_discrete_sequence=['#667eea', '#764ba2', '#4facfe', '#00f2fe']
            )
            fig_region.update_traces(textposition='inside', textinfo='percent+label+value')
            fig_region.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16
            )
            st.plotly_chart(fig_region, use_container_width=True)
        
        # New: Overall Region-Year Table and Chart
        st.markdown('<div class="section-header">🌍📅 Overall Region-Year Performance</div>', unsafe_allow_html=True)
        if not region_year_df.empty:
            st.markdown("### 📋 Detailed Metrics by Region and Year")
            st.dataframe(
                region_year_df.style
                .format({
                    'Opportunities': format_number,
                    'Pipeline': format_currency,
                    'Revenue': format_currency,
                    'Win Rate (%)': '{:.2f}%',
                    'Deal Size': format_currency
                })
                .background_gradient(subset=['Revenue', 'Pipeline'], cmap='Blues'),
                use_container_width=True,
                hide_index=False
            )
            
            # Region-Year Chart
            fig_region_year = px.bar(
                region_year_df,
                x='Region',
                y='Revenue',
                color='Year',
                barmode='group',
                title="Overall Revenue by Region and Year",
                color_discrete_sequence=['#667eea', '#764ba2', '#4facfe']
            )
            fig_region_year.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
            fig_region_year.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=16
            )
            fig_region_year.update_xaxes(tickangle=45)
            st.plotly_chart(fig_region_year, use_container_width=True)
        
        # Option for Overall Channel Table
        if st.checkbox("📊 Show Overall Detailed Channel Table (All Persons & Channels)"):
            st.markdown("### 📋 Complete Channel Breakdown Table")
            st.dataframe(
                channel_detailed_df.sort_values(['Person', 'Channel']).style
                .format({
                    'Opportunities': format_number,
                    'Pipeline': format_currency,
                    'Revenue': format_currency,
                    'Win Rate (%)': '{:.2f}%',
                    'Deal Size': format_currency
                })
                .background_gradient(subset=['Revenue', 'Pipeline', 'Deal Size'], cmap='Blues'),
                use_container_width=True,
                hide_index=False
            )
        
        # Option for Overall Person-Year-Region Table
        if st.checkbox("📊 Show Overall Person-Year-Region Table"):
            st.markdown("### 📋 Complete Person-Year-Region Breakdown Table")
            st.dataframe(
                person_year_region_df.sort_values(['Person', 'Year', 'Region']).style
                .format({
                    'Opportunities': format_number,
                    'Pipeline': format_currency,
                    'Revenue': format_currency,
                    'Win Rate (%)': '{:.2f}%',
                    'Deal Size': format_currency
                })
                .background_gradient(subset=['Revenue', 'Pipeline', 'Deal Size'], cmap='Blues'),
                use_container_width=True,
                hide_index=False
            )
    else:
        st.warning("No data available for the selected filters.")

# Footers
st.markdown("""
<div style="background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5a 100%); padding: 2rem; border-radius: 20px; text-align: center; margin-top: 3rem; border: 1px solid rgba(102, 126, 234, 0.3);">
    <p style="margin: 0.5rem 0; color: rgba(255,255,255,0.8);">🎯 Powered by AOP Analytics Dashboard</p>
    <p style="margin: 0.5rem 0; color: rgba(255,255,255,0.6);">Data-Driven Sales Intelligence</p>
</div>
""", unsafe_allow_html=True)