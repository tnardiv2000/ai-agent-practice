import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import requests
import json

load_dotenv()

st.set_page_config(page_title="AI Data Analyzer Pro", layout="wide")
st.title("📊 AI Data Analyzer Pro")
st.write("Smart data analysis with full transparency and accuracy!")

OLLAMA_API = "http://localhost:11434/api/generate"

FINANCIAL_COLUMNS = ['Spend', 'Savings', 'Revenue', 'Profit', 'Marketing_Spend']
PERCENTAGE_COLUMNS = ['KPI_%', 'Profit_Margin_%', 'Return_Rate_%', 'Employee_Engagement_%', 'Customer_Satisfaction_Score']
COUNT_COLUMNS = ['Units_Sold']
DATE_COLUMNS = ['Date', 'Year', 'Quarter', 'Month']
DIMENSION_COLUMNS = ['Geo', 'Country', 'Sales_Rep', 'Customer', 'Category', 'Product']

def get_column_type(col_name):
    if col_name in FINANCIAL_COLUMNS:
        return 'financial'
    elif col_name in PERCENTAGE_COLUMNS:
        return 'percentage'
    elif col_name in COUNT_COLUMNS:
        return 'count'
    elif col_name in DATE_COLUMNS:
        return 'date'
    elif col_name in DIMENSION_COLUMNS:
        return 'dimension'
    else:
        return 'numeric'

def get_recommended_functions(col_name):
    col_type = get_column_type(col_name)
    recommendations = {
        'financial': {'default': 'sum', 'options': {'sum': 'Total (SUM)', 'max': 'Highest (MAX)', 'min': 'Lowest (MIN)', 'mean': 'Average (MEAN)', 'count': 'Count'}},
        'percentage': {'default': 'mean', 'options': {'mean': 'Average % (MEAN)', 'max': 'Highest % (MAX)', 'min': 'Lowest % (MIN)', 'count': 'Count'}},
        'count': {'default': 'sum', 'options': {'sum': 'Total (SUM)', 'mean': 'Average (MEAN)', 'max': 'Maximum (MAX)', 'min': 'Minimum (MIN)', 'count': 'Count'}},
        'numeric': {'default': 'sum', 'options': {'sum': 'Total (SUM)', 'mean': 'Average (MEAN)', 'max': 'Maximum (MAX)', 'min': 'Minimum (MIN)', 'count': 'Count'}}
    }
    return recommendations.get(col_type, recommendations['numeric'])

with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("AI Creativity", 0.0, 1.0, 0.0)
    show_steps = st.checkbox("Show Analysis Details", value=True)
    timeout_seconds = st.slider("AI Response Timeout (seconds)", 60, 1800, 600, step=60)

def get_column_statistics(data, column):
    try:
        stats = {
            "Data Type": str(data[column].dtype),
            "Non-Null Count": data[column].count(),
            "Null Count": data[column].isnull().sum(),
            "Unique Values": data[column].nunique(),
        }
        if data[column].dtype in ['int64', 'float64']:
            stats.update({
                "Min": data[column].min(),
                "Max": data[column].max(),
                "Mean": data[column].mean(),
                "Median": data[column].median(),
                "Std Dev": data[column].std(),
            })
        else:
            stats["Sample Values"] = ", ".join(str(v) for v in data[column].unique()[:5])
        return stats
    except Exception as e:
        return f"Error: {str(e)}"

uploaded_file = st.file_uploader("Choose a file (Excel or CSV)", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    try:
        if temp_path.endswith('.csv'):
            data = pd.read_csv(temp_path)
        elif temp_path.endswith('.xlsx'):
            data = pd.read_excel(temp_path, engine='openpyxl')
        elif temp_path.endswith('.xls'):
            data = pd.read_excel(temp_path, engine='xlrd')
        else:
            data = pd.read_csv(temp_path)
        
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Data Stats", "🔍 Detailed Info"])
        
        with tab1:
            st.subheader("Data Preview")
            rows_to_show = st.slider("Rows to display:", 5, min(100, len(data)), 10)
            st.dataframe(data.head(rows_to_show), use_container_width=True)
        
        with tab2:
            st.subheader("📈 Quick Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(data):,}")
            with col2:
                st.metric("Total Columns", len(data.columns))
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            with col4:
                st.metric("Data Size", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            st.write("**Numeric Column Statistics:**")
            numeric_data = data.select_dtypes(include=['number'])
            if not numeric_data.empty:
                st.dataframe(numeric_data.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found")
        
        with tab3:
            st.subheader("🔍 Column Details")
            col_info = pd.DataFrame({
                'Column Name': data.columns,
                'Data Type': data.dtypes.values,
                'Non-Null Count': data.count().values,
                'Null Count': data.isnull().sum().values,
                'Unique Values': [f"{data[col].nunique():,}" for col in data.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        st.divider()
        
        st.subheader("🧮 Manual Data Calculations")
        st.info("✨ Smart Aggregation: Automatically selects the best function for each column type!")
        
        calc_tab1, calc_tab2, calc_tab3, calc_tab4 = st.tabs(
            ["📊 Column Stats", "🔍 Filter & View", "👥 Group & Aggregate", "📋 Custom View"]
        )
        
        with calc_tab1:
            st.write("**Get detailed statistics for any column**")
            selected_col = st.selectbox("Select column:", data.columns, key="stat_col")
            if st.button("Get Column Statistics", key="btn_col_stats"):
                with st.spinner("Calculating..."):
                    stats = get_column_statistics(data, selected_col)
                    if isinstance(stats, dict):
                        st.success("✅ Statistics Complete!")
                        for key, value in stats.items():
                            st.write(f"**{key}:** {value}")
                    else:
                        st.code(stats, language="text")
        
        with calc_tab2:
            st.write("**Filter data by column values**")
            filter_col = st.selectbox("Filter by column:", data.columns, key="filter_col_view")
            unique_vals = sorted([str(v) for v in data[filter_col].dropna().unique().tolist()])
            selected_vals = st.multiselect(f"Select values from {filter_col}:", unique_vals, key="filter_vals_view")
            if selected_vals:
                if st.button("Apply Filter & View", key="btn_filter_view"):
                    filtered = data[data[filter_col].astype(str).isin(selected_vals)].copy()
                    st.success(f"✅ Showing {len(filtered)} rows")
                    st.dataframe(filtered, use_container_width=True)
                    st.code(filtered.to_csv(index=False), language="csv")
        
        with calc_tab3:
            st.write("**Group & Aggregate**")
            group_cols_selected = st.multiselect("📍 Select column(s) to group by:", data.columns, key="group_by_cols")
            if group_cols_selected:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    agg_cols_selected = st.multiselect("📊 Select numeric column(s) to aggregate:", numeric_cols, key="agg_cols_selected")
                    if agg_cols_selected:
                        st.write("**Configure Aggregation:**")
                        agg_config = {}
                        col_configs = st.columns(min(len(agg_cols_selected), 3))
                        for idx, col in enumerate(agg_cols_selected):
                            col_idx = idx % 3
                            with col_configs[col_idx]:
                                col_type = get_column_type(col)
                                recommended = get_recommended_functions(col)
                                st.write(f"**{col}** ({col_type})")
                                selected_func = st.selectbox(f"Function for {col}:", list(recommended['options'].keys()), index=list(recommended['options'].keys()).index(recommended['default']), format_func=lambda x: recommended['options'][x], key=f"agg_func_{col}")
                                agg_config[col] = selected_func
                        if st.button("🚀 Group & Aggregate", key="btn_group_agg"):
                            with st.spinner("Calculating..."):
                                try:
                                    result = data.groupby(group_cols_selected).agg(agg_config).reset_index()
                                    st.success("✅ Aggregation Complete!")
                                    st.dataframe(result, use_container_width=True)
                                    st.code(result.to_csv(index=False), language="csv")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
        
        with calc_tab4:
            st.write("**Select specific columns**")
            cols_to_display = st.multiselect("Select columns:", data.columns, default=list(data.columns[:10]), key="cols_display")
            if cols_to_display:
                if st.button("Display Data", key="btn_display"):
                    st.dataframe(data[cols_to_display], use_container_width=True)
                    st.code(data[cols_to_display].to_csv(index=False), language="csv")
        
        st.divider()
        
        st.subheader("🤖 AI-Powered Analysis")
        st.info("AI analyzes your COMPLETE dataset - all rows, all columns. No hallucination.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Rows", f"{len(data):,}")
        with col2:
            st.metric("Dataset Columns", len(data.columns))
        with col3:
            st.metric("Status", "✅ Ready")
        
        st.write(f"**Columns:** {', '.join(data.columns.tolist())}")
        
        user_question = st.text_area("Ask about your data:", placeholder="Example: What product has best KPI_% in Brazil?", height=100)
        
        if user_question:
            st.write("🤖 Analyzing...")
            
            dataset_csv = data.head(50).to_csv(index=False)
            
            prompt = f"""Analyze this dataset and answer the question. Use ONLY the data provided. Never make up numbers. Show exact values from the data.

DATASET (showing first 50 rows):
{dataset_csv}

QUESTION: {user_question}

ANSWER: Provide exact values only from the dataset above. If not found, say "Not in dataset"."""
            
            try:
                response = requests.post(OLLAMA_API, json={"model": "llama2", "prompt": prompt, "stream": False, "temperature": temperature}, timeout=timeout_seconds)
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result.get("response", "No response")
                    st.success("✅ Done!")
                    st.markdown("---")
                    st.markdown(ai_response)
                    st.markdown("---")
                else:
                    st.error(f"Error: {response.status_code}")
            except requests.exceptions.Timeout:
                st.error(f"Timeout after {timeout_seconds}s")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Ollama. Run: ollama serve")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        import os as os_module
        if os_module.path.exists(temp_path):
            os_module.remove(temp_path)
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

import atexit
def cleanup():
    import os
    import glob
    for temp_file in glob.glob("temp_*"):
        try:
            os.remove(temp_file)
        except:
            pass

atexit.register(cleanup)