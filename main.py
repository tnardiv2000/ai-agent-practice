import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import requests
import json

load_dotenv()

st.set_page_config(page_title="AI Data Analyzer Pro", layout="wide")
st.title("📊 AI Data Analyzer Pro")
st.write("Smart data analysis with AI understanding + 100% accurate Python calculations!")

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
    temperature = st.slider("AI Creativity", 0.0, 1.0, 0.3)
    show_steps = st.checkbox("Show AI Reasoning", value=True)
    timeout_seconds = st.slider("AI Response Timeout (seconds)", 60, 1800, 600, step=60)
    if st.button("🔄 Clear All Cache"):
        st.cache_data.clear()
        st.rerun()

def get_column_statistics(data, column):
    try:
        stats = {
            "Data Type": str(data[column].dtype),
            "Non-Null Count": int(data[column].count()),
            "Null Count": int(data[column].isnull().sum()),
            "Unique Values": int(data[column].nunique()),
        }
        if data[column].dtype in ['int64', 'float64']:
            stats.update({
                "Min": float(data[column].min()),
                "Max": float(data[column].max()),
                "Mean": float(data[column].mean()),
                "Median": float(data[column].median()),
                "Std Dev": float(data[column].std()),
            })
        else:
            stats["Sample Values"] = ", ".join(str(v) for v in data[column].unique()[:5])
        return stats
    except Exception as e:
        return f"Error: {str(e)}"

def ai_understand_query(user_question, available_columns, timeout_seconds, temperature):
    """Use AI to understand what the user is asking"""
    columns_str = ", ".join(available_columns)
    
    prompt = f"""You are a data analyst. Read this question and identify what analysis is needed.

Available columns: {columns_str}

Dimension columns (for grouping): Geo, Country, Sales_Rep, Customer, Category, Product
Metric columns (for values): Spend, Savings, Revenue, Profit, KPI_%, Profit_Margin_%, Units_Sold, Marketing_Spend, Customer_Satisfaction_Score, Employee_Engagement_%, Return_Rate_%
Time columns: Year, Quarter, Month, Date

Question: {user_question}

Extract the most relevant column that the question asks about. If multiple columns mentioned, pick the PRIMARY one being asked.

Answer with ONLY these 7 lines, nothing else:
QUERY_TYPE: [total_by_period OR best_worst_by_category OR filter_and_sum]
PERIOD_COLUMN: [Year OR Quarter OR Month OR NONE]
CATEGORY_COLUMN: [Geo OR Country OR Sales_Rep OR Customer OR Category OR Product OR NONE]
VALUE_COLUMN: [the main metric column being asked about]
FILTER_COLUMN: [dimension to filter by or NONE]
FILTER_VALUE: [exact value from question or NONE]
REASONING: [one sentence]"""
    
    try:
        response = requests.post(OLLAMA_API, json={"model": "llama2", "prompt": prompt, "stream": False, "temperature": 0}, timeout=timeout_seconds)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "").strip()
            
            # Parse the text response
            lines = ai_response.split('\n')
            query_params = {
                "query_type": "total_by_period",
                "period_column": None,
                "category_column": None,
                "value_column": None,
                "filter_column": None,
                "filter_value": None,
                "reasoning": "Analysis requested"
            }
            
            for line in lines:
                if "QUERY_TYPE:" in line:
                    val = line.split(":", 1)[1].strip().lower()
                    if "best_worst" in val or "best" in val or "worst" in val or "category" in val:
                        query_params["query_type"] = "best_worst_by_category"
                    elif "filter" in val:
                        query_params["query_type"] = "filter_and_sum"
                    else:
                        query_params["query_type"] = "total_by_period"
                
                elif "PERIOD_COLUMN:" in line:
                    val = line.split(":", 1)[1].strip()
                    query_params["period_column"] = val if val.upper() != "NONE" else None
                
                elif "CATEGORY_COLUMN:" in line:
                    val = line.split(":", 1)[1].strip()
                    query_params["category_column"] = val if val.upper() != "NONE" else None
                
                elif "VALUE_COLUMN:" in line:
                    val = line.split(":", 1)[1].strip()
                    query_params["value_column"] = val if val.upper() != "NONE" else None
                
                elif "FILTER_COLUMN:" in line:
                    val = line.split(":", 1)[1].strip()
                    query_params["filter_column"] = val if val.upper() != "NONE" else None
                
                elif "FILTER_VALUE:" in line:
                    val = line.split(":", 1)[1].strip()
                    query_params["filter_value"] = val if val.upper() != "NONE" else None
                
                elif "REASONING:" in line:
                    query_params["reasoning"] = line.split(":", 1)[1].strip()
            
            # POST-PROCESSING: Detect query type from keywords if AI got it wrong
            question_lower = user_question.lower()
            
            # Check for best/worst keywords
            if any(word in question_lower for word in ['highest', 'lowest', 'best', 'worst', 'top', 'bottom', 'leader', 'champion']):
                if query_params["query_type"] != "filter_and_sum":
                    query_params["query_type"] = "best_worst_by_category"
                    
                    # If no category column was found, try to infer it
                    if not query_params["category_column"]:
                        for dim in DIMENSION_COLUMNS:
                            if dim.lower() in question_lower:
                                query_params["category_column"] = dim
                                break
            
            # Check for "by" keyword (indicates grouping/category)
            if " by " in question_lower and query_params["query_type"] == "total_by_period":
                query_params["query_type"] = "best_worst_by_category"
                # Extract what comes after "by"
                by_index = question_lower.find(" by ")
                after_by = question_lower[by_index + 4:].split()[0]
                for dim in DIMENSION_COLUMNS:
                    if dim.lower() in after_by.lower() or after_by.lower() in dim.lower():
                        query_params["category_column"] = dim
                        break
            
            return query_params, None
        else:
            return None, f"AI Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return None, f"AI Timeout after {timeout_seconds}s"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Ollama. Run: ollama serve"
    except Exception as e:
        return None, f"Error: {str(e)}"

def execute_query(data, query_params):
    """Execute the query using Python"""
    try:
        query_type = query_params.get("query_type", "").lower()
        
        if query_type == "total_by_period":
            period_col = query_params.get("period_column")
            value_col = query_params.get("value_column")
            
            if not period_col or not value_col:
                return None, "Missing period or value column"
            
            if period_col not in data.columns or value_col not in data.columns:
                return None, f"Column not found. Looking for Period: '{period_col}', Value: '{value_col}'. Available: {list(data.columns)}"
            
            result = data.groupby(period_col)[value_col].sum().reset_index()
            result.columns = [period_col, f"Total {value_col}"]
            return result, None
        
        elif query_type == "best_worst_by_category":
            category_col = query_params.get("category_column")
            value_col = query_params.get("value_column")
            
            if not category_col or not value_col:
                return None, f"Missing category or value column. Category: {category_col}, Value: {value_col}"
            
            if category_col not in data.columns:
                return None, f"Category column '{category_col}' not found. Available: {list(data.columns)}"
            
            if value_col not in data.columns:
                return None, f"Value column '{value_col}' not found. Available: {list(data.columns)}"
            
            # Determine aggregation function based on column type
            value_col_type = get_column_type(value_col)
            
            if value_col_type == 'percentage':
                result = data.groupby(category_col)[value_col].mean().reset_index()
                result = result.sort_values(value_col, ascending=False)
                result.columns = [category_col, f"Average {value_col}"]
            elif value_col_type == 'financial':
                result = data.groupby(category_col)[value_col].sum().reset_index()
                result = result.sort_values(value_col, ascending=False)
                result.columns = [category_col, f"Total {value_col}"]
            else:
                result = data.groupby(category_col)[value_col].max().reset_index()
                result = result.sort_values(value_col, ascending=False)
                result.columns = [category_col, f"Max {value_col}"]
            
            return result, None
        
        elif query_type == "filter_and_sum":
            filter_col = query_params.get("filter_column")
            filter_val = query_params.get("filter_value")
            value_col = query_params.get("value_column")
            
            if not filter_col or not filter_val or not value_col:
                return None, f"Missing filter parameters. Filter Column: {filter_col}, Filter Value: {filter_val}, Value Column: {value_col}"
            
            if filter_col not in data.columns or value_col not in data.columns:
                return None, "Invalid column names"
            
            filtered = data[data[filter_col].astype(str).str.contains(str(filter_val), case=False, na=False)]
            total = filtered[value_col].sum()
            
            result_table = filtered[[filter_col, value_col]].copy()
            
            return (result_table, total), None
        
        else:
            return None, "Unknown query type"
    
    except Exception as e:
        return None, f"Execution error: {str(e)}"

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
            st.dataframe(data.head(rows_to_show), width='stretch')
        
        with tab2:
            st.subheader("📈 Quick Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(data):,}")
            with col2:
                st.metric("Total Columns", len(data.columns))
            with col3:
                st.metric("Missing Values", int(data.isnull().sum().sum()))
            with col4:
                st.metric("Data Size", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            st.write("**Numeric Column Statistics:**")
            numeric_data = data.select_dtypes(include=['number'])
            if not numeric_data.empty:
                st.dataframe(numeric_data.describe(), width='stretch')
            else:
                st.info("No numeric columns found")
        
        with tab3:
            st.subheader("🔍 Column Details")
            col_info = pd.DataFrame({
                'Column Name': data.columns,
                'Data Type': [str(dt) for dt in data.dtypes.values],
                'Non-Null Count': [int(data[col].count()) for col in data.columns],
                'Null Count': [int(data[col].isnull().sum()) for col in data.columns],
                'Unique Values': [f"{int(data[col].nunique()):,}" for col in data.columns]
            })
            st.dataframe(col_info, width='stretch')
        
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
                    st.dataframe(filtered, width='stretch')
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
                                    st.dataframe(result, width='stretch')
                                    st.code(result.to_csv(index=False), language="csv")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
        
        with calc_tab4:
            st.write("**Select specific columns**")
            cols_to_display = st.multiselect("Select columns:", data.columns, default=list(data.columns[:10]), key="cols_display")
            if cols_to_display:
                if st.button("Display Data", key="btn_display"):
                    st.dataframe(data[cols_to_display], width='stretch')
                    st.code(data[cols_to_display].to_csv(index=False), language="csv")
        
        st.divider()
        
        st.subheader("🤖 AI-Powered Natural Language Query")
        st.info("Ask questions in natural language. AI understands → Python calculates → Exact results!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Rows", f"{len(data):,}")
        with col2:
            st.metric("Dataset Columns", len(data.columns))
        with col3:
            st.metric("Status", "✅ Ready")
        
        st.write(f"**Available columns:** {', '.join(data.columns.tolist())}")
        
        # Initialize session state for question tracking
        if 'last_question' not in st.session_state:
            st.session_state.last_question = ""
        
        user_question = st.text_area("Ask about your data (natural language):", placeholder="Example: What was the yearly total spend? or What GEO had the highest KPI_% over all years?", height=100, key="user_question_input")
        
        # Only process if question is new
        if user_question and user_question != st.session_state.last_question:
            st.session_state.last_question = user_question
            st.write("🤖 Processing your question...")
            
            # Step 1: AI understands the question
            with st.spinner("🧠 AI analyzing question..."):
                query_params, ai_error = ai_understand_query(user_question, data.columns.tolist(), timeout_seconds, temperature)
            
            if ai_error:
                st.error(f"❌ AI Error: {ai_error}")
            elif query_params:
                # Show what AI understood
                if show_steps:
                    with st.expander("🔍 AI Understanding"):
                        st.write(f"**Your Question:** {user_question}")
                        st.write(f"**AI Understood:** {query_params.get('reasoning', 'N/A')}")
                        st.write(f"**Query Type:** `{query_params.get('query_type')}`")
                        st.write("**Columns to Use:**")
                        st.json({
                            "Period Column": query_params.get('period_column'),
                            "Category Column": query_params.get('category_column'),
                            "Value Column": query_params.get('value_column'),
                            "Filter Column": query_params.get('filter_column'),
                            "Filter Value": query_params.get('filter_value')
                        })
                        st.write(f"**Available columns in dataset:** {data.columns.tolist()}")
                
                # Step 2: Python executes
                with st.spinner("📊 Calculating results..."):
                    result, error = execute_query(data, query_params)
                
                if error and result is None:
                    st.error(f"❌ Error: {error}")
                else:
                    st.success("✅ Results Calculated!")
                    
                    if isinstance(result, tuple):
                        # Filter & Sum result
                        result_table, total = result
                        st.metric("Total", f"{total:,.2f}")
                        st.dataframe(result_table, width='stretch')
                    elif isinstance(result, pd.DataFrame):
                        st.dataframe(result, width='stretch')
                        st.code(result.to_csv(index=False), language="csv")
        
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