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

def find_correct_column(data, col_name):
    """Find the correct column name in data, handling typos and variations"""
    if not col_name:
        return None
    
    # Exact match
    if col_name in data.columns:
        return col_name
    
    col_name_lower = col_name.lower()
    
    # Try case-insensitive match
    for actual_col in data.columns:
        if actual_col.lower() == col_name_lower:
            return actual_col
    
    # Try removing underscores/spaces
    col_normalized = col_name_lower.replace('_', '').replace(' ', '')
    for actual_col in data.columns:
        actual_normalized = actual_col.lower().replace('_', '').replace(' ', '')
        if actual_normalized == col_normalized:
            return actual_col
    
    # Try partial match
    for actual_col in data.columns:
        if col_name_lower in actual_col.lower() or actual_col.lower() in col_name_lower:
            return actual_col
    
    return None

def find_correct_filter_column(data, filter_value):
    """Search ALL dimension columns to find which one contains this value"""
    if not filter_value:
        return None
    
    filter_value_lower = str(filter_value).lower().strip()
    
    # Try exact match first, then partial match
    for col in DIMENSION_COLUMNS:
        if col in data.columns:
            for actual_value in data[col].dropna().unique():
                actual_value_lower = str(actual_value).lower().strip()
                if actual_value_lower == filter_value_lower:
                    return col
    
    # Try partial match
    for col in DIMENSION_COLUMNS:
        if col in data.columns:
            for actual_value in data[col].dropna().unique():
                actual_value_lower = str(actual_value).lower().strip()
                if filter_value_lower in actual_value_lower or actual_value_lower in filter_value_lower:
                    return col
    
    return None

def detect_aggregation_function(question):
    """Detect what aggregation function the user wants based on keywords"""
    question_lower = question.lower()
    
    # Check for average/mean
    if any(word in question_lower for word in ['average', 'avg', 'mean']):
        return 'mean'
    
    # Check for total/sum
    if any(word in question_lower for word in ['total', 'sum']):
        return 'sum'
    
    # Check for highest/max
    if any(word in question_lower for word in ['highest', 'max', 'maximum']):
        return 'max'
    
    # Check for lowest/min
    if any(word in question_lower for word in ['lowest', 'min', 'minimum']):
        return 'min'
    
    # Check for count
    if any(word in question_lower for word in ['count', 'how many']):
        return 'count'
    
    return None

with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("AI Creativity", 0.0, 1.0, 0.3)
    show_steps = st.checkbox("Show AI Reasoning", value=True)
    timeout_seconds = st.slider("AI Response Timeout (seconds)", 60, 1800, 600, step=60)

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

def ai_understand_query(user_question, data, timeout_seconds, temperature):
    """Use AI to understand what the user is asking"""
    
    prompt = f"""TASK: Extract what the user is asking from a data question.

Available Metrics: Spend, Savings, Revenue, Profit, KPI_%, Profit_Margin_%, Units_Sold, Marketing_Spend, Customer_Satisfaction_Score, Employee_Engagement_%, Return_Rate_%
Available Dimensions: Geo, Country, Sales_Rep, Customer, Category, Product
Available Time: Year, Quarter, Month, Date

QUESTION: {user_question}

RULES:
- METRIC: Which NUMBER column is being asked about?
- DIMENSION: Which dimension to GROUP BY or FILTER (Geo, Country, Product, etc)?
- FILTER_VALUE: Any SPECIFIC value mentioned (North America, 2022, etc)?
- TIME_PERIOD: How to break down by time (Year, Month, Quarter)?

RESPOND WITH EXACTLY 7 LINES:
METRIC: [metric name or NONE]
DIMENSION: [dimension name or NONE]
TIME_PERIOD: [Year OR Quarter OR Month OR NONE]
FILTER_VALUE: [specific value mentioned or NONE]
QUERY_PATTERN: [yearly_total OR category_comparison OR filtered_total]
REASONING: [one line]
AI_CONFIDENCE: [high OR medium OR low]"""
    
    try:
        response = requests.post(OLLAMA_API, json={"model": "llama2", "prompt": prompt, "stream": False, "temperature": 0}, timeout=timeout_seconds)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "").strip()
            
            lines = ai_response.split('\n')
            extracted = {
                "metric": None,
                "dimension": None,
                "time_period": None,
                "filter_value": None,
                "query_pattern": None,
                "reasoning": "Analysis",
                "confidence": "medium"
            }
            
            for line in lines:
                if "METRIC:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE":
                        extracted["metric"] = val
                
                elif "DIMENSION:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE":
                        extracted["dimension"] = val
                
                elif "TIME_PERIOD:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE":
                        extracted["time_period"] = val
                
                elif "FILTER_VALUE:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE":
                        extracted["filter_value"] = val
                
                elif "QUERY_PATTERN:" in line:
                    val = line.split(":", 1)[1].strip().lower()
                    extracted["query_pattern"] = val
                
                elif "REASONING:" in line:
                    extracted["reasoning"] = line.split(":", 1)[1].strip()
                
                elif "AI_CONFIDENCE:" in line:
                    val = line.split(":", 1)[1].strip().lower()
                    extracted["confidence"] = val
            
            # Fix column names using smart matching
            if extracted["metric"]:
                correct_metric = find_correct_column(data, extracted["metric"])
                if correct_metric:
                    extracted["metric"] = correct_metric
            
            if extracted["dimension"]:
                correct_dimension = find_correct_column(data, extracted["dimension"])
                if correct_dimension:
                    extracted["dimension"] = correct_dimension
            
            if extracted["time_period"]:
                correct_time = find_correct_column(data, extracted["time_period"])
                if correct_time:
                    extracted["time_period"] = correct_time
            
            # Build query_params
            query_params = {
                "query_type": "total_by_period",
                "period_column": extracted["time_period"],
                "category_column": None,
                "value_column": extracted["metric"],
                "filter_column": None,
                "filter_value": None,
                "reasoning": extracted["reasoning"],
                "aggregation_function": None
            }
            
            # Detect aggregation function from question
            agg_func = detect_aggregation_function(user_question)
            if agg_func:
                query_params["aggregation_function"] = agg_func
            
            # Determine query type and find filter column
            question_lower = user_question.lower()
            
            # Check if this is a filter query (has a specific value)
            if extracted["filter_value"]:
                correct_col = find_correct_filter_column(data, extracted["filter_value"])
                
                if correct_col:
                    query_params["query_type"] = "filter_and_sum"
                    query_params["filter_column"] = correct_col
                    query_params["filter_value"] = extracted["filter_value"]
                    query_params["period_column"] = None
                else:
                    query_params["query_type"] = "filter_and_sum"
                    query_params["filter_column"] = extracted["dimension"]
                    query_params["filter_value"] = extracted["filter_value"]
                    query_params["period_column"] = None
            
            # Check if asking about "highest/lowest/best/worst"
            elif any(word in question_lower for word in ['highest', 'lowest', 'best', 'worst', 'top']):
                query_params["query_type"] = "best_worst_by_category"
                query_params["category_column"] = extracted["dimension"]
                query_params["period_column"] = None
            
            # Check if asking "by X" for grouping
            elif " by " in question_lower and extracted["dimension"]:
                query_params["query_type"] = "best_worst_by_category"
                query_params["category_column"] = extracted["dimension"]
                query_params["period_column"] = None
            
            # Otherwise use time period grouping
            else:
                query_params["query_type"] = "total_by_period"
                query_params["period_column"] = extracted["time_period"] or "Year"
            
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
            agg_func = query_params.get("aggregation_function")
            
            if not category_col or not value_col:
                return None, f"Missing category or value column. Category: {category_col}, Value: {value_col}"
            
            if category_col not in data.columns:
                return None, f"Category column '{category_col}' not found. Available: {list(data.columns)}"
            
            if value_col not in data.columns:
                return None, f"Value column '{value_col}' not found. Available: {list(data.columns)}"
            
            # Determine aggregation function
            if agg_func == 'mean':
                result = data.groupby(category_col)[value_col].mean().reset_index()
                label = f"Average {value_col}"
            elif agg_func == 'sum':
                result = data.groupby(category_col)[value_col].sum().reset_index()
                label = f"Total {value_col}"
            elif agg_func == 'max':
                result = data.groupby(category_col)[value_col].max().reset_index()
                label = f"Max {value_col}"
            elif agg_func == 'min':
                result = data.groupby(category_col)[value_col].min().reset_index()
                label = f"Min {value_col}"
            else:
                # Default based on column type
                value_col_type = get_column_type(value_col)
                
                if value_col_type == 'percentage':
                    result = data.groupby(category_col)[value_col].mean().reset_index()
                    label = f"Average {value_col}"
                elif value_col_type == 'financial':
                    result = data.groupby(category_col)[value_col].sum().reset_index()
                    label = f"Total {value_col}"
                else:
                    result = data.groupby(category_col)[value_col].max().reset_index()
                    label = f"Max {value_col}"
            
            result = result.sort_values(value_col, ascending=False)
            result.columns = [category_col, label]
            
            return result, None
        
        elif query_type == "filter_and_sum":
            filter_col = query_params.get("filter_column")
            filter_val = query_params.get("filter_value")
            value_col = query_params.get("value_column")
            
            if not filter_col or not filter_val or not value_col:
                return None, f"Missing filter parameters. Filter Column: {filter_col}, Filter Value: {filter_val}, Value Column: {value_col}"
            
            if filter_col not in data.columns or value_col not in data.columns:
                return None, f"Invalid columns. Filter: {filter_col}, Value: {value_col}. Available: {list(data.columns)}"
            
            filtered = data[data[filter_col].astype(str).str.contains(str(filter_val), case=False, na=False)]
            if len(filtered) == 0:
                return None, f"No data found for {filter_col}='{filter_val}'. Available values in {filter_col}: {data[filter_col].unique()[:5].tolist()}"
            
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
        
        with st.form("query_form", clear_on_submit=True):
            user_question = st.text_area("Ask about your data (natural language):", placeholder="Example: What was the yearly total spend? or What GEO had the highest KPI_% over all years?", height=100)
            submit_button = st.form_submit_button("📊 Analyze", use_container_width=True)
        
        if submit_button and user_question:
            st.write("🤖 Processing your question...")
            
            with st.spinner("🧠 AI analyzing question..."):
                query_params, ai_error = ai_understand_query(user_question, data, timeout_seconds, temperature)
            
            if ai_error:
                st.error(f"❌ AI Error: {ai_error}")
            elif query_params:
                if show_steps:
                    with st.expander("🔍 AI Understanding"):
                        st.write(f"**Your Question:** {user_question}")
                        st.write(f"**AI Understood:** {query_params.get('reasoning', 'N/A')}")
                        st.write(f"**Query Type:** `{query_params.get('query_type')}`")
                        st.write(f"**Aggregation Function:** `{query_params.get('aggregation_function') or 'default'}`")
                        st.write("**Columns to Use:**")
                        st.json({
                            "Period Column": query_params.get('period_column'),
                            "Category Column": query_params.get('category_column'),
                            "Value Column": query_params.get('value_column'),
                            "Filter Column": query_params.get('filter_column'),
                            "Filter Value": query_params.get('filter_value')
                        })
                        st.write(f"**Available columns in dataset:** {data.columns.tolist()}")
                
                with st.spinner("📊 Calculating results..."):
                    result, error = execute_query(data, query_params)
                
                if error and result is None:
                    st.error(f"❌ Error: {error}")
                else:
                    st.success("✅ Results Calculated!")
                    
                    if isinstance(result, tuple):
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