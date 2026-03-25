import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import requests
import json
import re

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

ALL_METRICS = FINANCIAL_COLUMNS + PERCENTAGE_COLUMNS + COUNT_COLUMNS

def profile_columns(data):
    """
    Intelligently profile columns to determine their type and purpose.
    Works with ANY dataset regardless of domain.
    """
    column_profiles = {}
    
    for col in data.columns:
        col_lower = col.lower()
        dtype = data[col].dtype
        
        profile = {
            'name': col,
            'dtype': str(dtype),
            'null_count': int(data[col].isnull().sum()),
            'unique_count': int(data[col].nunique()),
            'inferred_type': None,
            'synonyms': [col_lower, col],
        }
        
        # NUMERIC ANALYSIS
        if dtype in ['int64', 'float64']:
            profile['numeric'] = True
            profile['min'] = float(data[col].min()) if not data[col].isnull().all() else None
            profile['max'] = float(data[col].max()) if not data[col].isnull().all() else None
            profile['mean'] = float(data[col].mean()) if not data[col].isnull().all() else None
            profile['std'] = float(data[col].std()) if not data[col].isnull().all() else None
            
            # Is it a percentage?
            if profile['max'] and profile['min'] is not None:
                if profile['max'] <= 100 and profile['min'] >= 0 and '%' in col_lower:
                    profile['inferred_type'] = 'metric_percentage'
                    profile['synonyms'].extend(['percentage', 'percent', 'pct', '%'])
            
            # Is it a count/quantity?
            if any(word in col_lower for word in ['count', 'quantity', 'qty', 'units', 'items', 'volume', 'total', 'number', 'mp', 'w', 't', 'l', 'gf', 'ga', 'matches', 'wins', 'losses', 'draws']):
                profile['inferred_type'] = 'metric_count'
                profile['synonyms'].extend(['count', 'quantity', 'qty', 'units', 'items', 'volume', 'total', 'number', 'matches', 'wins', 'losses', 'draws'])
            
            # Is it a financial/scoring metric?
            elif any(word in col_lower for word in ['spend', 'cost', 'revenue', 'sales', 'profit', 'income', 'price', 'amount', 'value', 'fees', 'budget', 'points', 'score', 'goals']):
                profile['inferred_type'] = 'metric_financial'
                profile['synonyms'].extend(['spend', 'cost', 'revenue', 'sales', 'profit', 'income', 'amount', 'value', 'fees', 'budget', 'points', 'score', 'goals'])
            
            # Default: assume it's a metric
            else:
                profile['inferred_type'] = 'metric'
                profile['synonyms'].extend(['value', 'amount', 'total', 'sum', 'metric'])
        
        # STRING/CATEGORY ANALYSIS
        else:
            profile['numeric'] = False
            cardinality = profile['unique_count']
            total_rows = len(data)
            cardinality_ratio = cardinality / total_rows if total_rows > 0 else 0
            
            # Check column name FIRST - if it looks like a dimension, treat it as such
            is_likely_dimension = any(word in col_lower for word in [
                'team', 'player', 'product', 'customer', 'country', 'region', 'sport', 'league',
                'category', 'type', 'group', 'name', 'sales_rep', 'rep', 'agent', 'user',
                'brand', 'manufacturer', 'department', 'segment', 'state', 'geo', 'location',
                'athlete', 'coach', 'company', 'organization', 'division', 'conference'
            ])
            
            if is_likely_dimension:
                profile['inferred_type'] = 'dimension'
                profile['synonyms'].extend(['category', 'type', 'group', 'classification', 'segment'])
                
                # Add specific synonyms
                if 'team' in col_lower:
                    profile['synonyms'].extend(['team', 'teams', 'club'])
                if 'player' in col_lower:
                    profile['synonyms'].extend(['player', 'players', 'athlete'])
                if 'product' in col_lower:
                    profile['synonyms'].extend(['product', 'item', 'sku', 'brand'])
                if 'customer' in col_lower:
                    profile['synonyms'].extend(['customer', 'client', 'account'])
                if any(word in col_lower for word in ['region', 'geo', 'location', 'country']):
                    profile['synonyms'].extend(['region', 'geo', 'location', 'country', 'state'])
            
            # High cardinality strings - STILL treat as potential dimension if name suggests it
            elif cardinality_ratio > 0.5:
                profile['inferred_type'] = 'id_or_name'
                profile['synonyms'].extend(['name', 'id', 'identifier', 'description', 'title'])
            
            # Low cardinality strings
            elif cardinality <= 100:
                profile['inferred_type'] = 'dimension'
                profile['synonyms'].extend(['category', 'type', 'group', 'classification', 'segment'])
            
            # Time columns
            if any(word in col_lower for word in ['date', 'time', 'year', 'month', 'quarter', 'week', 'day', 'timestamp']):
                profile['inferred_type'] = 'time'
                profile['synonyms'].extend(['date', 'time', 'year', 'month', 'quarter', 'week', 'day', 'timestamp'])
                
                try:
                    pd.to_datetime(data[col])
                    profile['is_date'] = True
                except:
                    profile['is_date'] = False
        
        column_profiles[col] = profile
    
    return column_profiles

def categorize_columns(column_profiles):
    """Categorize profiled columns into buckets."""
    categorized = {
        'metrics': {},
        'dimensions': {},
        'time': {},
        'ids': {},
        'unknown': {}
    }
    
    for col, profile in column_profiles.items():
        inferred = profile['inferred_type']
        
        if inferred and 'metric' in inferred:
            categorized['metrics'][col] = profile
        elif inferred == 'dimension':
            categorized['dimensions'][col] = profile
        elif inferred == 'time':
            categorized['time'][col] = profile
        elif inferred == 'id_or_name':
            categorized['ids'][col] = profile
        else:
            categorized['unknown'][col] = profile
    
    return categorized

def find_best_metric(question, categorized_columns):
    """Smart metric detection that works across domains."""
    if not categorized_columns['metrics'] or not question:
        return None
    
    question_lower = question.lower()
    best_match = None
    best_score = 0
    
    for col, profile in categorized_columns['metrics'].items():
        for synonym in profile['synonyms']:
            if synonym in question_lower:
                score = len(synonym)
                if score > best_score:
                    best_score = score
                    best_match = col
    
    return best_match

def find_best_dimension(question, categorized_columns):
    """Smart dimension detection - check ALL categorical columns."""
    if not question:
        return None
    
    question_lower = question.lower()
    best_match = None
    best_score = 0
    
    # Check DIMENSION columns first (high priority)
    for col, profile in categorized_columns['dimensions'].items():
        for synonym in profile['synonyms']:
            if synonym in question_lower:
                score = len(synonym)
                if score > best_score:
                    best_score = score
                    best_match = col
    
    # IMPORTANT: Also check ID/NAME columns (they can be dimensions!)
    # This is crucial for sports tables with team names
    for col, profile in categorized_columns['ids'].items():
        for synonym in profile['synonyms']:
            if synonym in question_lower:
                score = len(synonym)
                if score > best_score:
                    best_score = score
                    best_match = col
    
    # If still no match but question asks "which", "by", "per", etc - use ANY text column
    if not best_match and any(word in question_lower for word in ['which', 'by ', 'per ', 'for each', 'by each']):
        if categorized_columns['dimensions']:
            best_match = list(categorized_columns['dimensions'].keys())[0]
        elif categorized_columns['ids']:
            best_match = list(categorized_columns['ids'].keys())[0]
    
    return best_match

def find_best_time_column(question, categorized_columns):
    """Smart time column detection."""
    if not categorized_columns['time'] or not question:
        return None
    
    question_lower = question.lower()
    
    if re.search(r'\b(q[1-4]|quarter)\b', question_lower):
        for col in categorized_columns['time'].keys():
            if 'quarter' in col.lower():
                return col
    
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|month)\b', question_lower):
        for col in categorized_columns['time'].keys():
            if 'month' in col.lower():
                return col
    
    if re.search(r'\b(19|20)\d{2}\b|year\b', question_lower):
        for col in categorized_columns['time'].keys():
            if 'year' in col.lower():
                return col
    
    if categorized_columns['time']:
        return list(categorized_columns['time'].keys())[0]
    
    return None

def detect_time_period_value(question):
    """Detect time period values from question."""
    question_lower = question.lower()
    words = re.findall(r'\b\w+\b', question_lower)
    
    for word in words:
        if word in ['q1', 'q2', 'q3', 'q4']:
            return word.upper()
    
    full_months = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']
    for word in words:
        if word in full_months:
            return word.capitalize()
    
    month_abbrs = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for word in words:
        if word in month_abbrs:
            return word.capitalize()
    
    for word in words:
        if re.match(r'^(19|20)\d{2}$', word):
            return word
    
    return None

def get_column_type(col_name):
    """Determine column type for aggregation defaults."""
    col_lower = col_name.lower()
    
    if any(word in col_lower for word in ['spend', 'cost', 'revenue', 'sales', 'profit', 'income']):
        return 'financial'
    elif '%' in col_lower or any(word in col_lower for word in ['rate', 'percentage', 'percent']):
        return 'percentage'
    elif any(word in col_lower for word in ['count', 'units', 'quantity', 'qty', 'points', 'goals']):
        return 'count'
    elif any(word in col_lower for word in ['date', 'time', 'year', 'month', 'quarter']):
        return 'date'
    else:
        return 'numeric'

def get_recommended_functions(col_name):
    """Get recommended aggregation functions."""
    col_type = get_column_type(col_name)
    recommendations = {
        'financial': {'default': 'sum', 'options': {'sum': 'Total (SUM)', 'max': 'Highest (MAX)', 'min': 'Lowest (MIN)', 'mean': 'Average (MEAN)'}},
        'percentage': {'default': 'mean', 'options': {'mean': 'Average % (MEAN)', 'max': 'Highest % (MAX)', 'min': 'Lowest % (MIN)'}},
        'count': {'default': 'sum', 'options': {'sum': 'Total (SUM)', 'mean': 'Average (MEAN)', 'max': 'Maximum (MAX)', 'min': 'Minimum (MIN)'}},
        'numeric': {'default': 'sum', 'options': {'sum': 'Total (SUM)', 'mean': 'Average (MEAN)', 'max': 'Maximum (MAX)', 'min': 'Minimum (MIN)'}}
    }
    return recommendations.get(col_type, recommendations['numeric'])

def find_correct_column(data, col_name):
    """Find correct column name in data."""
    if not col_name:
        return None
    
    if col_name in data.columns:
        return col_name
    
    col_name_lower = col_name.lower()
    
    for actual_col in data.columns:
        if actual_col.lower() == col_name_lower:
            return actual_col
    
    return None

def find_correct_filter_column(data, filter_value, categorized_columns):
    """Find which dimension column contains this filter value."""
    if not filter_value:
        return None
    
    filter_value_lower = str(filter_value).lower().strip()
    
    # Search dimensions first
    for col in categorized_columns['dimensions'].keys():
        if col in data.columns:
            for actual_value in data[col].dropna().unique():
                if str(actual_value).lower().strip() == filter_value_lower:
                    return col
    
    return None

def normalize_filter_value(filter_value, filter_column, data):
    """Normalize filter values to match data format."""
    if not filter_value or not filter_column:
        return filter_value
    
    filter_value_str = str(filter_value).lower().strip()
    
    if 'quarter' in filter_column.lower():
        if filter_value_str.startswith('q'):
            quarter_num = filter_value_str[1:]
            if quarter_num in ['1', '2', '3', '4']:
                return quarter_num
    
    return filter_value

def detect_aggregation_function(question, value_col, category_col=None):
    """Smart aggregation detection."""
    question_lower = question.lower()
    col_type = get_column_type(value_col)
    
    if category_col:
        if any(word in question_lower for word in ['lowest', 'highest', 'best', 'worst', 'top', 'most']):
            if col_type == 'percentage':
                return 'mean'
            elif col_type in ['financial', 'count']:
                return 'max' if 'highest' in question_lower or 'most' in question_lower or 'best' in question_lower else 'min'
    
    if any(word in question_lower for word in ['average', 'avg', 'mean', 'per ']):
        return 'mean'
    if any(word in question_lower for word in ['total', 'sum']):
        return 'sum'
    if any(word in question_lower for word in ['count', 'how many']):
        return 'count'
    if any(word in question_lower for word in ['minimum', 'min', 'lowest']):
        return 'min'
    if any(word in question_lower for word in ['maximum', 'max', 'highest', 'most']):
        return 'max'
    
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
                "Min": float(data[column].min()) if not data[column].isnull().all() else None,
                "Max": float(data[column].max()) if not data[column].isnull().all() else None,
                "Mean": float(data[column].mean()) if not data[column].isnull().all() else None,
            })
        else:
            stats["Sample Values"] = ", ".join(str(v) for v in data[column].unique()[:5])
        return stats
    except Exception as e:
        return f"Error: {str(e)}"

def ai_understand_query(user_question, data, timeout_seconds, temperature, categorized_columns):
    """Use AI with smart column detection."""
    
    smart_metric = find_best_metric(user_question, categorized_columns)
    smart_dimension = find_best_dimension(user_question, categorized_columns)
    smart_time = find_best_time_column(user_question, categorized_columns)
    pre_time_value = detect_time_period_value(user_question)
    
    metrics_list = ", ".join(categorized_columns['metrics'].keys())
    dimensions_list = ", ".join(list(categorized_columns['dimensions'].keys()) + list(categorized_columns['ids'].keys()))
    time_list = ", ".join(categorized_columns['time'].keys())
    
    prompt = f"""TASK: Extract EXACT column names from a data question. RETURN ONLY EXACT COLUMN NAMES.

AVAILABLE METRICS: {metrics_list or 'none'}
AVAILABLE DIMENSIONS: {dimensions_list or 'none'}
AVAILABLE TIME COLUMNS: {time_list or 'none'}

QUESTION: {user_question}

RULES:
1. METRIC: Return EXACTLY ONE metric name from AVAILABLE METRICS.
   HINT: Likely: {smart_metric or 'unknown'}
2. DIMENSION: Return dimension name ONLY if asking "which", "by", "per", or comparing.
   HINT: Likely: {smart_dimension or 'none'}
3. TIME_COLUMN: Return time column name ONLY if grouping/filtering by time.
   HINT: Likely: {smart_time or 'none'}
4. FILTER_VALUE: Return a specific value - ONLY if explicitly mentioned.
5. NEVER add values that aren't explicitly in the question.

RESPOND EXACTLY 7 LINES (no extra text):
METRIC: [exact name or NONE]
DIMENSION: [exact name or NONE]
TIME_COLUMN: [exact name or NONE]
FILTER_VALUE: [value or NONE]
QUERY_PATTERN: [category_comparison OR filtered_total OR time_total]
REASONING: [one line]
AI_CONFIDENCE: [high OR medium OR low]"""
    
    try:
        response = requests.post(OLLAMA_API, json={"model": "llama2", "prompt": prompt, "stream": False, "temperature": 0}, timeout=timeout_seconds)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "").strip()
            
            lines = ai_response.split('\n')
            extracted = {
                "metric": smart_metric,
                "dimension": smart_dimension,
                "time_column": smart_time,
                "filter_value": None,
                "query_pattern": None,
                "reasoning": "Analysis",
                "confidence": "medium",
                "time_value": pre_time_value
            }
            
            for line in lines:
                if "METRIC:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE" and val.strip():
                        extracted["metric"] = val
                elif "DIMENSION:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE" and val.strip():
                        extracted["dimension"] = val
                elif "TIME_COLUMN:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE" and val.strip():
                        extracted["time_column"] = val
                elif "FILTER_VALUE:" in line:
                    val = line.split(":", 1)[1].strip()
                    if val.upper() != "NONE" and val.strip():
                        extracted["filter_value"] = val
                elif "QUERY_PATTERN:" in line:
                    val = line.split(":", 1)[1].strip().lower()
                    extracted["query_pattern"] = val
                elif "REASONING:" in line:
                    extracted["reasoning"] = line.split(":", 1)[1].strip()
                elif "AI_CONFIDENCE:" in line:
                    val = line.split(":", 1)[1].strip().lower()
                    extracted["confidence"] = val
            
            query_params = {
                "query_type": "total_by_time",
                "time_column": extracted["time_column"],
                "category_column": None,
                "value_column": extracted["metric"],
                "filter_column": None,
                "filter_value": None,
                "reasoning": extracted["reasoning"],
                "aggregation_function": None,
                "time_value": extracted["time_value"]
            }
            
            if extracted["time_value"] and extracted["time_column"]:
                extracted["time_value"] = normalize_filter_value(
                    extracted["time_value"],
                    extracted["time_column"],
                    data
                )
            
            agg_func = detect_aggregation_function(user_question, extracted["metric"], extracted["dimension"])
            if agg_func:
                query_params["aggregation_function"] = agg_func
            
            question_lower = user_question.lower()
            
            # PRIORITY 1: Group by dimension
            if extracted["dimension"] and (any(word in question_lower for word in ['which', 'highest', 'lowest', 'best', 'worst', 'top', 'most', 'by ', 'per ', 'for each']) or ' by ' in question_lower):
                query_params["query_type"] = "best_worst_by_category"
                query_params["category_column"] = extracted["dimension"]
                query_params["time_column"] = None
                query_params["filter_column"] = None
                query_params["filter_value"] = None
            
            # PRIORITY 2: Filter by time
            elif extracted["time_value"]:
                query_params["query_type"] = "filter_by_time"
                query_params["time_column"] = extracted["time_column"]
                query_params["filter_value"] = extracted["time_value"]
                query_params["filter_column"] = extracted["time_column"]
            
            # PRIORITY 3: Total by time
            else:
                query_params["query_type"] = "total_by_time"
                query_params["time_column"] = extracted["time_column"]
            
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
    """Execute the query using Python."""
    try:
        query_type = query_params.get("query_type", "").lower()
        
        if query_type in ["total_by_time", "filter_by_time"]:
            time_col = query_params.get("time_column")
            value_col = query_params.get("value_column")
            filter_val = query_params.get("filter_value")
            
            if not value_col:
                return None, "Missing value column"
            
            if value_col not in data.columns:
                return None, f"Value column '{value_col}' not found. Available: {list(data.columns)}"
            
            if time_col and filter_val:
                if time_col not in data.columns:
                    return None, f"Time column '{time_col}' not found"
                filtered = data[data[time_col].astype(str) == str(filter_val)]
                if len(filtered) == 0:
                    return None, f"No data found for {time_col}='{filter_val}'"
                total = filtered[value_col].sum()
                return pd.DataFrame({time_col: [filter_val], f"Total {value_col}": [total]}), None
            elif time_col:
                if time_col not in data.columns:
                    return None, f"Time column '{time_col}' not found"
                result = data.groupby(time_col)[value_col].sum().reset_index()
                result.columns = [time_col, f"Total {value_col}"]
                return result, None
            else:
                total = data[value_col].sum()
                return pd.DataFrame({f"Total {value_col}": [total]}), None
        
        elif query_type == "best_worst_by_category":
            category_col = query_params.get("category_column")
            value_col = query_params.get("value_column")
            agg_func = query_params.get("aggregation_function")
            
            if not category_col or not value_col:
                return None, "Missing category or value column"
            
            if category_col not in data.columns or value_col not in data.columns:
                return None, f"Column not found. Category: '{category_col}', Value: '{value_col}'. Available: {list(data.columns)}"
            
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
                col_type = get_column_type(value_col)
                if col_type == 'percentage':
                    result = data.groupby(category_col)[value_col].mean().reset_index()
                    label = f"Average {value_col}"
                else:
                    result = data.groupby(category_col)[value_col].max().reset_index()
                    label = f"Max {value_col}"
            
            result = result.sort_values(value_col, ascending=False)
            result.columns = [category_col, label]
            return result, None
        
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
        
        # PROFILE AND CATEGORIZE COLUMNS
        column_profiles = profile_columns(data)
        categorized_columns = categorize_columns(column_profiles)
        
        # Show detected column types
        with st.expander("🔍 Detected Column Types", expanded=False):
            st.write("**Metrics:**", list(categorized_columns['metrics'].keys()))
            st.write("**Dimensions:**", list(categorized_columns['dimensions'].keys()))
            st.write("**IDs/Names:**", list(categorized_columns['ids'].keys()))
            st.write("**Time Columns:**", list(categorized_columns['time'].keys()))
            if categorized_columns['unknown']:
                st.write("**Unknown:**", list(categorized_columns['unknown'].keys()))
        
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
                'Data Type': [str(data[col].dtype) for col in data.columns],
                'Unique Count': [f"{int(data[col].nunique()):,}" for col in data.columns]
            })
            st.dataframe(col_info, width='stretch')
        
        st.divider()
        
        st.subheader("🤖 AI-Powered Natural Language Query")
        st.info("Ask questions in natural language. Works with ANY dataset!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(data):,}")
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Status", "✅ Ready")
        
        with st.form("query_form", clear_on_submit=True):
            user_question = st.text_area("Ask about your data:", placeholder="Example: Which team had the most points?", height=100)
            submit_button = st.form_submit_button("📊 Analyze", use_container_width=True)
        
        if submit_button and user_question:
            with st.spinner("🧠 AI analyzing..."):
                query_params, ai_error = ai_understand_query(user_question, data, timeout_seconds, temperature, categorized_columns)
            
            if ai_error:
                st.error(f"❌ Error: {ai_error}")
            elif query_params:
                if show_steps:
                    with st.expander("🔍 AI Understanding"):
                        st.write(f"**Question:** {user_question}")
                        st.json({
                            "Query Type": query_params.get('query_type'),
                            "Metric": query_params.get('value_column'),
                            "Dimension": query_params.get('category_column'),
                            "Time Column": query_params.get('time_column'),
                            "Aggregation": query_params.get('aggregation_function'),
                        })
                
                with st.spinner("📊 Calculating..."):
                    result, error = execute_query(data, query_params)
                
                if error and result is None:
                    st.error(f"❌ Error: {error}")
                else:
                    st.success("✅ Results!")
                    if isinstance(result, tuple):
                        result_table, total = result
                        st.metric("Total", f"{total:,.2f}")
                        st.dataframe(result_table, width='stretch')
                    elif isinstance(result, pd.DataFrame):
                        st.dataframe(result, width='stretch')
        
        import os as os_module
        if os_module.path.exists(temp_path):
            os_module.remove(temp_path)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

import atexit
def cleanup():
    import os, glob
    for f in glob.glob("temp_*"):
        try:
            os.remove(f)
        except:
            pass

atexit.register(cleanup)