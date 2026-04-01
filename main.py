import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import requests
import json
import re

load_dotenv()

st.set_page_config(page_title="AI Data Analyzer Pro - SALES TEST", layout="wide")
st.title("📊 AI Data Analyzer Pro - Sales Data Accuracy Testing")
st.write("Target: 95%+ accuracy on sales dataset with data validation")

OLLAMA_API = "http://localhost:11434/api/generate"

# Column definitions
FINANCIAL_COLUMNS = ['Spend', 'Savings', 'Revenue', 'Profit', 'Marketing_Spend']
PERCENTAGE_COLUMNS = ['KPI_%', 'Profit_Margin_%', 'Return_Rate_%', 'Employee_Engagement_%', 'Customer_Satisfaction_Score']
COUNT_COLUMNS = ['Units_Sold']
DATE_COLUMNS = ['Date', 'Year', 'Quarter', 'Month']
DIMENSION_COLUMNS = ['Geo', 'Country', 'Sales_Rep', 'Customer', 'Category', 'Product']

ALL_METRICS = FINANCIAL_COLUMNS + PERCENTAGE_COLUMNS + COUNT_COLUMNS

# TEST SUITES
PHASE_1_TESTS = [
    ("What was the total revenue in 2024?", "filter_by_time", "Revenue", None, "2024"),
    ("Which product had the highest KPI_%?", "best_worst_by_category", "KPI_%", "Product", None),
    ("Show me the average spend per country", "best_worst_by_category", "Spend", "Country", None),
    ("What sales rep had the lowest Profit_Margin_%?", "best_worst_by_category", "Profit_Margin_%", "Sales_Rep", None),
    ("How much did we spend in Q3?", "filter_by_time", "Spend", None, "3"),
]

PHASE_2_EDGE_CASES = [
    ("revenue for 2024", "filter_by_time", "Revenue", None, "2024"),
    ("Q1 spending", "filter_by_time", "Spend", None, "1"),
    ("Q2 savings", "filter_by_time", "Savings", None, "2"),
    ("total profit by category", "best_worst_by_category", "Profit", "Category", None),
    ("average KPI per product", "best_worst_by_category", "KPI_%", "Product", None),
    ("which country had the most revenue?", "best_worst_by_category", "Revenue", "Country", None),
    ("top customer by spending", "best_worst_by_category", "Spend", "Customer", None),
    ("what was total units sold?", "total_by_time", "Units_Sold", None, None),
    ("best employee engagement score", "total_by_time", "Employee_Engagement_%", None, None),
    ("minimum profit margin", "total_by_time", "Profit_Margin_%", None, None),
]

PHASE_3_COMPLEX = [
    ("worst return rate by product", "best_worst_by_category", "Return_Rate_%", "Product", None),
    ("how much revenue did we make?", "total_by_time", "Revenue", None, None),
    ("highest marketing spend by geo", "best_worst_by_category", "Marketing_Spend", "Geo", None),
    ("lowest profit margin among sales reps", "best_worst_by_category", "Profit_Margin_%", "Sales_Rep", None),
    ("total savings across all customers", "total_by_time", "Savings", None, None),
    ("which sales rep had the highest KPI?", "best_worst_by_category", "KPI_%", "Sales_Rep", None),
]

def profile_columns(data):
    """Intelligently profile columns to determine their type and purpose."""
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
        
        if dtype in ['int64', 'float64']:
            profile['numeric'] = True
            profile['min'] = float(data[col].min()) if not data[col].isnull().all() else None
            profile['max'] = float(data[col].max()) if not data[col].isnull().all() else None
            profile['mean'] = float(data[col].mean()) if not data[col].isnull().all() else None
            profile['std'] = float(data[col].std()) if not data[col].isnull().all() else None
            
            if profile['max'] and profile['min'] is not None:
                if profile['max'] <= 100 and profile['min'] >= 0 and '%' in col_lower:
                    profile['inferred_type'] = 'metric_percentage'
                    profile['synonyms'].extend(['percentage', 'percent', 'pct', '%'])
            
            if any(word in col_lower for word in ['count', 'quantity', 'qty', 'units', 'items', 'volume', 'total', 'number']):
                profile['inferred_type'] = 'metric_count'
                profile['synonyms'].extend(['count', 'quantity', 'qty', 'units', 'items', 'volume', 'total', 'number'])
            
            elif any(word in col_lower for word in ['spend', 'cost', 'revenue', 'sales', 'profit', 'income', 'price', 'amount', 'value', 'fees', 'budget', 'points', 'score', 'goals']):
                profile['inferred_type'] = 'metric_financial'
                profile['synonyms'].extend(['spend', 'cost', 'revenue', 'sales', 'profit', 'income', 'amount', 'value', 'fees', 'budget', 'points', 'score', 'goals'])
            
            else:
                profile['inferred_type'] = 'metric'
                profile['synonyms'].extend(['value', 'amount', 'total', 'sum', 'metric'])
        
        else:
            profile['numeric'] = False
            cardinality = profile['unique_count']
            total_rows = len(data)
            cardinality_ratio = cardinality / total_rows if total_rows > 0 else 0
            
            is_likely_dimension = any(word in col_lower for word in [
                'team', 'player', 'product', 'customer', 'country', 'region', 'sport', 'league',
                'category', 'type', 'group', 'name', 'sales_rep', 'rep', 'agent', 'user',
                'brand', 'manufacturer', 'department', 'segment', 'state', 'geo', 'location',
                'athlete', 'coach', 'company', 'organization', 'division', 'conference'
            ])
            
            if is_likely_dimension:
                profile['inferred_type'] = 'dimension'
                profile['synonyms'].extend(['category', 'type', 'group', 'classification', 'segment'])
                
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
            
            elif cardinality_ratio > 0.5:
                profile['inferred_type'] = 'id_or_name'
                profile['synonyms'].extend(['name', 'id', 'identifier', 'description', 'title'])
            
            elif cardinality <= 100:
                profile['inferred_type'] = 'dimension'
                profile['synonyms'].extend(['category', 'type', 'group', 'classification', 'segment'])
                
                if any(word in col_lower for word in ['region', 'geo', 'location', 'country', 'state', 'area', 'territory']):
                    profile['synonyms'].extend(['region', 'geo', 'location', 'country', 'state', 'area', 'territory'])
                if any(word in col_lower for word in ['product', 'item', 'sku', 'brand', 'line']):
                    profile['synonyms'].extend(['product', 'item', 'sku', 'brand', 'line'])
                if any(word in col_lower for word in ['customer', 'client', 'account', 'company', 'business', 'organization']):
                    profile['synonyms'].extend(['customer', 'client', 'account', 'company', 'business', 'organization'])
                if any(word in col_lower for word in ['sales_rep', 'sales rep', 'rep', 'salesman', 'agent', 'representative', 'employee', 'staff']):
                    profile['synonyms'].extend(['sales_rep', 'sales rep', 'rep', 'salesman', 'agent', 'representative', 'employee', 'staff'])
                if any(word in col_lower for word in ['category', 'type', 'class', 'segment', 'department']):
                    profile['synonyms'].extend(['category', 'type', 'class', 'segment', 'department'])
            
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

def smart_column_match(question, available_columns):
    """Smart exact matching - prioritize LONGER column names (more specific)."""
    if not available_columns or not question:
        return None
    
    question_lower = question.lower()
    question_words = re.findall(r'\b\w+\b', question_lower)
    
    for col in available_columns:
        col_lower = col.lower()
        if col_lower in question_lower:
            return col
    
    for col in available_columns:
        col_clean = col.lower().replace('_', ' ')
        if col_clean in question_lower:
            return col
    
    sorted_cols = sorted(available_columns, key=lambda x: len(x), reverse=True)
    for col in sorted_cols:
        col_lower = col.lower()
        if col_lower in question_words:
            return col
    
    return None

def find_best_metric_strict(question, categorized_columns):
    """Strict metric detection with special handling for ambiguous metrics."""
    if not categorized_columns['metrics']:
        return None
    
    question_lower = question.lower()
    available_metrics = list(categorized_columns['metrics'].keys())
    
    if 'margin' in question_lower:
        for metric in available_metrics:
            if 'Profit_Margin' in metric:
                return metric
    
    if 'engagement' in question_lower:
        for metric in available_metrics:
            if 'Engagement' in metric:
                return metric
    
    if 'return' in question_lower:
        for metric in available_metrics:
            if 'Return_Rate' in metric:
                return metric
    
    if 'marketing' in question_lower:
        for metric in available_metrics:
            if 'Marketing' in metric:
                return metric
    
    if 'kpi' in question_lower:
        for metric in available_metrics:
            if 'KPI' in metric:
                return metric
    
    return smart_column_match(question, available_metrics)

def find_best_dimension_strict(question, categorized_columns):
    """Strict dimension detection using EXACT column matching."""
    if not question:
        return None
    
    has_dimension_keyword = any(word in question.lower() for word in [
        'which', 'by ', 'per ', 'for each', 'by each', 'top ', 'best ', 'worst ', 
        'highest', 'lowest', 'most', 'least'
    ])
    
    if not has_dimension_keyword:
        return None
    
    available_dims = list(categorized_columns['dimensions'].keys()) + list(categorized_columns['ids'].keys())
    
    if not available_dims:
        return None
    
    return smart_column_match(question, available_dims)

def find_best_time_column(question, categorized_columns):
    """Smart time column detection - SKIP Date, prioritize Year/Quarter."""
    if not categorized_columns['time'] or not question:
        return None
    
    question_lower = question.lower()
    time_cols = list(categorized_columns['time'].keys())
    
    # REMOVE Date column from consideration - we only want Year, Quarter, Month
    time_cols = [col for col in time_cols if 'date' not in col.lower()]
    
    if not time_cols:
        return None
    
    # PRIORITY 1: Quarter mentions - match Quarter column
    if re.search(r'\b(q[1-4]|quarter)\b', question_lower):
        for col in time_cols:
            if 'quarter' in col.lower():
                return col
    
    # PRIORITY 2: Month mentions
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|month)\b', question_lower):
        for col in time_cols:
            if 'month' in col.lower():
                return col
    
    # PRIORITY 3: Year mentions - match Year column
    if re.search(r'\b(19|20)\d{2}\b|year\b', question_lower):
        for col in time_cols:
            if 'year' in col.lower():
                return col
    
    # Default to first available (non-Date) time column
    if time_cols:
        return time_cols[0]
    
    return None

def detect_time_period_value(question):
    """Detect time period values from the current question."""
    question_lower = question.lower()
    words = re.findall(r'\b\w+\b', question_lower)
    
    for word in words:
        if word in ['q1', 'q2', 'q3', 'q4']:
            return word[1]  # Return just the number: 1, 2, 3, 4
    
    full_months = {
        'january': 'January', 'february': 'February', 'march': 'March',
        'april': 'April', 'may': 'May', 'june': 'June',
        'july': 'July', 'august': 'August', 'september': 'September',
        'october': 'October', 'november': 'November', 'december': 'December'
    }
    
    for word in words:
        if word in full_months:
            return full_months[word]
    
    month_abbrs = {
        'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
        'may': 'May', 'jun': 'June', 'jul': 'July', 'aug': 'August',
        'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December'
    }
    
    for word in words:
        if word in month_abbrs:
            return month_abbrs[word]
    
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

def detect_aggregation_function(question, value_col, category_col=None):
    """Smart aggregation detection."""
    question_lower = question.lower()
    col_type = get_column_type(value_col)
    
    if category_col:
        if any(word in question_lower for word in ['lowest', 'minimum', 'min', 'least']):
            if col_type == 'percentage':
                return 'mean'
            else:
                return 'min'
        elif any(word in question_lower for word in ['highest', 'maximum', 'max', 'most', 'top', 'best']):
            if col_type == 'percentage':
                return 'mean'
            else:
                return 'max'
    
    if any(word in question_lower for word in ['average', 'avg', 'mean', 'per ']):
        return 'mean'
    if any(word in question_lower for word in ['total', 'sum']):
        return 'sum'
    
    return None

def normalize_filter_value(filter_value, filter_column, data):
    """Normalize filter values to match actual data format."""
    if not filter_value or not filter_column:
        return filter_value
    
    filter_value_str = str(filter_value).lower().strip()
    
    if 'quarter' in filter_column.lower():
        if filter_value_str.startswith('q'):
            return filter_value_str[1]
        return str(filter_value)
    
    return str(filter_value)

def ai_understand_query(user_question, data, timeout_seconds, temperature, categorized_columns):
    """Use strict column detection with NO AI hallucinations."""
    
    smart_metric = find_best_metric_strict(user_question, categorized_columns)
    smart_dimension = find_best_dimension_strict(user_question, categorized_columns)
    smart_time = find_best_time_column(user_question, categorized_columns)
    pre_time_value = detect_time_period_value(user_question)
    
    if not smart_metric:
        return None, f"❌ Could not identify metric. Available: {', '.join(categorized_columns['metrics'].keys())}"
    
    # Normalize time value if we have both time column and value
    if pre_time_value and smart_time:
        pre_time_value = normalize_filter_value(pre_time_value, smart_time, data)
    
    agg_func = detect_aggregation_function(user_question, smart_metric, smart_dimension)
    question_lower = user_question.lower()
    
    # PRIORITY 1: Group by dimension
    if smart_dimension and (any(word in question_lower for word in ['which', 'highest', 'lowest', 'best', 'worst', 'top', 'most', 'by ', 'per ']) or ' by ' in question_lower):
        query_params = {
            "query_type": "best_worst_by_category",
            "category_column": smart_dimension,
            "value_column": smart_metric,
            "aggregation_function": agg_func,
            "time_column": None,
            "filter_value": None,
        }
    
    # PRIORITY 2: Filter by time (if we have time column AND time value detected)
    elif smart_time and pre_time_value:
        query_params = {
            "query_type": "filter_by_time",
            "time_column": smart_time,
            "value_column": smart_metric,
            "filter_value": pre_time_value,
            "category_column": None,
            "aggregation_function": None,
        }
    
    # PRIORITY 3: Total by time
    else:
        query_params = {
            "query_type": "total_by_time",
            "time_column": smart_time,
            "value_column": smart_metric,
            "category_column": None,
            "filter_value": None,
            "aggregation_function": None,
        }
    
    return query_params, None

def execute_query_with_validation(data, query_params):
    """Execute query AND validate results against actual data."""
    try:
        query_type = query_params.get("query_type", "").lower()
        value_col = query_params.get("value_column")
        
        if not value_col or value_col not in data.columns:
            return None, f"Value column '{value_col}' not found", None
        
        if query_type == "best_worst_by_category":
            category_col = query_params.get("category_column")
            agg_func = query_params.get("aggregation_function")
            
            if not category_col or category_col not in data.columns:
                return None, f"Category column '{category_col}' not found", None
            
            if agg_func == 'mean':
                result = data.groupby(category_col)[value_col].mean().reset_index()
            elif agg_func == 'min':
                result = data.groupby(category_col)[value_col].min().reset_index()
            else:  # max
                result = data.groupby(category_col)[value_col].max().reset_index()
            
            result = result.sort_values(value_col, ascending=(agg_func == 'min'))
            result.columns = [category_col, f"{'Min' if agg_func == 'min' else 'Max' if agg_func == 'max' else 'Average'} {value_col}"]
            
            top_category = result.iloc[0][category_col] if len(result) > 0 else None
            top_value = result.iloc[0][result.columns[1]] if len(result) > 0 else None
            
            validation = {
                "query_type": query_type,
                "category_col": category_col,
                "value_col": value_col,
                "agg_func": agg_func or "max",
                "top_category": top_category,
                "top_value": round(top_value, 2) if isinstance(top_value, (int, float)) else top_value,
                "total_rows": len(data),
                "total_groups": len(result),
                "instruction": f"✅ Verify: Pivot table with '{category_col}' as rows, '{value_col}' as values ({agg_func or 'MAX'}). Top result: {top_category}: {round(top_value, 2) if isinstance(top_value, (int, float)) else top_value}",
            }
            
            return result, None, validation
        
        elif query_type == "filter_by_time":
            time_col = query_params.get("time_column")
            filter_val = query_params.get("filter_value")
            
            if not time_col or time_col not in data.columns:
                return None, f"Time column '{time_col}' not found", None
            
            filtered = data[data[time_col].astype(str) == str(filter_val)]
            
            if len(filtered) == 0:
                available_vals = sorted(data[time_col].dropna().unique())
                return None, f"No data for {time_col}={filter_val}. Available values: {available_vals}", None
            
            total = filtered[value_col].sum()
            result = pd.DataFrame({f"Total {value_col}": [round(total, 2)]})
            
            dim_cols = [c for c in data.columns if c not in [time_col, value_col] and data[c].dtype == 'object']
            breakdown = {}
            if dim_cols:
                breakdown = filtered.groupby(dim_cols[0])[value_col].sum().round(2).to_dict()
            
            validation = {
                "query_type": query_type,
                "time_col": time_col,
                "filter_val": filter_val,
                "value_col": value_col,
                "total_result": round(total, 2),
                "rows_included": len(filtered),
                "rows_total": len(data),
                "breakdown": breakdown,
                "instruction": f"✅ Verify: In Excel, filter {time_col} = {filter_val}, SUM {value_col}. Should equal {round(total, 2)}",
            }
            
            return result, None, validation
        
        else:  # total_by_time
            total = data[value_col].sum()
            result = pd.DataFrame({f"Total {value_col}": [round(total, 2)]})
            
            time_col = query_params.get("time_column")
            breakdown = {}
            if time_col and time_col in data.columns:
                breakdown = data.groupby(time_col)[value_col].sum().round(2).to_dict()
            
            validation = {
                "query_type": query_type,
                "value_col": value_col,
                "total_result": round(total, 2),
                "total_rows": len(data),
                "breakdown": breakdown,
                "null_values": int(data[value_col].isnull().sum()),
                "instruction": f"✅ Verify: SUM all {value_col} column. Should equal {round(total, 2)}",
            }
            
            return result, None, validation
    
    except Exception as e:
        return None, f"Error: {str(e)}", None

def run_test_suite_with_execution(data, categorized_columns, test_suite, phase_name):
    """Run test suite AND execute queries to verify results."""
    results = []
    passed = 0
    
    for question, expected_type, expected_metric, expected_dim, expected_time in test_suite:
        query_params, error = ai_understand_query(question, data, 600, 0.3, categorized_columns)
        
        if error:
            results.append({
                "Question": question[:50],
                "Status": "❌ ERROR",
                "Expected": expected_type,
                "Got": "ERROR",
                "Execution": error[:40],
            })
            continue
        
        type_match = query_params.get("query_type") == expected_type
        metric_match = query_params.get("value_column") == expected_metric
        dim_match = (query_params.get("category_column") == expected_dim) if expected_dim else True
        
        exec_result, exec_error, validation = execute_query_with_validation(data, query_params)
        
        execution_ok = exec_error is None and exec_result is not None
        
        if type_match and metric_match and dim_match and execution_ok:
            results.append({
                "Question": question[:50],
                "Status": "✅ PASS",
                "Expected": expected_type,
                "Got": query_params.get('query_type'),
                "Execution": "✅ OK",
            })
            passed += 1
        else:
            issue_list = []
            if not type_match:
                issue_list.append(f"Type: {expected_type}→{query_params.get('query_type')}")
            if not metric_match:
                issue_list.append(f"Metric: {expected_metric}→{query_params.get('value_column')}")
            if not dim_match:
                issue_list.append(f"Dim: {expected_dim}→{query_params.get('category_column')}")
            if not execution_ok:
                issue_list.append(f"Exec: {exec_error[:30] if exec_error else 'Failed'}")
            
            results.append({
                "Question": question[:50],
                "Status": "❌ FAIL",
                "Expected": expected_type,
                "Got": query_params.get('query_type'),
                "Execution": " | ".join(issue_list) if issue_list else "Unknown",
            })
    
    st.subheader(f"📊 {phase_name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(test_suite))
    with col2:
        st.metric("✅ Passed", passed)
    with col3:
        st.metric("❌ Failed", len(test_suite) - passed)
    with col4:
        accuracy = (passed / len(test_suite)) * 100 if test_suite else 0
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    with st.expander("📋 Detailed Results"):
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, height=300)
    
    return passed, len(test_suite)

# MAIN APP
st.sidebar.header("⚙️ Settings")
show_column_detection = st.sidebar.checkbox("Show Column Detection", value=True)

uploaded_file = st.file_uploader("📁 Upload Sales Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        if temp_path.endswith('.csv'):
            data = pd.read_csv(temp_path)
        else:
            data = pd.read_excel(temp_path, engine='openpyxl')
        
        column_profiles = profile_columns(data)
        categorized_columns = categorize_columns(column_profiles)
        
        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
        
        if show_column_detection:
            with st.expander("🔍 Detected Columns"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("**Metrics:**")
                    for m in list(categorized_columns['metrics'].keys()):
                        st.write(f"- {m}")
                with col2:
                    st.write("**Dimensions:**")
                    for d in list(categorized_columns['dimensions'].keys()):
                        st.write(f"- {d}")
                with col3:
                    st.write("**Time (excluding Date):**")
                    time_cols = [t for t in categorized_columns['time'].keys() if 'date' not in t.lower()]
                    for t in time_cols:
                        st.write(f"- {t}")
                with col4:
                    st.write("**Sample Values:**")
                    if time_cols:
                        first_time_col = time_cols[0]
                        for val in data[first_time_col].unique()[:5]:
                            st.write(f"- {val}")
        
        st.divider()
        
        st.subheader("📋 Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(data):,}")
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Status", "✅ Ready")
        
        st.dataframe(data.head(10), use_container_width=True, height=250)
        
        st.divider()
        
        st.header("🧪 Test Suites")
        
        p1_passed, p1_total = run_test_suite_with_execution(data, categorized_columns, PHASE_1_TESTS, "Phase 1: Core Queries")
        st.divider()
        
        p2_passed, p2_total = run_test_suite_with_execution(data, categorized_columns, PHASE_2_EDGE_CASES, "Phase 2: Edge Cases")
        st.divider()
        
        p3_passed, p3_total = run_test_suite_with_execution(data, categorized_columns, PHASE_3_COMPLEX, "Phase 3: Complex Queries")
        
        st.divider()
        st.header("📈 Overall Accuracy Report")
        
        total_passed = p1_passed + p2_passed + p3_passed
        total_tests = p1_total + p2_total + p3_total
        overall_accuracy = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("✅ Passed", total_passed)
        with col3:
            st.metric("❌ Failed", total_tests - total_passed)
        with col4:
            st.metric("Accuracy", f"{overall_accuracy:.1f}%")
        with col5:
            if overall_accuracy >= 95:
                st.metric("Status", "🎉 READY", delta="95%+")
            else:
                st.metric("Status", f"⚠️ {95 - overall_accuracy:.1f}% away")
        
        st.divider()
        st.header("🔬 Manual Query Tester")
        st.write("Test any custom question and verify against Excel:")
        
        with st.form("manual_query"):
            test_question = st.text_area("Question:", placeholder="Example: Which product had the highest KPI_%?", height=80)
            submit = st.form_submit_button("🧪 Test Query", use_container_width=True)
        
        if submit and test_question:
            query_params, error = ai_understand_query(test_question, data, 600, 0.3, categorized_columns)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ Query Detected!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Detection:**")
                    st.json(query_params)
                
                with col2:
                    st.write("**Result:**")
                    result, exec_error, validation = execute_query_with_validation(data, query_params)
                    if exec_error:
                        st.error(exec_error)
                    else:
                        st.dataframe(result, use_container_width=True)
                
                if validation and not exec_error:
                    st.divider()
                    st.subheader("✅ How to Verify in Excel")
                    
                    if query_params.get("query_type") == "best_worst_by_category":
                        st.write(f"**📊 Result Details:**")
                        st.write(f"• Category Column: `{validation['category_col']}`")
                        st.write(f"• Value Column: `{validation['value_col']}`")
                        st.write(f"• Aggregation: `{validation['agg_func'].upper()}`")
                        st.write(f"• Total Groups: `{validation['total_groups']}`")
                        st.write("")
                        st.write(f"**🏆 Top Result:**")
                        st.write(f"• {validation['category_col']}: **`{validation['top_category']}`**")
                        st.write(f"• {validation['agg_func'].upper()} {validation['value_col']}: **`{validation['top_value']}`**")
                        st.write("")
                        st.info(validation['instruction'])
                    
                    elif query_params.get("query_type") == "filter_by_time":
                        st.write(f"**📊 Result Details:**")
                        st.write(f"• Time Column: `{validation['time_col']}`")
                        st.write(f"• Filter Value: `{validation['filter_val']}`")
                        st.write(f"• Value Column: `{validation['value_col']}`")
                        st.write(f"• Rows Matched: `{validation['rows_included']} / {validation['rows_total']}`")
                        st.write("")
                        st.write(f"**💰 Total Result:**")
                        st.write(f"• **`{validation['total_result']}`**")
                        if validation['breakdown']:
                            st.write(f"• Breakdown: {validation['breakdown']}")
                        st.write("")
                        st.info(validation['instruction'])
                    
                    else:  # total
                        st.write(f"**📊 Result Details:**")
                        st.write(f"• Value Column: `{validation['value_col']}`")
                        st.write(f"• Total Rows: `{validation['total_rows']}`")
                        st.write(f"• Null Values: `{validation['null_values']}`")
                        st.write("")
                        st.write(f"**💰 Grand Total:**")
                        st.write(f"• **`{validation['total_result']}`**")
                        if validation['breakdown']:
                            st.write(f"• Breakdown by period: {validation['breakdown']}")
                        st.write("")
                        st.info(validation['instruction'])
        
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👉 Upload a sales file")