import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import json
import re

load_dotenv()

st.set_page_config(page_title="AI Data Analyzer Pro", layout="wide")
st.title("📊 AI Data Analyzer Pro - Sales Data Accuracy Testing")
st.write("Target: 95%+ accuracy with proper Year/Quarter/Month filtering")

# ============================================================================
# TEST SUITES - CORRECTED
# ============================================================================
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

# ============================================================================
# COLUMN PROFILING & CATEGORIZATION
# ============================================================================
def profile_columns(data):
    """Profile all columns to determine their type and purpose."""
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
            'sample_values': list(data[col].dropna().unique()[:3]) if dtype == 'object' else None,
        }
        
        # PRIORITY 1: Exact TIME column names
        if col in ['Date', 'Year', 'Quarter', 'Month', 'Week', 'Day', 'Timestamp']:
            profile['inferred_type'] = 'time'
            profile['time_values'] = sorted(data[col].dropna().unique()[:10])
            column_profiles[col] = profile
            continue
        
        # PRIORITY 2: Known DIMENSION columns
        known_dimensions = ['Geo', 'Country', 'Sales_Rep', 'Customer', 'Category', 'Product']
        if col in known_dimensions:
            profile['inferred_type'] = 'dimension'
            profile['dimension_cardinality'] = len(data[col].unique())
            column_profiles[col] = profile
            continue
        
        # PRIORITY 3: Known METRIC columns
        known_metrics = [
            'Revenue', 'Spend', 'Savings', 'Profit', 'Marketing_Spend',
            'KPI_%', 'Profit_Margin_%', 'Return_Rate_%', 'Employee_Engagement_%', 'Customer_Satisfaction_Score',
            'Units_Sold'
        ]
        if col in known_metrics:
            profile['inferred_type'] = 'metric'
            profile['metric_type'] = 'percentage' if '%' in col else 'financial'
            profile['metric_min'] = float(data[col].min()) if not data[col].isnull().all() else None
            profile['metric_max'] = float(data[col].max()) if not data[col].isnull().all() else None
            profile['metric_mean'] = float(data[col].mean()) if not data[col].isnull().all() else None
            column_profiles[col] = profile
            continue
        
        # PRIORITY 4: Infer from data type
        if dtype in ['int64', 'float64']:
            profile['numeric'] = True
            profile['metric_min'] = float(data[col].min()) if not data[col].isnull().all() else None
            profile['metric_max'] = float(data[col].max()) if not data[col].isnull().all() else None
            profile['inferred_type'] = 'metric'
            column_profiles[col] = profile
            continue
        
        # PRIORITY 5: Other strings (likely dimensions)
        if dtype == 'object':
            profile['numeric'] = False
            cardinality = profile['unique_count']
            total_rows = len(data)
            cardinality_ratio = cardinality / total_rows if total_rows > 0 else 0
            profile['cardinality_ratio'] = cardinality_ratio
            
            if cardinality_ratio > 0.5:
                profile['inferred_type'] = 'id_or_name'
            else:
                profile['inferred_type'] = 'dimension'
                profile['dimension_cardinality'] = cardinality
            
            column_profiles[col] = profile
            continue
        
        profile['inferred_type'] = 'unknown'
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

# ============================================================================
# QUERY DETECTION FUNCTIONS
# ============================================================================
def find_metric(question, available_metrics):
    """Find metric column from question - handles special cases."""
    question_lower = question.lower()
    
    # SPECIAL CASE 1: "margin" -> "Profit_Margin_%"
    if 'margin' in question_lower:
        for m in available_metrics:
            if 'Profit_Margin' in m:
                return m
    
    # SPECIAL CASE 2: "engagement" -> "Employee_Engagement_%"
    if 'engagement' in question_lower:
        for m in available_metrics:
            if 'Engagement' in m:
                return m
    
    # SPECIAL CASE 3: "return" -> "Return_Rate_%"
    if 'return' in question_lower:
        for m in available_metrics:
            if 'Return_Rate' in m:
                return m
    
    # SPECIAL CASE 4: "marketing" -> "Marketing_Spend"
    if 'marketing' in question_lower:
        for m in available_metrics:
            if 'Marketing' in m:
                return m
    
    # SPECIAL CASE 5: "kpi" -> "KPI_%"
    if 'kpi' in question_lower:
        for m in available_metrics:
            if 'KPI' in m:
                return m
    
    # SPECIAL CASE 6: "units" or "sold" -> "Units_Sold"
    if 'units' in question_lower or 'sold' in question_lower:
        for m in available_metrics:
            if 'Units' in m or 'Sold' in m:
                return m
    
    # DEFAULT: Exact substring match (longest first)
    for m in sorted(available_metrics, key=len, reverse=True):
        if m.lower() in question_lower:
            return m
    
    # FALLBACK: Word match
    words = re.findall(r'\b\w+\b', question_lower)
    for m in sorted(available_metrics, key=len, reverse=True):
        if m.lower() in words:
            return m
    
    return None

def find_dimension(question, available_dims):
    """Find dimension column from question - IMPROVED detection."""
    if not available_dims:
        return None
    
    question_lower = question.lower()
    
    # CHECK: Must have dimension keywords (comprehensive list)
    dimension_keywords = [
        'which',      # "which sales rep"
        'by ',        # "by sales rep" or "by category"
        'per ',       # "per country"
        'top ',       # "top customer"
        'best ',      # "best product"
        'worst ',     # "worst product"
        'highest',    # "highest ... by geo"
        'lowest',     # "lowest ... by product" or "lowest ... among sales_rep"
        'most',       # "most revenue by category"
        'among',      # "among sales reps" or "among customers"
        'for each',   # "for each product"
    ]
    
    has_dim_keyword = any(keyword in question_lower for keyword in dimension_keywords)
    
    if not has_dim_keyword:
        return None
    
    # PRIORITY 1: Exact substring match (longest names first for "Sales_Rep" > "Rep")
    for d in sorted(available_dims, key=len, reverse=True):
        if d.lower() in question_lower:
            return d
    
    # PRIORITY 2: Word match (longest names first)
    words = re.findall(r'\b\w+\b', question_lower)
    for d in sorted(available_dims, key=len, reverse=True):
        # Match the word part: "Sales_Rep" matches word "sales" or "rep"
        d_parts = d.lower().split('_')
        for part in d_parts:
            if part in words:
                return d
    
    return None

def find_time_column(question, available_time_cols):
    """Find TIME column - requires exact matching for Year/Quarter/Month."""
    if not available_time_cols:
        return None
    
    question_lower = question.lower()
    
    # PRIORITY 1: Quarter mentions
    if re.search(r'\b(q[1-4]|quarter)\b', question_lower):
        if 'Quarter' in available_time_cols:
            return 'Quarter'
    
    # PRIORITY 2: Month mentions
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', question_lower):
        if 'Month' in available_time_cols:
            return 'Month'
    
    # PRIORITY 3: Year mentions
    if re.search(r'\b(19|20)\d{2}\b|year\b', question_lower):
        if 'Year' in available_time_cols:
            return 'Year'
    
    # PRIORITY 4: Prefer Year/Quarter/Month over Date
    for col in ['Year', 'Quarter', 'Month']:
        if col in available_time_cols:
            return col
    
    # LAST RESORT: Return first time column
    return available_time_cols[0] if available_time_cols else None

def find_time_value(question):
    """Extract time period value from question."""
    question_lower = question.lower()
    words = re.findall(r'\b\w+\b', question_lower)
    
    # EXTRACT: Quarter (Q1, Q2, Q3, Q4)
    for word in words:
        if word in ['q1', 'q2', 'q3', 'q4']:
            return word[1]  # Return: 1, 2, 3, 4
    
    # EXTRACT: Month names
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7,
        'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for word in words:
        if word in months:
            return str(months[word])
    
    # EXTRACT: Year (2023, 2024, etc)
    for word in words:
        if re.match(r'^(19|20)\d{2}$', word):
            return word
    
    return None

def detect_aggregation(question, metric_col):
    """Detect aggregation function from question keywords."""
    question_lower = question.lower()
    
    # MIN: lowest, minimum, min, least, worst
    if any(w in question_lower for w in ['lowest', 'minimum', 'min', 'least', 'worst']):
        return 'min'
    
    # MAX: highest, maximum, max, most, top, best
    if any(w in question_lower for w in ['highest', 'maximum', 'max', 'most', 'top', 'best']):
        return 'max'
    
    # MEAN: average, avg, mean, per
    if any(w in question_lower for w in ['average', 'avg', 'mean', 'per ']):
        return 'mean'
    
    # SUM: total, sum
    if any(w in question_lower for w in ['total', 'sum', 'all']):
        return 'sum'
    
    return 'max'  # DEFAULT

def understand_query(question, data, categorized_cols):
    """Main query understanding logic - determines query type and parameters."""
    
    # STEP 1: Find metric
    metric = find_metric(question, list(categorized_cols['metrics'].keys()))
    if not metric:
        available = ', '.join(list(categorized_cols['metrics'].keys())[:10])
        return None, f"❌ Could not find metric. Available: {available}"
    
    # STEP 2: Find dimension
    dimension = find_dimension(question, list(categorized_cols['dimensions'].keys()))
    
    # STEP 3: Find time column and time value
    time_col = find_time_column(question, list(categorized_cols['time'].keys()))
    time_val = find_time_value(question)
    
    # STEP 4: Detect aggregation
    agg_func = detect_aggregation(question, metric)
    
    question_lower = question.lower()
    
    # DECISION TREE: Determine query type
    
    # TYPE A: Group by dimension (highest priority when dimension + aggregation keywords present)
    if dimension and any(w in question_lower for w in ['which', 'by ', 'top ', 'best ', 'worst ', 'highest', 'lowest', 'most', 'per ', 'among']):
        return {
            "query_type": "best_worst_by_category",
            "category_column": dimension,
            "value_column": metric,
            "aggregation_function": agg_func,
        }, None
    
    # TYPE B: Filter by time (when time value detected)
    elif time_col and time_val:
        return {
            "query_type": "filter_by_time",
            "time_column": time_col,
            "value_column": metric,
            "filter_value": time_val,
        }, None
    
    # TYPE C: Total/Aggregate (fallback - can have aggregation like MAX, MIN)
    else:
        return {
            "query_type": "total_by_time",
            "time_column": time_col,
            "value_column": metric,
            "aggregation_function": agg_func,
        }, None

# ============================================================================
# QUERY EXECUTION & VALIDATION
# ============================================================================
def execute_query(data, params):
    """Execute detected query and return results with detailed validation."""
    try:
        query_type = params.get("query_type")
        value_col = params.get("value_column")
        
        # VALIDATION: Check value column exists
        if not value_col or value_col not in data.columns:
            return None, f"❌ Value column '{value_col}' not found in data", None
        
        if query_type == "best_worst_by_category":
            category_col = params.get("category_column")
            agg = params.get("aggregation_function", "max")
            
            # VALIDATION: Check category column exists
            if not category_col or category_col not in data.columns:
                return None, f"❌ Category column '{category_col}' not found", None
            
            # EXECUTE: Group by category and aggregate
            if agg == 'mean':
                result = data.groupby(category_col)[value_col].mean().reset_index()
            elif agg == 'min':
                result = data.groupby(category_col)[value_col].min().reset_index()
            elif agg == 'max':
                result = data.groupby(category_col)[value_col].max().reset_index()
            else:  # sum
                result = data.groupby(category_col)[value_col].sum().reset_index()
            
            # SORT: Best/worst first
            result = result.sort_values(value_col, ascending=(agg == 'min'))
            result.columns = [category_col, f"{agg.upper()} {value_col}"]
            
            # EXTRACT: Top result for validation
            top_category = result.iloc[0][category_col] if len(result) > 0 else None
            top_value = result.iloc[0][result.columns[1]] if len(result) > 0 else None
            
            # BUILD: Detailed validation
            validation = {
                "type": "category_aggregation",
                "category_col": category_col,
                "value_col": value_col,
                "agg_func": agg,
                "top_result_category": top_category,
                "top_result_value": round(top_value, 2) if isinstance(top_value, (int, float)) else top_value,
                "total_groups": len(result),
                "result_sample": result.head(3).to_dict('records'),
                "verification_instruction": f"Create pivot table: {category_col} (rows) × {value_col} ({agg.upper()}). Top should be: {top_category}: {round(top_value, 2) if isinstance(top_value, (int, float)) else top_value}"
            }
            
            return result, None, validation
        
        elif query_type == "filter_by_time":
            time_col = params.get("time_column")
            filter_val = params.get("filter_value")
            
            # VALIDATION: Check time column exists
            if not time_col or time_col not in data.columns:
                return None, f"❌ Time column '{time_col}' not found", None
            
            # EXECUTE: Filter data
            filtered = data[data[time_col].astype(str) == str(filter_val)]
            
            # VALIDATION: Check if filter returned results
            if len(filtered) == 0:
                available_vals = sorted([str(v) for v in data[time_col].dropna().unique()])
                return None, f"❌ No data for {time_col}={filter_val}. Available: {available_vals[:5]}", None
            
            # EXECUTE: Calculate total
            total = filtered[value_col].sum()
            result = pd.DataFrame({f"Total {value_col}": [round(total, 2)]})
            
            # EXTRACT: Breakdown by dimension
            dim_cols = [c for c in data.columns if c not in [time_col, value_col] and data[c].dtype == 'object']
            breakdown = {}
            sample_records = []
            if dim_cols:
                breakdown = filtered.groupby(dim_cols[0])[value_col].sum().round(2).to_dict()
                sample_records = filtered.head(5).to_dict('records')
            
            # BUILD: Detailed validation
            validation = {
                "type": "time_filter",
                "time_col": time_col,
                "filter_val": filter_val,
                "value_col": value_col,
                "total_result": round(total, 2),
                "rows_matched": len(filtered),
                "rows_total": len(data),
                "percentage_matched": round((len(filtered) / len(data)) * 100, 1),
                "breakdown_by_dimension": breakdown,
                "sample_records": sample_records[:3],
                "verification_instruction": f"Filter {time_col}={filter_val}, SUM {value_col}. Should equal: {round(total, 2)}"
            }
            
            return result, None, validation
        
        else:  # total_by_time with optional aggregation
            agg = params.get("aggregation_function", "sum")
            time_col = params.get("time_column")
            
            # EXECUTE: Calculate aggregate
            if agg == 'max':
                total = data[value_col].max()
                agg_label = "MAX"
            elif agg == 'min':
                total = data[value_col].min()
                agg_label = "MIN"
            elif agg == 'mean':
                total = data[value_col].mean()
                agg_label = "AVERAGE"
            else:
                total = data[value_col].sum()
                agg_label = "TOTAL"
            
            result = pd.DataFrame({f"{agg_label} {value_col}": [round(total, 2)]})
            
            # EXTRACT: Breakdown by time period (if time column exists)
            breakdown_by_time = {}
            if time_col and time_col in data.columns:
                if agg == 'sum':
                    breakdown_by_time = data.groupby(time_col)[value_col].sum().round(2).to_dict()
                elif agg == 'mean':
                    breakdown_by_time = data.groupby(time_col)[value_col].mean().round(2).to_dict()
                elif agg == 'max':
                    breakdown_by_time = data.groupby(time_col)[value_col].max().round(2).to_dict()
                elif agg == 'min':
                    breakdown_by_time = data.groupby(time_col)[value_col].min().round(2).to_dict()
            
            # BUILD: Detailed validation
            validation = {
                "type": "total_aggregate",
                "value_col": value_col,
                "agg_func": agg,
                "total_result": round(total, 2),
                "total_rows": len(data),
                "null_values": int(data[value_col].isnull().sum()),
                "breakdown_by_time": breakdown_by_time,
                "verification_instruction": f"{agg_label}({value_col}) across all data. Should equal: {round(total, 2)}"
            }
            
            return result, None, validation
    
    except Exception as e:
        return None, f"❌ Execution error: {str(e)}", None

# ============================================================================
# TEST RUNNER
# ============================================================================
def run_tests(data, categorized_cols, test_suite, phase_name):
    """Run complete test suite with detailed results."""
    results = []
    passed = 0
    
    for idx, (question, expected_type, expected_metric, expected_dim, expected_time) in enumerate(test_suite):
        # DETECT: Query parameters
        params, error = understand_query(question, data, categorized_cols)
        
        if error:
            results.append({
                "Q": question[:45],
                "Status": "❌",
                "Expected": expected_type,
                "Got": "ERROR",
                "Issue": error[:50]
            })
            continue
        
        # VALIDATE: Check detection
        type_ok = params.get("query_type") == expected_type
        metric_ok = params.get("value_column") == expected_metric
        dim_ok = (params.get("category_column") == expected_dim) if expected_dim else True
        
        # EXECUTE: Query
        result, exec_error, validation = execute_query(data, params)
        exec_ok = exec_error is None and result is not None
        
        # DETERMINE: PASS/FAIL
        if type_ok and metric_ok and dim_ok and exec_ok:
            results.append({
                "Q": question[:45],
                "Status": "✅",
                "Expected": expected_type,
                "Got": params.get("query_type"),
                "Issue": "OK"
            })
            passed += 1
        else:
            issues = []
            if not type_ok:
                issues.append("Type")
            if not metric_ok:
                issues.append("Metric")
            if not dim_ok:
                issues.append("Dim")
            if not exec_ok:
                issues.append("Exec")
            
            results.append({
                "Q": question[:45],
                "Status": "❌",
                "Expected": expected_type,
                "Got": params.get("query_type") if params else "ERROR",
                "Issue": " | ".join(issues) if issues else "Unknown"
            })
    
    # DISPLAY: Results
    st.subheader(f"📊 {phase_name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(test_suite))
    with col2:
        st.metric("✅ Passed", passed)
    with col3:
        st.metric("❌ Failed", len(test_suite) - passed)
    with col4:
        acc = (passed / len(test_suite)) * 100 if test_suite else 0
        st.metric("Accuracy", f"{acc:.1f}%")
    
    with st.expander("📋 Detailed Results"):
        st.dataframe(pd.DataFrame(results), use_container_width=True, height=300)
    
    return passed, len(test_suite)

# ============================================================================
# MAIN APP
# ============================================================================
st.sidebar.header("⚙️ Settings")
show_detected = st.sidebar.checkbox("Show Column Detection", value=True)
show_manual_tester = st.sidebar.checkbox("Show Manual Tester", value=True)

uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # LOAD: Data
        if temp_path.endswith('.csv'):
            data = pd.read_csv(temp_path)
        else:
            data = pd.read_excel(temp_path, engine='openpyxl')
        
        # PROFILE: Columns
        column_profiles = profile_columns(data)
        categorized_cols = categorize_columns(column_profiles)
        
        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
        
        # DISPLAY: Column detection
        if show_detected:
            with st.expander("🔍 Column Detection Details"):
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    st.write("**📊 Metrics** ({})".format(len(categorized_cols['metrics'])))
                    for m in list(categorized_cols['metrics'].keys())[:12]:
                        st.caption(f"• {m}")
                
                with c2:
                    st.write("**🏷️ Dimensions** ({})".format(len(categorized_cols['dimensions'])))
                    for d in list(categorized_cols['dimensions'].keys())[:12]:
                        profile = column_profiles[d]
                        cardinality = profile.get('dimension_cardinality', '?')
                        st.caption(f"• {d} ({cardinality})")
                
                with c3:
                    st.write("**📅 Time Columns** ({})".format(len(categorized_cols['time'])))
                    for t in categorized_cols['time'].keys():
                        profile = column_profiles[t]
                        values = profile.get('time_values', [])
                        st.caption(f"• {t} ({len(values)} unique)")
                
                with c4:
                    st.write("**🆔 IDs/Names** ({})".format(len(categorized_cols['ids'])))
                    for i in list(categorized_cols['ids'].keys())[:5]:
                        st.caption(f"• {i}")
        
        st.divider()
        
        # DISPLAY: Data preview
        st.subheader("📋 Data Preview (First 10 rows)")
        st.dataframe(data.head(10), use_container_width=True, height=250)
        
        st.divider()
        
        # RUN: Test suites
        st.header("🧪 Automated Test Suites")
        
        p1_p, p1_t = run_tests(data, categorized_cols, PHASE_1_TESTS, "Phase 1: Core Queries")
        st.divider()
        
        p2_p, p2_t = run_tests(data, categorized_cols, PHASE_2_EDGE_CASES, "Phase 2: Edge Cases")
        st.divider()
        
        p3_p, p3_t = run_tests(data, categorized_cols, PHASE_3_COMPLEX, "Phase 3: Complex Queries")
        
        # DISPLAY: Overall results
        st.divider()
        st.header("📈 Overall Accuracy Report")
        
        total_p = p1_p + p2_p + p3_p
        total_t = p1_t + p2_t + p3_t
        overall_acc = (total_p / total_t) * 100 if total_t > 0 else 0
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Tests", total_t)
        with c2:
            st.metric("✅ Passed", total_p)
        with c3:
            st.metric("❌ Failed", total_t - total_p)
        with c4:
            st.metric("Accuracy", f"{overall_acc:.1f}%")
        with c5:
            if overall_acc >= 95:
                st.metric("Status", "🎉 READY", delta="95%+")
            else:
                remaining = 95 - overall_acc
                st.metric("Status", f"⚠️ Away", delta=f"-{remaining:.1f}%")
        
        # MANUAL TESTER
        if show_manual_tester:
            st.divider()
            st.header("🔬 Manual Query Tester")
            st.write("Test any custom question and verify results:")
            
            with st.form("manual_tester"):
                q = st.text_area("Enter Question:", placeholder="e.g., What was the average revenue by country in 2024?", height=80)
                submit = st.form_submit_button("🧪 Test Query", use_container_width=True)
            
            if submit and q:
                params, error = understand_query(q, data, categorized_cols)
                
                if error:
                    st.error(f"❌ Detection Failed: {error}")
                else:
                    st.success("✅ Query Successfully Detected!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📌 Detected Parameters:**")
                        st.json(params)
                    
                    with col2:
                        st.write("**📊 Query Result:**")
                        result, exec_error, validation = execute_query(data, params)
                        if exec_error:
                            st.error(f"Execution Error: {exec_error}")
                        else:
                            st.dataframe(result, use_container_width=True)
                    
                    if validation and not exec_error:
                        st.divider()
                        st.subheader("✅ Detailed Validation & Verification")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**Validation Summary:**")
                            st.json({k: v for k, v in validation.items() if k not in ['result_sample', 'sample_records', 'breakdown_by_dimension', 'breakdown_by_time']})
                        
                        with col2:
                            st.write("**How to Verify in Excel:**")
                            st.info(validation.get('verification_instruction', 'N/A'))
                        
                        # Show detailed data
                        if validation['type'] == 'category_aggregation' and 'result_sample' in validation:
                            st.subheader("Sample Results (Top 3)")
                            st.dataframe(pd.DataFrame(validation['result_sample']))
                        
                        elif validation['type'] == 'time_filter' and validation.get('breakdown_by_dimension'):
                            st.subheader("Breakdown by Dimension")
                            st.json(validation['breakdown_by_dimension'])
                            if validation.get('sample_records'):
                                st.subheader("Sample Records")
                                st.dataframe(pd.DataFrame(validation['sample_records']))
                        
                        elif validation['type'] == 'total_aggregate' and validation.get('breakdown_by_time'):
                            st.subheader("Breakdown by Time Period")
                            if validation['breakdown_by_time']:
                                st.bar_chart(pd.Series(validation['breakdown_by_time']).rename("Total"))
        
        # CLEANUP
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

else:
    st.info("👉 Upload a CSV or Excel file to begin testing")