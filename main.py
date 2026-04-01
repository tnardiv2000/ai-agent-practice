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
# TEST SUITES
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
    """Profile all columns - identify their type."""
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
        }
        
        # PRIORITY 1: Exact TIME column names
        if col in ['Date', 'Year', 'Quarter', 'Month', 'Week', 'Day', 'Timestamp']:
            profile['inferred_type'] = 'time'
            column_profiles[col] = profile
            continue
        
        # PRIORITY 2: Dimension columns (by name)
        dimension_keywords = [
            'geo', 'country', 'region', 'location', 'state',
            'product', 'category', 'brand',
            'customer', 'client', 'account',
            'sales_rep', 'rep', 'agent', 'employee', 'staff',
            'team', 'player', 'athlete'
        ]
        
        if any(keyword in col_lower for keyword in dimension_keywords):
            profile['inferred_type'] = 'dimension'
            column_profiles[col] = profile
            continue
        
        # PRIORITY 3: Metric columns (numeric)
        if dtype in ['int64', 'float64']:
            profile['numeric'] = True
            profile['min'] = float(data[col].min()) if not data[col].isnull().all() else None
            profile['max'] = float(data[col].max()) if not data[col].isnull().all() else None
            
            # Percentage metrics
            if '%' in col_lower:
                profile['inferred_type'] = 'metric_percentage'
            # Financial metrics
            elif any(word in col_lower for word in ['spend', 'cost', 'revenue', 'sales', 'profit', 'income', 'amount', 'value']):
                profile['inferred_type'] = 'metric_financial'
            # Count metrics
            elif any(word in col_lower for word in ['count', 'qty', 'quantity', 'units', 'items']):
                profile['inferred_type'] = 'metric_count'
            # Default numeric
            else:
                profile['inferred_type'] = 'metric'
            
            column_profiles[col] = profile
            continue
        
        # PRIORITY 4: Other strings (might be dimensions)
        if dtype == 'object':
            profile['numeric'] = False
            cardinality = profile['unique_count']
            total_rows = len(data)
            cardinality_ratio = cardinality / total_rows if total_rows > 0 else 0
            
            if cardinality_ratio > 0.5:
                profile['inferred_type'] = 'id_or_name'
            else:
                profile['inferred_type'] = 'dimension'
            
            column_profiles[col] = profile
            continue
        
        # Default
        profile['inferred_type'] = 'unknown'
        column_profiles[col] = profile
    
    return column_profiles

def categorize_columns(column_profiles):
    """Group columns by type."""
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
    """Find metric column - handle special cases."""
    question_lower = question.lower()
    
    # Special case: "margin" -> "Profit_Margin_%"
    if 'margin' in question_lower:
        for m in available_metrics:
            if 'Profit_Margin' in m:
                return m
    
    # Special case: "engagement" -> "Employee_Engagement_%"
    if 'engagement' in question_lower:
        for m in available_metrics:
            if 'Engagement' in m:
                return m
    
    # Special case: "return" -> "Return_Rate_%"
    if 'return' in question_lower:
        for m in available_metrics:
            if 'Return_Rate' in m:
                return m
    
    # Special case: "marketing" -> "Marketing_Spend"
    if 'marketing' in question_lower:
        for m in available_metrics:
            if 'Marketing' in m:
                return m
    
    # Special case: "kpi" -> "KPI_%"
    if 'kpi' in question_lower:
        for m in available_metrics:
            if 'KPI' in m:
                return m
    
    # Default: exact substring match
    for m in available_metrics:
        if m.lower() in question_lower:
            return m
    
    # Fallback: word match (sorted by length - longer names first)
    words = re.findall(r'\b\w+\b', question_lower)
    for m in sorted(available_metrics, key=len, reverse=True):
        if m.lower() in words:
            return m
    
    return None

def find_dimension(question, available_dims):
    """Find dimension column - only if dimension keywords present."""
    # Check for dimension keywords
    has_dim_keyword = any(word in question.lower() for word in [
        'which', 'by ', 'per ', 'top ', 'best ', 'worst ', 'highest', 'lowest', 'most'
    ])
    
    if not has_dim_keyword:
        return None
    
    question_lower = question.lower()
    
    # Exact substring match
    for d in available_dims:
        if d.lower() in question_lower:
            return d
    
    # Word match (longest first)
    words = re.findall(r'\b\w+\b', question_lower)
    for d in sorted(available_dims, key=len, reverse=True):
        if d.lower() in words:
            return d
    
    return None

def find_time_column(question, available_time_cols):
    """Find TIME column - EXACT matching for Year/Quarter/Month."""
    if not available_time_cols:
        return None
    
    question_lower = question.lower()
    
    # PRIORITY 1: Quarter
    if re.search(r'\b(q[1-4]|quarter)\b', question_lower):
        if 'Quarter' in available_time_cols:
            return 'Quarter'
    
    # PRIORITY 2: Month
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', question_lower):
        if 'Month' in available_time_cols:
            return 'Month'
    
    # PRIORITY 3: Year
    if re.search(r'\b(19|20)\d{2}\b|year\b', question_lower):
        if 'Year' in available_time_cols:
            return 'Year'
    
    # Default: prefer Year/Quarter/Month over Date
    for col in ['Year', 'Quarter', 'Month']:
        if col in available_time_cols:
            return col
    
    # Last resort: return first time column
    return available_time_cols[0] if available_time_cols else None

def find_time_value(question):
    """Extract time period value (year, quarter, month number)."""
    question_lower = question.lower()
    words = re.findall(r'\b\w+\b', question_lower)
    
    # Quarter: Q1, Q2, Q3, Q4
    for word in words:
        if word in ['q1', 'q2', 'q3', 'q4']:
            return word[1]  # Return: 1, 2, 3, 4
    
    # Month names
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7,
        'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for word in words:
        if word in months:
            return str(months[word])
    
    # Year: 2023, 2024, etc
    for word in words:
        if re.match(r'^(19|20)\d{2}$', word):
            return word
    
    return None

def detect_aggregation(question, metric_col):
    """Detect aggregation function (min, max, mean, sum)."""
    question_lower = question.lower()
    
    if any(w in question_lower for w in ['lowest', 'minimum', 'min']):
        return 'min'
    if any(w in question_lower for w in ['highest', 'maximum', 'max', 'top', 'best']):
        return 'max'
    if any(w in question_lower for w in ['average', 'avg', 'mean']):
        return 'mean'
    if any(w in question_lower for w in ['total', 'sum']):
        return 'sum'
    
    return 'max'  # Default

def understand_query(question, data, categorized_cols):
    """Main query understanding logic."""
    
    # Step 1: Find metric
    metric = find_metric(question, list(categorized_cols['metrics'].keys()))
    if not metric:
        return None, f"❌ Could not find metric. Available: {', '.join(categorized_cols['metrics'].keys())}"
    
    # Step 2: Find dimension
    dimension = find_dimension(question, list(categorized_cols['dimensions'].keys()))
    
    # Step 3: Find time column and value
    time_col = find_time_column(question, list(categorized_cols['time'].keys()))
    time_val = find_time_value(question)
    
    # Step 4: Determine query type
    question_lower = question.lower()
    
    # Type A: Group by dimension
    if dimension and any(w in question_lower for w in ['which', 'by ', 'top ', 'best ', 'worst ', 'highest', 'lowest']):
        agg = detect_aggregation(question, metric)
        return {
            "query_type": "best_worst_by_category",
            "category_column": dimension,
            "value_column": metric,
            "aggregation_function": agg,
        }, None
    
    # Type B: Filter by time
    elif time_col and time_val:
        return {
            "query_type": "filter_by_time",
            "time_column": time_col,
            "value_column": metric,
            "filter_value": time_val,
        }, None
    
    # Type C: Total
    else:
        return {
            "query_type": "total_by_time",
            "time_column": time_col,
            "value_column": metric,
        }, None

# ============================================================================
# QUERY EXECUTION
# ============================================================================
def execute_query(data, params):
    """Execute the detected query and return results + validation."""
    try:
        query_type = params.get("query_type")
        value_col = params.get("value_column")
        
        if not value_col or value_col not in data.columns:
            return None, f"Column '{value_col}' not found", None
        
        if query_type == "best_worst_by_category":
            category_col = params.get("category_column")
            agg = params.get("aggregation_function", "max")
            
            if not category_col or category_col not in data.columns:
                return None, f"Category column '{category_col}' not found", None
            
            # Group and aggregate
            if agg == 'mean':
                result = data.groupby(category_col)[value_col].mean().reset_index()
            elif agg == 'min':
                result = data.groupby(category_col)[value_col].min().reset_index()
            elif agg == 'max':
                result = data.groupby(category_col)[value_col].max().reset_index()
            else:
                result = data.groupby(category_col)[value_col].sum().reset_index()
            
            result = result.sort_values(value_col, ascending=(agg == 'min'))
            result.columns = [category_col, f"{agg.upper()} {value_col}"]
            
            top_category = result.iloc[0][category_col]
            top_value = result.iloc[0][result.columns[1]]
            
            validation = {
                "type": "category",
                "category_col": category_col,
                "value_col": value_col,
                "agg": agg,
                "top_category": top_category,
                "top_value": round(top_value, 2) if isinstance(top_value, (int, float)) else top_value,
                "total_groups": len(result),
            }
            
            return result, None, validation
        
        elif query_type == "filter_by_time":
            time_col = params.get("time_column")
            filter_val = params.get("filter_value")
            
            if not time_col or time_col not in data.columns:
                return None, f"Time column '{time_col}' not found", None
            
            # Filter data
            filtered = data[data[time_col].astype(str) == str(filter_val)]
            
            if len(filtered) == 0:
                available = sorted(data[time_col].dropna().unique())
                return None, f"No data for {time_col}={filter_val}. Available: {available}", None
            
            # Calculate total
            total = filtered[value_col].sum()
            result = pd.DataFrame({f"Total {value_col}": [round(total, 2)]})
            
            validation = {
                "type": "time_filter",
                "time_col": time_col,
                "filter_val": filter_val,
                "value_col": value_col,
                "total": round(total, 2),
                "rows_matched": len(filtered),
                "rows_total": len(data),
            }
            
            return result, None, validation
        
        else:  # total
            total = data[value_col].sum()
            result = pd.DataFrame({f"Total {value_col}": [round(total, 2)]})
            
            validation = {
                "type": "total",
                "value_col": value_col,
                "total": round(total, 2),
                "rows": len(data),
            }
            
            return result, None, validation
    
    except Exception as e:
        return None, str(e), None

# ============================================================================
# TEST RUNNER
# ============================================================================
def run_tests(data, categorized_cols, test_suite, phase_name):
    """Run test suite."""
    results = []
    passed = 0
    
    for question, expected_type, expected_metric, expected_dim, expected_time in test_suite:
        params, error = understand_query(question, data, categorized_cols)
        
        if error:
            results.append({
                "Q": question[:50],
                "Status": "❌",
                "Expected": expected_type,
                "Got": "ERROR",
                "Issue": error[:40]
            })
            continue
        
        # Check if matches
        type_ok = params.get("query_type") == expected_type
        metric_ok = params.get("value_column") == expected_metric
        dim_ok = (params.get("category_column") == expected_dim) if expected_dim else True
        
        # Execute to verify
        result, exec_error, validation = execute_query(data, params)
        exec_ok = exec_error is None and result is not None
        
        if type_ok and metric_ok and dim_ok and exec_ok:
            results.append({
                "Q": question[:50],
                "Status": "✅",
                "Expected": expected_type,
                "Got": params.get("query_type"),
                "Issue": "OK"
            })
            passed += 1
        else:
            issues = []
            if not type_ok:
                issues.append(f"Type")
            if not metric_ok:
                issues.append(f"Metric")
            if not dim_ok:
                issues.append(f"Dim")
            if not exec_ok:
                issues.append(f"Exec")
            
            results.append({
                "Q": question[:50],
                "Status": "❌",
                "Expected": expected_type,
                "Got": params.get("query_type") if params else "ERROR",
                "Issue": " | ".join(issues)
            })
    
    # Display
    st.subheader(f"📊 {phase_name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(test_suite))
    with col2:
        st.metric("✅", passed)
    with col3:
        st.metric("❌", len(test_suite) - passed)
    with col4:
        acc = (passed / len(test_suite)) * 100 if test_suite else 0
        st.metric("Accuracy", f"{acc:.1f}%")
    
    with st.expander("Details"):
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    return passed, len(test_suite)

# ============================================================================
# MAIN APP
# ============================================================================
st.sidebar.header("Settings")
show_detected = st.sidebar.checkbox("Show Detected Columns", True)

uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])

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
        categorized_cols = categorize_columns(column_profiles)
        
        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
        
        if show_detected:
            with st.expander("🔍 Column Detection"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("**Metrics:**")
                    for m in list(categorized_cols['metrics'].keys())[:10]:
                        st.write(f"- {m}")
                with col2:
                    st.write("**Dimensions:**")
                    for d in list(categorized_cols['dimensions'].keys())[:10]:
                        st.write(f"- {d}")
                with col3:
                    st.write("**Time:**")
                    for t in categorized_cols['time'].keys():
                        st.write(f"- {t}")
                with col4:
                    st.write("**Sample Data:**")
                    if categorized_cols['time']:
                        tc = list(categorized_cols['time'].keys())[0]
                        for v in data[tc].unique()[:5]:
                            st.write(f"- {v}")
        
        st.divider()
        
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(10), use_container_width=True, height=250)
        
        st.divider()
        st.header("🧪 Tests")
        
        p1_p, p1_t = run_tests(data, categorized_cols, PHASE_1_TESTS, "Phase 1")
        st.divider()
        
        p2_p, p2_t = run_tests(data, categorized_cols, PHASE_2_EDGE_CASES, "Phase 2")
        st.divider()
        
        p3_p, p3_t = run_tests(data, categorized_cols, PHASE_3_COMPLEX, "Phase 3")
        
        st.divider()
        st.header("📈 Overall")
        
        total_p = p1_p + p2_p + p3_p
        total_t = p1_t + p2_t + p3_t
        overall_acc = (total_p / total_t) * 100 if total_t > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total", total_t)
        with col2:
            st.metric("Passed", total_p)
        with col3:
            st.metric("Failed", total_t - total_p)
        with col4:
            st.metric("Accuracy", f"{overall_acc:.1f}%")
        with col5:
            if overall_acc >= 95:
                st.metric("Status", "🎉 READY")
            else:
                st.metric("Status", f"⚠️ {95 - overall_acc:.1f}% away")
        
        st.divider()
        st.header("🔬 Manual Tester")
        
        with st.form("tester"):
            q = st.text_area("Question:", placeholder="e.g., What was the total revenue in 2024?")
            submit = st.form_submit_button("Test", use_container_width=True)
        
        if submit and q:
            params, error = understand_query(q, data, categorized_cols)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ Detected!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Query Parameters:**")
                    st.json(params)
                
                with col2:
                    st.write("**Result:**")
                    result, exec_error, validation = execute_query(data, params)
                    if exec_error:
                        st.error(f"❌ {exec_error}")
                    else:
                        st.dataframe(result, use_container_width=True)
                
                if validation:
                    st.divider()
                    st.subheader("✅ Validation")
                    
                    if validation['type'] == 'category':
                        st.write(f"**Top Result:** {validation['top_category']}: **{validation['top_value']}**")
                        st.write(f"**Total Groups:** {validation['total_groups']}")
                    elif validation['type'] == 'time_filter':
                        st.write(f"**Filter:** {validation['time_col']} = {validation['filter_val']}")
                        st.write(f"**Result:** {validation['total']} ({validation['rows_matched']}/{validation['rows_total']} rows)")
                    else:
                        st.write(f"**Total:** {validation['total']}")
        
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👉 Upload a file to begin")