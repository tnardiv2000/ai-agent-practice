import re

# Check main.py has all critical functions
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

required_functions = [
    'def get_column_type',
    'def find_correct_column',
    'def find_correct_filter_column',
    'def normalize_filter_value',
    'def detect_time_period_value',
    'def detect_aggregation_function',
    'def ai_understand_query',
    'def execute_query',
    'def get_column_statistics',
]

required_constants = [
    'FINANCIAL_COLUMNS',
    'PERCENTAGE_COLUMNS',
    'COUNT_COLUMNS',
    'DATE_COLUMNS',
    'DIMENSION_COLUMNS',
]

print("✅ Checking critical functions...")
missing = []
for func in required_functions:
    if func in content:
        print(f"  ✓ {func}")
    else:
        print(f"  ✗ MISSING: {func}")
        missing.append(func)

print("\n✅ Checking constants...")
for const in required_constants:
    if const in content:
        print(f"  ✓ {const}")
    else:
        print(f"  ✗ MISSING: {const}")
        missing.append(const)

print(f"\n📊 Total lines: {len(content.split(chr(10)))}")

if missing:
    print(f"\n❌ MISSING ITEMS: {missing}")
else:
    print("\n✅ ALL CRITICAL COMPONENTS PRESENT!")