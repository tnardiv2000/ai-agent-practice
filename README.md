# 📊 AI Data Analyzer Pro

A smart data analysis application that combines **manual data calculations** with **AI-powered insights** using Streamlit and Ollama.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [How to Use](#how-to-use)
7. [Code Walkthrough](#code-walkthrough)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)

---

## **Overview**

AI Data Analyzer Pro is a Streamlit web application that allows you to:
- Upload Excel or CSV files
- Perform smart data aggregations with intelligent function selection
- Get verified, accurate results matching your pivot tables
- Ask AI to analyze patterns and provide insights

**Key Innovation:** Results are verified through manual calculations BEFORE AI analysis, ensuring accuracy and preventing hallucination.

---

## **Features**

### **📊 Data Preview & Statistics**
- View raw data with customizable row display
- Quick statistics dashboard (row count, column count, missing values, file size)
- Detailed column information including data types and unique value counts

### **🧮 Manual Data Calculations**
All calculations happen first, verified data goes to AI:

1. **Column Statistics** - Detailed stats (min, max, mean, median, std dev) for any column
2. **Filter & View** - Filter data by column values and view complete rows
3. **Group & Aggregate** - Smart aggregation with intelligent function selection:
   - Financial columns (Spend, Savings, etc.) → Default: SUM, Options: MAX, MIN, MEAN
   - Percentage columns (KPI_%, Profit_Margin_%, etc.) → Default: MEAN, Options: MAX, MIN
   - Count columns (Units_Sold) → Default: SUM, Options: MEAN, MAX, MIN
   - Different functions for different columns in same query
4. **Custom View** - Select specific columns and apply optional filters

### **🤖 AI-Powered Analysis**
- Ask questions about your data
- AI references verified aggregation results
- Temperature control for precision vs. creativity
- Timeout settings for long-running queries
- Show/hide analysis steps for transparency

---

## **Requirements**

### **Software**
- **Python 3.8+**
- **Ollama** (for local AI/LLM)
- **Git** (optional, for version control)

### **Python Libraries**