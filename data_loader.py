import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import openpyxl
import xlrd
from datetime import datetime
import warnings
import re
from decimal import Decimal, InvalidOperation

def parse_currency(value: str) -> Optional[float]:
    """Parse currency strings into float values."""
    if pd.isna(value):
        return None
    
    # Convert to string if not already
    value = str(value).strip()
    
    # Remove currency symbols and other characters
    value = re.sub(r'[^\d.-]', '', value)
    
    try:
        return float(value)
    except ValueError:
        return None

def is_financial_column(column_name: str) -> bool:
    """Check if a column name suggests financial data."""
    financial_keywords = [
        'revenue', 'income', 'expense', 'profit', 'loss', 'earnings',
        'cash', 'debt', 'equity', 'assets', 'liabilities', 'balance',
        'cost', 'price', 'value', 'amount', 'total', 'net', 'gross',
        'margin', 'ratio', 'rate', 'percent', 'percentage', 'growth',
        'return', 'dividend', 'interest', 'tax', 'depreciation',
        'amortization', 'capital', 'investment', 'fund', 'budget',
        'forecast', 'projection', 'estimate', 'actual', 'target'
    ]
    
    column_name = str(column_name).lower()
    return any(keyword in column_name for keyword in financial_keywords)

def clean_financial_value(value: Union[str, float, int]) -> Optional[float]:
    """Clean and convert financial values to float."""
    if pd.isna(value):
        return None
    
    # Convert to string
    value = str(value).strip()
    
    # Handle common financial formats
    if value.startswith('(') and value.endswith(')'):
        value = '-' + value[1:-1]  # Convert (123) to -123
    
    # Handle K/M/B suffixes
    if value.upper().endswith('K'):
        value = str(float(value[:-1]) * 1000)
    elif value.upper().endswith('M'):
        value = str(float(value[:-1]) * 1000000)
    elif value.upper().endswith('B'):
        value = str(float(value[:-1]) * 1000000000)
    
    # Remove currency symbols, commas, and other formatting
    value = re.sub(r'[^\d.-]', '', value)
    
    try:
        return float(value)
    except ValueError:
        return None

def load_file(file_path: str) -> pd.DataFrame:
    """Load a file into a pandas DataFrame with proper cleaning."""
    file_path = Path(file_path)
    
    # Set pandas options
    pd.set_option('future.no_silent_downcasting', True)
    
    try:
        if file_path.suffix.lower() == '.csv':
            # Read CSV with comprehensive options
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                on_bad_lines='warn',  # Warn about problematic lines instead of skipping
                dtype=str,  # Read all columns as string initially
                low_memory=False,  # Prevent mixed type inference
                keep_default_na=True,  # Keep default NA values
                na_values=['', 'NA', 'N/A', 'NULL', 'null', 'None', 'none']  # Additional NA values
            )
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            # Try to read all sheets
            try:
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                sheet_names = excel_file.sheet_names
                
                # Read each sheet and combine
                dfs = []
                for sheet in sheet_names:
                    # Read Excel with comprehensive options
                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet,
                        engine='openpyxl',
                        dtype=str,  # Read all columns as string initially
                        na_filter=True,  # Enable NA filtering
                        keep_default_na=True,  # Keep default NA values
                        na_values=['', 'NA', 'N/A', 'NULL', 'null', 'None', 'none']  # Additional NA values
                    )
                    if not df.empty:
                        # Add sheet name as a column
                        df['sheet_name'] = sheet
                        dfs.append(df)
                
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                else:
                    raise ValueError("No data found in any sheet")
                    
            except Exception as e:
                warnings.warn(f"Failed to read with openpyxl: {str(e)}. Trying with xlrd...")
                # Read with xlrd with comprehensive options
                df = pd.read_excel(
                    file_path,
                    engine='xlrd',
                    dtype=str,  # Read all columns as string initially
                    na_filter=True,  # Enable NA filtering
                    keep_default_na=True,  # Keep default NA values
                    na_values=['', 'NA', 'N/A', 'NULL', 'null', 'None', 'none']  # Additional NA values
                )
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    # Clean the data
    df = clean_dataframe(df)
    
    # Add source file information
    df['source_file'] = str(file_path)
    
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a DataFrame by handling missing values and converting types."""
    # Replace empty strings with NA
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    
    # Drop rows where all values are NA
    df = df.dropna(how='all')
    
    # Handle merged cells by forward filling
    df = df.ffill()
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ['source_file', 'sheet_name']:  # Skip metadata columns
            try:
                # Check if column might contain financial data
                is_financial = is_financial_column(col)
                
                # First check if column contains any non-numeric strings
                has_non_numeric = False
                sample_size = min(1000, len(df))  # Check only a sample for large files
                for val in df[col].dropna().head(sample_size):
                    if isinstance(val, str):
                        # Check for special formats
                        if (val.startswith('FY') and val[2:].isdigit()) or \
                           val.endswith('%') or \
                           val.startswith('$') or \
                           re.match(r'^[+-]?\d*\.?\d+[KMB]?$', val, re.IGNORECASE):
                            continue
                        
                        # Check if it's a regular number
                        if not val.replace('.', '').replace('-', '').replace(',', '').isdigit():
                            has_non_numeric = True
                            break
                
                if not has_non_numeric or is_financial:
                    # Convert the column, handling special cases
                    df[col] = df[col].apply(lambda x: clean_financial_value(x) if pd.notna(x) else x)
                    
                    # If it's a financial column, ensure numeric type
                    if is_financial:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
            except (ValueError, TypeError):
                # Keep original data if conversion fails
                continue
    
    return df

def scan_for_files() -> list:
    """Scan Downloads folder for Excel and CSV files."""
    downloads_path = Path.home() / "Downloads"
    
    # Get all Excel and CSV files
    excel_files = list(downloads_path.glob('*.xlsx')) + list(downloads_path.glob('*.xls'))
    csv_files = list(downloads_path.glob('*.csv'))
    
    # Combine and sort by modification time (newest first)
    all_files = excel_files + csv_files
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return all_files

def load_data(folder_path: str) -> pd.DataFrame:
    """Load all CSV and Excel files from a folder into a single DataFrame."""
    folder_path = Path(folder_path)
    
    # Get all CSV and Excel files
    csv_files = list(folder_path.glob('*.csv'))
    xls_files = list(folder_path.glob('*.xls'))
    xlsx_files = list(folder_path.glob('*.xlsx'))
    
    all_files = csv_files + xls_files + xlsx_files
    
    if not all_files:
        raise ValueError(f"No CSV or Excel files found in {folder_path}")
    
    # Load each file
    dfs = []
    for file in all_files:
        try:
            df = load_file(str(file))
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            warnings.warn(f"Error loading {file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No data could be loaded from any files")
    
    # Combine all DataFrames
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df 