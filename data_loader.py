import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

def load_file(file_path: str) -> pd.DataFrame:
    """Load a single file (CSV or Excel) into a DataFrame."""
    file_path = Path(file_path)
    print(f"\nProcessing file: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.csv':
            print("Reading CSV file...")
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.xlsx':
            print("Reading XLSX file...")
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.suffix.lower() == '.xls':
            print("Reading XLS file...")
            df = pd.read_excel(file_path, engine='xlrd')
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Add source file column
        df['source_file'] = str(file_path)
        
        # Clean data
        df = df.replace(r'^\s*$', pd.NA, regex=True)  # Replace empty strings with NA
        df = df.dropna(how='all')  # Drop rows that are all NA
        
        print(f"Successfully loaded {len(df)} rows")
        return df
    
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def scan_for_files():
    """Scan for Excel and CSV files in Downloads folder."""
    downloads_path = Path.home() / "Downloads"
    print(f"\nScanning Downloads folder: {downloads_path}")
    
    # Get all CSV and Excel files
    csv_files = list(downloads_path.glob('*.csv'))
    xls_files = list(downloads_path.glob('*.xls'))
    xlsx_files = list(downloads_path.glob('*.xlsx'))
    
    all_files = csv_files + xls_files + xlsx_files
    print(f"\nFound {len(all_files)} files:")
    for file in all_files:
        print(f"- {file}")
    
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
        df = load_file(str(file))
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No data could be loaded from any files")
    
    # Combine all DataFrames
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(final_df)}")
    return final_df 