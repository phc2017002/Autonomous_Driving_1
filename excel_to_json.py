import pandas as pd
import json
import os

def diagnose_excel_file(excel_file_path):
    """
    Thoroughly diagnose the Excel file structure
    """
    print("=" * 80)
    print(f"DIAGNOSING EXCEL FILE: {excel_file_path}")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(excel_file_path):
        print(f"ERROR: File '{excel_file_path}' does not exist!")
        return None
    
    # Try to read all sheets
    try:
        xl_file = pd.ExcelFile(excel_file_path)
        print(f"\nNumber of sheets found: {len(xl_file.sheet_names)}")
        print(f"Sheet names: {xl_file.sheet_names}")
        print()
        
        # Read each sheet
        for sheet_name in xl_file.sheet_names:
            print(f"\n{'='*60}")
            print(f"SHEET: '{sheet_name}'")
            print('='*60)
            
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"\nColumn names:")
            for i, col in enumerate(df.columns):
                print(f"  Column {i}: '{col}'")
            
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            print(f"\nData types:")
            print(df.dtypes)
            
            # Check if this might be the data sheet we're looking for
            if df.shape[1] >= 19:  # Expected to have at least 19 columns
                print("\n✓ This sheet might contain the expected data structure!")
                return sheet_name
            
        return xl_file.sheet_names[0] if xl_file.sheet_names else None
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def find_data_sheet(excel_file_path):
    """
    Find the sheet that contains the actual data (not text descriptions)
    """
    xl_file = pd.ExcelFile(excel_file_path)
    
    for sheet_name in xl_file.sheet_names:
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        
        # Check if first column contains numbers (video IDs)
        if len(df) > 0:
            first_col = df.iloc[:, 0]
            # Try to convert first few values to numbers
            try:
                numeric_count = 0
                for val in first_col.head(10):
                    try:
                        float(val)
                        numeric_count += 1
                    except:
                        pass
                
                # If most values are numeric, this is likely our data sheet
                if numeric_count > 5:
                    print(f"Found data sheet: '{sheet_name}'")
                    return sheet_name
            except:
                continue
    
    return None

def excel_to_json_flexible(excel_file_path, output_json_path, sheet_name=None):
    """
    Convert Excel file to JSON with flexible sheet selection
    """
    
    # If no sheet name specified, try to find the data sheet
    if sheet_name is None:
        sheet_name = find_data_sheet(excel_file_path)
        if sheet_name is None:
            # Try first sheet or default
            sheet_name = 0
    
    print(f"\nReading sheet: {sheet_name}")
    
    # Read the Excel file
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Check if we need to skip header rows
    # Sometimes the actual data starts after a few rows
    data_start_row = 0
    for i in range(min(10, len(df))):
        try:
            # Check if this row has a numeric first value
            float(df.iloc[i, 0])
            data_start_row = i
            break
        except:
            continue
    
    if data_start_row > 0:
        print(f"Data starts at row {data_start_row}, adjusting...")
        # Re-read with proper header
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=data_start_row)
    
    print(f"Processing {len(df)} rows with {len(df.columns)} columns")
    
    # Create a list to store all entries
    json_data = []
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Helper function to safely get numeric value
            def safe_int(val, default=0):
                try:
                    if pd.notna(val):
                        return int(float(val))
                    return default
                except:
                    return default
            
            # Helper function to safely get string value
            def safe_str(val, default=""):
                try:
                    if pd.notna(val):
                        return str(val)
                    return default
                except:
                    return default
            
            # Try to extract video number from first column
            video_val = row.iloc[0] if len(row) > 0 else 0
            video_num = safe_int(video_val, 0)
            
            # Skip rows that don't have a valid video number
            if video_num == 0:
                continue
            
            # Create entry based on column position
            entry = {
                "Video": f"{str(video_num).zfill(6)}.jpg",
                "weather(sunny,rainy,snowy,foggy)1-4": safe_int(row.iloc[1] if len(row) > 1 else 0),
                "light(day,night)1-2": safe_int(row.iloc[2] if len(row) > 2 else 0),
                "scenes(highway,tunnel,mountain,urban,rural)1-5": safe_int(row.iloc[3] if len(row) > 3 else 0),
                "linear(arterials,curve,intersection,T-junction,ramp) 1-5": safe_int(row.iloc[4] if len(row) > 4 else 0),
                "type": safe_int(row.iloc[5] if len(row) > 5 else 0),
                "whether an accident occurred (1/0)": safe_int(row.iloc[6] if len(row) > 6 else 0),
                "abnormal start frame": safe_int(row.iloc[7] if len(row) > 7 else 0),
                "abnormal end frame": safe_int(row.iloc[8] if len(row) > 8 else 0),
                "accident frame": safe_int(row.iloc[9] if len(row) > 9 else 0),
                "total frames": safe_int(row.iloc[10] if len(row) > 10 else 0),
                "[0,tai]": safe_int(row.iloc[11] if len(row) > 11 else 0),
                "[tai,tco]": safe_int(row.iloc[12] if len(row) > 12 else 0),
                "[tai,tae]": safe_int(row.iloc[13] if len(row) > 13 else 0),
                "[tco,tae]": safe_int(row.iloc[14] if len(row) > 14 else 0),
                "[tae,end]": safe_int(row.iloc[15] if len(row) > 15 else 0),
                "texts": safe_str(row.iloc[16] if len(row) > 16 else ""),
                "causes": safe_str(row.iloc[17] if len(row) > 17 else ""),
                "measures": safe_str(row.iloc[18] if len(row) > 18 else "")
            }
            
            json_data.append(entry)
            print(f"Processed video {video_num}")
            
        except Exception as e:
            print(f"Warning: Could not process row {idx}: {e}")
            continue
    
    if json_data:
        # Write to JSON file
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        
        print(f"\n✓ Successfully converted {len(json_data)} entries to JSON format")
        print(f"✓ Output saved to: {output_json_path}")
        
        # Show sample
        print("\nSample entry:")
        print(json.dumps(json_data[0], indent=2, ensure_ascii=False))
    else:
        print("\n✗ No valid data entries found to convert")
    
    return json_data

# Main execution
if __name__ == "__main__":
    # Your file paths
    excel_file = "cap_text_annotations.xls"  # Your Excel file
    json_output = "cap_annotations.json"  # Output JSON file
    
    # Step 1: Diagnose the Excel file
    print("STEP 1: Diagnosing Excel file structure...")
    suggested_sheet = diagnose_excel_file(excel_file)
    
    print("\n" + "="*80)
    print("STEP 2: Converting to JSON...")
    print("="*80)
    
    # Step 2: Convert to JSON
    # If you know the correct sheet name, specify it here
    # For example: sheet_name="Sheet1" or sheet_name=0 for first sheet
    result = excel_to_json_flexible(excel_file, json_output, sheet_name=suggested_sheet)
    
    if not result:
        print("\n" + "="*80)
        print("TROUBLESHOOTING TIPS:")
        print("="*80)
        print("1. Make sure your Excel file contains the actual data, not just text descriptions")
        print("2. Check if the data is in a different sheet")
        print("3. Verify the Excel file has columns in this order:")
        print("   video | weather | light | scenes | linear | type | accident | frames...")
        print("4. If your data is in a specific sheet, modify the code to use:")
        print("   result = excel_to_json_flexible(excel_file, json_output, sheet_name='YourSheetName')")