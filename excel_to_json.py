import pandas as pd
import json
import os

def excel_to_json_flexible(excel_file_path, output_json_path, sheet_name=None):
    """
    Convert Excel file to JSON with flexible sheet selection
    """

    # If no sheet name specified, just pick the first sheet
    xl_file = pd.ExcelFile(excel_file_path)
    if sheet_name is None:
        sheet_name = xl_file.sheet_names[1]

    print(f"\nReading sheet: {sheet_name}")

    # Read Excel into DataFrame
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    # Find the first row where data starts (numeric video ID)
    data_start_row = 0
    for i in range(min(10, len(df))):
        try:
            float(df.iloc[i, 0])  # check if first col is numeric
            data_start_row = i
            break
        except:
            continue

    if data_start_row > 0:
        print(f"Data starts at row {data_start_row}, adjusting...")
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=data_start_row)

    print(f"Processing {len(df)} rows with {len(df.columns)} columns")

    json_data = []

    for idx, row in df.iterrows():
        try:
            # Helpers
            def safe_int(val, default=0):
                try:
                    if pd.notna(val):
                        return int(float(val))
                    return default
                except:
                    return default

            def safe_str(val, default=""):
                try:
                    if pd.notna(val):
                        return str(val)
                    return default
                except:
                    return default

            # Video ID (keep folder name, not .jpg)
            video_val = row.iloc[0]
            try:
                video_num = int(float(video_val))
            except:
                continue

            if video_num == 0:
                continue

            entry = {
                "Video": f"{str(video_num).zfill(6)}",   # <<<< folder name only
                "weather(sunny,rainy,snowy,foggy)1-4": safe_int(row.iloc[1]),
                "light(day,night)1-2": safe_int(row.iloc[2]),
                "scenes(highway,tunnel,mountain,urban,rural)1-5": safe_int(row.iloc[3]),
                "linear(arterials,curve,intersection,T-junction,ramp) 1-5": safe_int(row.iloc[4]),
                "type": safe_int(row.iloc[5]),
                "whether an accident occurred (1/0)": safe_int(row.iloc[6]),
                "abnormal start frame": safe_int(row.iloc[7]),
                "abnormal end frame": safe_int(row.iloc[8]),
                "accident frame": safe_int(row.iloc[9]),
                "total frames": safe_int(row.iloc[10]),
                "[0,tai]": safe_int(row.iloc[11]),
                "[tai,tco]": safe_int(row.iloc[12]),
                "[tai,tae]": safe_int(row.iloc[13]),
                "[tco,tae]": safe_int(row.iloc[14]),
                "[tae,end]": safe_int(row.iloc[15]),
                "texts": safe_str(row.iloc[16]),
                "causes": safe_str(row.iloc[17]),
                "measures": safe_str(row.iloc[18])
            }

            json_data.append(entry)
            print(f"Processed video {video_num}")

        except Exception as e:
            print(f"Warning: Could not process row {idx}: {e}")
            continue

    if json_data:
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        print(f"\n✓ Successfully converted {len(json_data)} entries to JSON format")
        print(f"✓ Output saved to: {output_json_path}")
        print("\nSample entry:")
        print(json.dumps(json_data[0], indent=2, ensure_ascii=False))
    else:
        print("\n✗ No valid data entries found to convert")

    return json_data


# Example run
if __name__ == "__main__":
    excel_file = "cap_text_annotations.xls"
    json_output = "cap_annotations.json"

    result = excel_to_json_flexible(excel_file, json_output)
