import json
import pandas as pd

# Sample JSON data (replace with your json loading mechanism or file path)


def extract_holding_cost_fva(data):
    """
    Extracts only monthly holding cost entries from the FVA list.
    
    Parameters:
        data (list or dict): The raw experiment output.
        
    Returns:
        pd.DataFrame: A DataFrame containing filtered holding cost FVA entries.
    """
    # Ensure data structure handling if passed as a string or list
    if isinstance(data, str):
        data = json.loads(data)
        
    holding_cost_records = []
    
    for experiment in data:
        fva_records = experiment.get("fva", [])
        for record in fva_records:
            # Filter specifically for metrics starting with 'monthly_holding_cost'
            if record.get("metric", "").startswith("monthly_holding_cost"):
                holding_cost_records.append(record)
                
    return pd.DataFrame(holding_cost_records)



# Execute extraction
df_holding_cost_fva = extract_holding_cost_fva(json_data)

# Print tabular results
print(df_holding_cost_fva.to_string(index=False))