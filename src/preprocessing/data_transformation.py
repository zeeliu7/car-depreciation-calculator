# Original data from Kaggle: https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset

import pandas as pd
import re
import os

def transform_csv(input_file_path):
    df = pd.read_csv(input_file_path)
    
    # Make a copy to avoid modifying the original data
    transformed_df = df.copy()
    
    # Merge "brand", "model", "model_year" into "vehicle"
    transformed_df['vehicle'] = (
        transformed_df['brand'].astype(str) + ' ' + 
        transformed_df['model'].astype(str) + ' ' + 
        transformed_df['model_year'].astype(str)
    )
    
    # Create "age" field (2025 - model_year)
    transformed_df['age'] = 2025 - transformed_df['model_year']
    
    # Convert "milage" to int using regex
    def clean_milage(milage):
        if pd.isna(milage):
            return 0
        numbers_only = re.sub(r'[^\d]', '', str(milage))
        return int(numbers_only) if numbers_only else 0
    
    transformed_df['milage'] = transformed_df['milage'].apply(clean_milage)
    
    # Process "fuel_type"
    valid_fuel_types = ["Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel"]
    
    def process_fuel_type(row):
        fuel_type = str(row['fuel_type']).strip()
        engine = str(row.get('engine', '')).lower()
        
        # Check case-insensitive match with valid types
        for valid_type in valid_fuel_types:
            if fuel_type.lower() == valid_type.lower():
                return valid_type
        
        if 'gas/electric' in engine:
            return "Hybrid"
        elif 'electric' in engine:
            return "Electric"
        else:
            return "Others"
    
    transformed_df['fuel_type'] = transformed_df.apply(process_fuel_type, axis=1)

    # Process "transmission"
    def process_transmission(transmission):
        if pd.isna(transmission):
            return "Others"
            
        transmission = str(transmission).strip().lower()
        if 'a/t' in transmission or 'automatic' in transmission:
            return "Automatic"
        elif 'm/t' in transmission or 'manual' in transmission:
            return 'Manual'
        elif 'cvt' in transmission:
            return "CVT"
        elif 'dual shift' in transmission:
            return "Dual Shift"
        elif 'overdrive switch' in transmission:
            return "Overdrive Switch"
        else:
            return "Others"
    
    transformed_df['transmission'] = transformed_df['transmission'].apply(process_transmission)
    
    # Process "accident"
    def process_accident(accident):
        accident = str(accident).strip()
        if accident == "None reported":
            return "No"
        elif accident == "At least 1 accident or damage reported":
            return "Yes"
        else:
            return "Unknown"
    
    transformed_df['accident'] = transformed_df['accident'].apply(process_accident)
    
    # Process "clean_title"
    def process_clean_title(title):
        if pd.isna(title):
            return "No"
        title = str(title).strip()
        if title == "Yes":
            return "Yes"
        else:
            return "No"
    
    transformed_df['clean_title'] = transformed_df['clean_title'].apply(process_clean_title)
    
    # Convert "price" to int using regex
    def clean_price(price):
        if pd.isna(price):
            return 0
        numbers_only = re.sub(r'[^\d]', '', str(price))
        return int(numbers_only) if numbers_only else 0
    
    transformed_df['price'] = transformed_df['price'].apply(clean_price)

    # Data validation - remove invalid rows
    print(f"Before data validation: {len(transformed_df)} rows")
    
    def is_valid_row(row):
        # Price must be greater than 0
        if row['price'] <= 0:
            return False
            
        # Model year should be reasonable (1900-2025)
        if row['model_year'] < 1900 or row['model_year'] > 2025:
            return False
            
        # Age should be non-negative
        if row['age'] < 0:
            return False
            
        # milage should be reasonable (0 to 1,000,000 miles)
        if row['milage'] < 0 or row['milage'] > 1000000:
            return False
            
        # Vehicle name shouldn't contain 'nan' or be too short
        vehicle_str = str(row['vehicle']).lower()
        if 'nan' in vehicle_str or len(vehicle_str.strip()) < 5:
            return False
            
        return True
    
    # Apply validation and keep only valid rows
    valid_mask = transformed_df.apply(is_valid_row, axis=1)
    invalid_count = sum(~valid_mask)
    transformed_df = transformed_df[valid_mask]
    
    print(f"After data validation: {len(transformed_df)} rows")
    print(f"Removed {invalid_count} invalid rows")

    # Drop unnecessary columns
    columns_to_drop = ['brand', 'model', 'model_year', 'engine', 'ext_col', 'int_col']
    # Only drop columns that exist to avoid KeyError
    existing_columns_to_drop = [col for col in columns_to_drop if col in transformed_df.columns]
    transformed_df = transformed_df.drop(existing_columns_to_drop, axis=1)
    
    # Save the transformed CSV
    input_dir = os.path.dirname(input_file_path)
    input_filename = os.path.basename(input_file_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_transformed{ext}"
    output_path = os.path.join(input_dir, output_filename)
    transformed_df.to_csv(output_path, index=False)
    
    print(f"\nTransformation completed!")
    print(f"Original file: {input_file_path}")
    print(f"Transformed file saved as: {output_path}")
    print(f"Original rows: {len(df)}")
    print(f"Final rows: {len(transformed_df)}")
    
    return output_path, transformed_df

if __name__ == "__main__":
    input_file = "used_cars.csv"
    
    try:
        output_file, df_transformed = transform_csv(input_file)
        
        print("\nFirst 5 rows of transformed data:")
        print(df_transformed.head())
        
        print(f"\nColumns in transformed file: {list(df_transformed.columns)}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")