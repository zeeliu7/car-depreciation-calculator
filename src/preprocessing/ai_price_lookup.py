import csv
import pandas as pd
from openai import AsyncOpenAI
import asyncio
import os
import logging
import numpy as np
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel
import random
import json

# Global variable for current year
CURRENT_YEAR = 2025

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class VehiclePrice(BaseModel):
    vehicle: str
    price: str  # String format (integers as strings, or "N/A")

class VehiclePriceResponse(BaseModel):
    prices: List[VehiclePrice]

class SimpleVehiclePriceLookup:
    def __init__(self, api_key: str, csv_file_path: str = 'used_cars_transformed.csv'):
        """Initialize with hardcoded settings"""
        self.client = AsyncOpenAI(api_key=api_key)
        self.csv_file_path = csv_file_path
        
        # Hardcoded configuration
        self.num_workers = 3
        self.batch_size = 15
        self.rate_limit_delay = 5.0
        self.max_retries = 3
        
        # Control concurrent API calls
        self.semaphore = asyncio.Semaphore(self.num_workers)
        
        # Load CPI data
        self.cpi_data = self.load_cpi_data()
    
    def load_cpi_data(self) -> Dict[str, float]:
        """Load CPI data from JSON file"""
        cpi_file_path = 'cpi.json'
        try:
            with open(cpi_file_path, 'r') as f:
                cpi_data = json.load(f)
            logger.info(f"Loaded CPI data for {len(cpi_data)} years from {cpi_file_path}")
            
            # Validate that current year exists in CPI data
            current_year_str = str(CURRENT_YEAR)
            if current_year_str not in cpi_data:
                logger.error(f"CPI data for current year {CURRENT_YEAR} not found in {cpi_file_path}")
                raise ValueError(f"Missing CPI data for current year {CURRENT_YEAR}")
            
            return cpi_data
        except FileNotFoundError:
            logger.error(f"CPI data file not found: {cpi_file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing CPI JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading CPI data: {e}")
            raise
    
    def read_csv(self) -> pd.DataFrame:
        """Read the CSV file"""
        try:
            df = pd.read_csv(self.csv_file_path)
            logger.info(f"Read {len(df)} vehicles from {self.csv_file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
    
    def create_batches(self, vehicles: List[str]) -> List[List[str]]:
        """Split vehicles into batches"""
        batches = []
        for i in range(0, len(vehicles), self.batch_size):
            batches.append(vehicles[i:i + self.batch_size])
        
        logger.info(f"Created {len(batches)} batches of up to {self.batch_size} vehicles each")
        return batches
    
    async def lookup_batch_prices_with_retry(self, batch: List[str], batch_id: int) -> Dict[str, str]:
        """Look up prices for a batch with retry logic"""
        for attempt in range(1, self.max_retries + 1):
            try:
                result = await self._lookup_batch_prices_single_attempt(batch, batch_id, attempt)
                return result
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"Batch {batch_id}: All {self.max_retries} attempts failed. Final error: {e}")
                    # Return N/A for all vehicles in permanently failed batch
                    return {vehicle: "N/A" for vehicle in batch}
                else:
                    # Calculate exponential backoff delay
                    delay = self.rate_limit_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.warning(f"Batch {batch_id}, attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        return {vehicle: "N/A" for vehicle in batch}
    
    async def _lookup_batch_prices_single_attempt(self, batch: List[str], batch_id: int, attempt: int) -> Dict[str, str]:
        """Single attempt to look up prices for a batch"""
        async with self.semaphore:
            try:
                await asyncio.sleep(self.rate_limit_delay * 0.1)
                
                vehicle_list = '\n'.join(batch)
                
                prompt = f"""
                Look up the original manufacturer's suggested retail price (MSRP) in USD for these vehicles when first released.
                I need the base model price without options - just the original selling price.
                
                Use web search to find accurate pricing from reliable automotive sources.
                
                Vehicles:
                {vehicle_list}
                
                For each vehicle, provide:
                - The exact vehicle name as given
                - The original MSRP price as an integer string (no decimals, dollar signs, or commas)
                - If you cannot find the price, use "N/A"
                
                Example: For "Toyota Corolla LE 2020", if MSRP was $24,000, return "24000"
                """
                
                response = await self.client.responses.parse(
                    model="gpt-4o-mini",
                    tools=[{"type": "web_search_preview"}],
                    input=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that looks up vehicle pricing. Use web search for accurate MSRP prices from reliable sources. Provide structured responses."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    text_format=VehiclePriceResponse
                )
                
                # Validate the response
                if not response or not hasattr(response, 'output_parsed') or not response.output_parsed:
                    raise ValueError("Invalid or empty response from OpenAI API")
                
                # Validate the parsed output structure
                parsed_output = response.output_parsed
                if not hasattr(parsed_output, 'prices') or not parsed_output.prices:
                    raise ValueError("Response missing 'prices' field or prices list is empty")
                
                # Convert to dictionary with validation
                prices = {}
                for vehicle_price in parsed_output.prices:
                    if not hasattr(vehicle_price, 'vehicle') or not hasattr(vehicle_price, 'price'):
                        logger.warning(f"Batch {batch_id}: Skipping malformed price entry")
                        continue
                    
                    vehicle_name = vehicle_price.vehicle.strip()
                    price_value = vehicle_price.price.strip()
                    
                    # Validate vehicle name is not empty
                    if not vehicle_name:
                        logger.warning(f"Batch {batch_id}: Skipping entry with empty vehicle name")
                        continue
                    
                    # Validate price format (should be integer string or "N/A")
                    if price_value != "N/A":
                        try:
                            # Check if it's a valid integer string
                            int(price_value)
                        except ValueError:
                            logger.warning(f"Batch {batch_id}: Invalid price format '{price_value}' for '{vehicle_name}', setting to N/A")
                            price_value = "N/A"
                    
                    prices[vehicle_name] = price_value
                
                # Ensure all vehicles in the batch have entries
                missing_vehicles = []
                for vehicle in batch:
                    if vehicle not in prices:
                        prices[vehicle] = "N/A"
                        missing_vehicles.append(vehicle)
                
                if missing_vehicles:
                    logger.warning(f"Batch {batch_id}: Missing prices for {len(missing_vehicles)} vehicles, set to N/A")
                
                # Validate we have the right number of results
                if len(prices) != len(batch):
                    logger.warning(f"Batch {batch_id}: Expected {len(batch)} results, got {len(prices)}")
                
                found_prices = sum(1 for p in prices.values() if p != 'N/A')
                logger.info(f"Batch {batch_id} (attempt {attempt}): {len(prices)} vehicles, {found_prices} prices found")
                
                return prices
                
            except asyncio.TimeoutError:
                raise Exception("Request timed out")
            except Exception as e:
                # Log the specific error type for better debugging
                error_type = type(e).__name__
                raise Exception(f"{error_type}: {str(e)}")
    
    async def lookup_batch_prices(self, batch: List[str], batch_id: int) -> Dict[str, str]:
        """Look up prices for a batch of vehicles (kept for backward compatibility)"""
        return await self.lookup_batch_prices_with_retry(batch, batch_id)
    
    async def process_all_batches(self, batches: List[List[str]]) -> Dict[str, str]:
        """Process all batches concurrently with improved error handling"""
        logger.info(f"Starting processing with {self.num_workers} workers, {self.max_retries} retries per batch, web search enabled")
        
        # Create tasks for all batches
        tasks = []
        for i, batch in enumerate(batches):
            task = asyncio.create_task(self.lookup_batch_prices_with_retry(batch, i+1))
            tasks.append(task)
            
            # Add delay between task creation
            if i > 0 and i % self.num_workers == 0:
                await asyncio.sleep(self.rate_limit_delay)
        
        # Wait for all tasks to complete with better error handling
        logger.info(f"Created {len(tasks)} tasks, waiting for completion...")
        
        all_prices = {}
        completed_batches = 0
        failed_batches = 0
        
        # Process results as they complete
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                if isinstance(result, dict):
                    all_prices.update(result)
                    completed_batches += 1
                    
                    # Log progress every 5 batches or on completion
                    if (completed_batches + failed_batches) % 5 == 0 or (completed_batches + failed_batches) == len(tasks):
                        progress = ((completed_batches + failed_batches) / len(tasks)) * 100
                        logger.info(f"Progress: {progress:.1f}% ({completed_batches} successful, {failed_batches} failed)")
                else:
                    failed_batches += 1
                    logger.error(f"Task returned invalid result type: {type(result)}")
                    
            except Exception as e:
                failed_batches += 1
                logger.error(f"Task failed with exception: {e}")
        
        logger.info(f"Processing completed: {completed_batches} successful, {failed_batches} failed batches")
        
        return all_prices
    
    def filter_records_with_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records where no original price was found (selling_price is N/A)"""
        initial_count = len(df)
        
        # Filter out records with N/A selling prices
        df_filtered = df[df['selling_price'] != 'N/A'].copy()
        
        removed_count = initial_count - len(df_filtered)
        logger.info(f"Removed {removed_count} records with no original price found")
        logger.info(f"Remaining records: {len(df_filtered)}")
        
        return df_filtered
    
    def adjust_prices_for_cpi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust selling prices based on CPI data to current year dollars"""
        logger.info(f"Adjusting selling prices for CPI to {CURRENT_YEAR} dollars...")
        
        df_adjusted = df.copy()
        
        # Convert selling_price to numeric for calculation
        df_adjusted['selling_price_numeric'] = pd.to_numeric(df_adjusted['selling_price'], errors='coerce')
        
        # Assume there's a 'model_year' column in the dataframe
        if 'model_year' not in df_adjusted.columns:
            logger.error("Model year column not found in dataframe. Cannot adjust for CPI.")
            return df_adjusted
        
        # Get current year CPI
        current_year_str = str(CURRENT_YEAR)
        current_year_cpi = self.cpi_data[current_year_str]
        
        adjusted_count = 0
        missing_cpi_data = 0
        
        for idx, row in df_adjusted.iterrows():
            model_year = str(int(row['model_year'])) if pd.notna(row['model_year']) else None
            selling_price = row['selling_price_numeric']
            
            if model_year and pd.notna(selling_price) and model_year in self.cpi_data:
                transaction_year_cpi = self.cpi_data[model_year]
                # Calculate multiplier: CPI[current_year] / CPI[transaction_year]
                cpi_multiplier = current_year_cpi / transaction_year_cpi
                adjusted_price = int(selling_price * cpi_multiplier)
                df_adjusted.loc[idx, 'selling_price'] = str(adjusted_price)
                adjusted_count += 1
                
                if adjusted_count <= 5:  # Log first few adjustments for verification
                    logger.info(f"Model year {model_year}: {selling_price} -> {adjusted_price} (CPI: {transaction_year_cpi} -> {current_year_cpi}, multiplier: {cpi_multiplier:.3f})")
                    
            elif model_year and model_year not in self.cpi_data:
                missing_cpi_data += 1
                logger.warning(f"No CPI data found for model year {model_year}")
        
        # Drop the temporary numeric column
        df_adjusted = df_adjusted.drop('selling_price_numeric', axis=1)
        
        logger.info(f"Adjusted prices for {adjusted_count} records using CPI")
        if missing_cpi_data > 0:
            logger.warning(f"Missing CPI data for {missing_cpi_data} records")
        
        return df_adjusted
    
    def calculate_depreciation_constant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate depreciation constant using exponential decay model"""
        logger.info("Calculating depreciation constants...")
        
        df_processed = df.copy()
        
        # Convert to numeric
        df_processed['selling_price_numeric'] = pd.to_numeric(df_processed['selling_price'], errors='coerce')
        df_processed['current_price_numeric'] = pd.to_numeric(df_processed['price'], errors='coerce')
        df_processed['age_numeric'] = pd.to_numeric(df_processed['age'], errors='coerce')
        
        # Initialize depreciation constant column
        df_processed['depreciation_constant'] = np.nan
        
        valid_count = 0
        
        for idx, row in df_processed.iterrows():
            V_0 = row['selling_price_numeric']  # Original price (inflation adjusted)
            V_t = row['current_price_numeric']  # Current price
            t = row['age_numeric']              # Age in years
            
            # Check if all values are valid
            if pd.notna(V_0) and pd.notna(V_t) and pd.notna(t) and V_0 > 0 and V_t > 0 and t > 0:
                try:
                    # Calculate: k = -ln(V_t/V_0) / t
                    k = -np.log(V_t / V_0) / t
                    
                    if not (np.isinf(k) or np.isnan(k)):
                        df_processed.loc[idx, 'depreciation_constant'] = k
                        valid_count += 1
                except:
                    continue
        
        # Drop temporary columns
        df_processed = df_processed.drop(['selling_price_numeric', 'current_price_numeric', 'age_numeric'], axis=1)
        
        logger.info(f"Calculated depreciation for {valid_count} vehicles")
        
        return df_processed
    
    def remove_unnecessary_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove vehicle, selling_price, and price columns"""
        columns_to_remove = ['vehicle', 'selling_price', 'price']
        columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        
        if columns_to_remove:
            df_cleaned = df.drop(columns=columns_to_remove)
            logger.info(f"Removed columns: {', '.join(columns_to_remove)}")
            return df_cleaned
        
        return df
    
    async def run_complete_process(self, output_file: str = None) -> pd.DataFrame:
        """Run the complete process with hardcoded settings"""
        try:
            # Read CSV
            df = self.read_csv()
            
            # Extract vehicles
            vehicles = df['vehicle'].tolist()
            logger.info(f"Processing {len(vehicles)} vehicles")
            
            # Create batches
            batches = self.create_batches(vehicles)
            
            # Process all batches
            all_prices = await self.process_all_batches(batches)
            
            # Add prices to dataframe
            df['selling_price'] = df['vehicle'].map(all_prices)
            
            found_prices = df['selling_price'].notna().sum()
            logger.info(f"Found prices for {found_prices}/{len(df)} vehicles")
            
            # NEW STEP 1: Remove records with no original price found
            df = self.filter_records_with_prices(df)
            
            # NEW STEP 2: Adjust selling prices for CPI
            df = self.adjust_prices_for_cpi(df)
            
            # Calculate depreciation constant
            df = self.calculate_depreciation_constant(df)
            
            # Remove unnecessary fields
            df = self.remove_unnecessary_fields(df)
            
            # Save to file
            if output_file is None:
                output_file = self.csv_file_path
            
            df.to_csv(output_file, index=False)
            logger.info(f"Saved results to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in complete process: {e}")
            raise
    
    def run(self, output_file: str = None) -> pd.DataFrame:
        """Synchronous wrapper for the async process"""
        return asyncio.run(self.run_complete_process(output_file))

def main():
    """Main function"""
    print("=== Simple Vehicle Price Lookup Tool ===")
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    
    # Initialize and run
    lookup_tool = SimpleVehiclePriceLookup(api_key)
    
    try:
        start_time = datetime.now()
        result_df = lookup_tool.run()
        end_time = datetime.now()
        
        print(f"\n=== Results ===")
        print(f"Processing time: {end_time - start_time}")
        print(f"Total vehicles: {len(result_df)}")
        print("\nFirst few rows:")
        print(result_df.head())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()