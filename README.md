# Car Depreciation Calculator: Quote Your Car's Price Instantly!

An API service that predicts your second-hand car's price with just a few factors for free!

## Features
- ✅ Instant depreciation constant predictions
- ✅ 5-year future value forecasts
- ✅ Inflation-adjusted calculations using CPI data
- ✅ Trained on 50+ years of vehicle data
- ✅ Free to use, no authentication required

## How it works
* Trained on vehicle data based on mileage, fuel type, transmission, accident history, clean title or not, and age.
* Prices adjusted for inflation to reflect actual depreciation based on vehicle properties
* Depreciation constants were calculated over exponential formula: $V(t) = V_0 \cdot e^{(-k \cdot t)}$
    * $V_0$ is the original selling price (after inflation adjustment)
    * $t$ is the age in years
    * $V(t)$ is the price after $t$ years
    * $k$ is the depreciation constant

## API usages
Base URL: `car-depreciation-calculator.fyi`

Currently all input years are limited to 1970-2025. Model trained over data between 1974-2024 in U.S. dollars. This program will be updated once 2026's CPI is available.

### Get depreciation constant
Predict the depreciation constant for a vehicle based on its characteristics. Recommended for researchers/buyers interested in depreciation rate.

`POST <base>/api/predict/constant`
* `mileage` (integer, required): Car mileage in miles
    * Minimum 0
* `fuel_type` (string, required): Type of fuel
    * Allowed values: "Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel", "Others"
* `transmission` (string, required): Transmission type
    * Allowed values: "Automatic", "Manual", "CVT", "Dual Switch", "Overdrive Switch", "Others"
* `accident` (string, required): Accident history
    * Allowed values: "Yes", "No", "Unknown"
* `clean_title` (string, required): Clean title status
    * Allowed values: "Yes", "No"
* `model_year` (integer, required): Vehicle model year
    * Limited to 1970-2025

Example request:
```bash
curl -X POST http://car-depreciation-calculator.fyi/api/predict/constant \
    -H "Content-Type: application/json" \
    -d '{
        "mileage": 25000,
        "fuel_type": "Hybrid",
        "transmission": "CVT",
        "accident": "No",
        "clean_title": "Yes",
        "model_year": 2018
    }'
```

Response example:
```json
{
    "depreciation_constant":0.071291,
    "model_version":"1.0.0"
}
```

### Get 5-year predictions
Calculate future vehicle values for the next 5 years based on original selling price and predicted depreciation. Recommended for car owners with specific purchase information.

`POST <base>/api/predict/future`
* `mileage` (integer, required): Car mileage in miles
    * Minimum 0
* `fuel_type` (string, required): Type of fuel
    * Allowed values: "Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel", "Others"
* `transmission` (string, required): Transmission type
    * Allowed values: "Automatic", "Manual", "CVT", "Dual Switch", "Overdrive Switch", "Others"
* `accident` (string, required): Accident history
    * Allowed values: "Yes", "No", "Unknown"
* `clean_title` (string, required): Clean title status
    * Allowed values: "Yes", "No"
* `model_year` (integer, required): Vehicle model year
    * 1970-2025 only
* `selling_price` (number, required): Price when the car was sold (either new or second-hand)
    * Must be above 0
* `selling_year` (integer, required): Year when car was originally sold
    * Limited to 1970-2025

Request example:
```bash
curl -X POST http://car-depreciation-calculator.fyi/api/predict/future \
    -H "Content-Type: application/json" \
    -d '{
        "mileage": 50000,
        "fuel_type": "Gasoline",
        "transmission": "Automatic",
        "accident": "No",
        "clean_title": "Yes",
        "model_year": 2018,
        "selling_price": 25000.00,
        "selling_year": 2020
    }'
```

Response example:
```json
{
    "data": "Future Value Predictions:\nOriginal Price (2020): $25,000.00\nDepreciation Constant: 0.061117\nCurrent Value (2025): $18,417.32\n\nYear 2026: $17,325.42 (69.3% of original)\nYear 2027: $16,298.25 (65.2% of original)\nYear 2028: $15,331.99 (61.3% of original)\nYear 2029: $14,423.00 (57.7% of original)\nYear 2030: $13,567.91 (54.3% of original)"
}
```
Formatted output after parsing:
```
Future Value Predictions:
Original Price (2020): $25,000.00
Depreciation Constant: 0.061117
Current Value (2025): $18,417.32

Year 2026: $17,325.42 (69.3% of original)
Year 2027: $16,298.25 (65.2% of original)
Year 2028: $15,331.99 (61.3% of original)
Year 2029: $14,423.00 (57.7% of original)
Year 2030: $13,567.91 (54.3% of original)
```

### Donate a case
Donate a real case to help improve the prediction model. This endpoint calculates depreciation constants automatically based on your input. **Thank you for your donation!**

`POST <base>/api/donate`
* `mileage` (integer, required): Car mileage in miles
    * Minimum 0
* `fuel_type` (string, required): Type of fuel
    * Allowed values: "Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel", "Others"
* `transmission` (string, required): Transmission type
    * Allowed values: "Automatic", "Manual", "CVT", "Dual Switch", "Overdrive Switch", "Others"
* `accident` (string, required): Accident history
    * Allowed values: "Yes", "No", "Unknown"
* `clean_title` (string, required): Clean title status
    * Allowed values: "Yes", "No"
* `model_year` (integer, required): Vehicle model year
    * 1970-2025 only
* `selling_price` (number, required): Price when the car was sold (either new or second-hand)
    * Must be above 0
* `current_price` (number, required): Current market value
    * Must be above 0
* `transaction_year` (integer, optional): Year when car was originally sold
    * Defaults to model year if not provided

Request example:
```bash
curl -X POST http://car-depreciation-calculator.fyi/api/donate \
    -H "Content-Type: application/json" \
    -d '{
        "mileage": 75000,
        "fuel_type": "Gasoline",
        "transmission": "Automatic",
        "accident": "No",
        "clean_title": "Yes",
        "model_year": 2019,
        "selling_price": 28000.00,
        "current_price": 22000.00,
        "transaction_year": 2021
    }'
```

Response example:
```json
{
    "success":true,
    "message":"Record inserted successfully",
    "calculated_age":6,
    "calculated_depreciation_constant":0.072572,
    "inflation_multiplier":1.2144,
    "adjusted_selling_price":34003.82
}
```

Response fields:
* `success` (boolean): Whether the data insertion was successful
* `message` (string): Status message
* `calculated_age` (integer): Calculated vehicle age from model year
* `calculated_depreciation_constant` (number): Calculated depreciation constant
* `inflation_multiplier` (number): Applied CPI inflation multiplier
* `adjusted_selling_price` (number): Inflation-adjusted selling price

### API information
Get general information about the API service.

`GET <base>/api`

Response:
```json
{
  "message": "Car Depreciation Prediction API",
  "version": "1.0.0",
  "system": "CarDepreciationSystem",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "predict_constant": "/predict/constant",
    "predict_future": "/predict/future",
    "insert_data": "/data/insert"
  },
  "features": {
    "automatic_age_calculation": "Age calculated from model year",
    "inflation_adjustment": "Selling prices adjusted using CPI data",
    "depreciation_calculation": "Automatic calculation using k = -ln(V_t / V_0) / t",
    "future_predictions": "5-year future value predictions using exponential depreciation model"
  }
}
```

### API health check
Check the API service status and model availability.

`GET <base>/api/health`

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Service is running",
  "cpi_data_loaded": true
}
```

Response Fields:
* `status` (string): Service status - "healthy" or "unhealthy"
* `model_loaded` (boolean): Whether the ML model is loaded and functional
* `message` (string): Status description
* `cpi_data_loaded` (boolean): Whether CPI inflation data is available

## Acknowledgements
* [Vehicle dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) provided by Taeef Najib on Kaggle
* [Consumer Price Index (CPI)](https://data.bls.gov/timeseries/CUUR0000SA0) provided by U.S. Bureau of Labor Statistics
* MSRP of the vehicles inside the dataset provided by OpenAI
* Web service provided by AWS