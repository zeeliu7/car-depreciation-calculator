from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
import os
import logging
import json
import numpy as np
from contextlib import asynccontextmanager

from src import CarDepreciationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cpi_data() -> Dict[str, float]:
    try:
        with open('cpi.json', 'r') as f:
            cpi_data = json.load(f)
        return {int(year): float(cpi) for year, cpi in cpi_data.items()}
    except Exception as e:
        logger.error(f"Failed to load CPI data: {str(e)}")
        # Return empty dict, will cause errors if not available
        return {}

# Global CPI data
CPI_DATA = load_cpi_data()
CURRENT_YEAR = 2025

def calculate_inflation_multiplier(transaction_year: int, current_year: int = CURRENT_YEAR) -> float:
    """Calculate inflation multiplier using CPI data"""
    if current_year not in CPI_DATA or transaction_year not in CPI_DATA:
        logger.warning(f"CPI data not available for years {transaction_year} or {current_year}")
        return 1.0  # Default to no adjustment if data unavailable
    
    multiplier = CPI_DATA[current_year] / CPI_DATA[transaction_year]
    return multiplier

def calculate_age_from_model_year(model_year: int, current_year: int = CURRENT_YEAR) -> int:
    """Calculate car age from model year, minimum age is 1"""
    age = current_year - model_year
    return max(1, age)  # Minimum age is 1

def calculate_depreciation_constant(current_value: float, selling_price: float, 
                                  transaction_year: int, age: int) -> float:
    """Calculate depreciation constant using the formula: k = -ln(V_t / V_0) / t"""
    try:
        # Adjust selling price for inflation
        multiplier = calculate_inflation_multiplier(transaction_year)
        adjusted_selling_price = selling_price * multiplier  # V_0
        
        # Calculate depreciation constant: k = -ln(V_t / V_0) / t
        if adjusted_selling_price <= 0 or current_value <= 0 or age <= 0:
            logger.warning("Invalid values for depreciation calculation")
            return 0.0
        
        k = -np.log(current_value / adjusted_selling_price) / age
        return float(k)
        
    except Exception as e:
        logger.error(f"Error calculating depreciation constant: {str(e)}")
        return 0.0

def calculate_future_value(current_value: float, depreciation_constant: float, years_ahead: int) -> float:
    """Calculate future value using exponential depreciation: V(t) = V_0 * e^(-k*t)"""
    try:
        if current_value <= 0 or years_ahead < 0:
            return 0.0
        
        future_value = current_value * np.exp(-depreciation_constant * years_ahead)
        return max(0.0, float(future_value))  # Ensure non-negative
        
    except Exception as e:
        logger.error(f"Error calculating future value: {str(e)}")
        return 0.0

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    mileage: int = Field(..., ge=0, le=500000, description="Car mileage in miles")
    fuel_type: str = Field(..., description="Type of fuel")
    transmission: str = Field(..., description="Transmission type")
    accident: str = Field(..., description="Accident history (Yes/No/Unknown)")
    clean_title: str = Field(..., description="Clean title status (Yes/No)")
    model_year: int = Field(..., ge=1970, le=CURRENT_YEAR, description="Car model year")
    
    @field_validator('fuel_type')
    @classmethod
    def validate_fuel_type(cls, v):
        allowed_types = ["Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel", "Others"]
        if v not in allowed_types:
            raise ValueError(f'fuel_type must be one of: {allowed_types}')
        return v
    
    @field_validator('transmission')
    @classmethod
    def validate_transmission(cls, v):
        allowed_types = ['Automatic', 'Manual', 'CVT', 'Dual Switch', 'Overdrive Switch', 'Others']
        if v not in allowed_types:
            raise ValueError(f'transmission must be one of: {allowed_types}')
        return v
    
    @field_validator('accident')
    @classmethod
    def validate_accident(cls, v):
        if v not in ['Yes', 'No', 'Unknown']:
            raise ValueError('Must be "Yes", "No", or "Unknown"')
        return v
    
    @field_validator('clean_title')
    @classmethod
    def validate_clean_title(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Must be "Yes" or "No"')
        return v

class FuturePredictionRequest(BaseModel):
    mileage: int = Field(..., ge=0, le=500000, description="Car mileage in miles")
    fuel_type: str = Field(..., description="Type of fuel")
    transmission: str = Field(..., description="Transmission type")
    accident: str = Field(..., description="Accident history (Yes/No/Unknown)")
    clean_title: str = Field(..., description="Clean title status (Yes/No)")
    model_year: int = Field(..., ge=1900, le=CURRENT_YEAR, description="Car model year")
    selling_price: float = Field(..., gt=0, description="Original selling price")
    selling_year: int = Field(..., ge=1970, le=CURRENT_YEAR, description="Year when car was originally sold")
    
    @field_validator('fuel_type')
    @classmethod
    def validate_fuel_type(cls, v):
        allowed_types = ["Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel", "Others"]
        if v not in allowed_types:
            raise ValueError(f'fuel_type must be one of: {allowed_types}')
        return v
    
    @field_validator('transmission')
    @classmethod
    def validate_transmission(cls, v):
        allowed_types = ['Automatic', 'Manual', 'CVT', 'Dual Switch', 'Overdrive Switch', 'Others']
        if v not in allowed_types:
            raise ValueError(f'transmission must be one of: {allowed_types}')
        return v
    
    @field_validator('accident')
    @classmethod
    def validate_accident(cls, v):
        if v not in ['Yes', 'No', 'Unknown']:
            raise ValueError('Must be "Yes", "No", or "Unknown"')
        return v
    
    @field_validator('clean_title')
    @classmethod
    def validate_clean_title(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Must be "Yes" or "No"')
        return v

class InsertDataRequest(BaseModel):
    mileage: int = Field(..., ge=0, le=500000, description="Car mileage in miles")
    fuel_type: str = Field(..., description="Type of fuel")
    transmission: str = Field(..., description="Transmission type")
    accident: str = Field(..., description="Accident history (Yes/No/Unknown)")
    clean_title: str = Field(..., description="Clean title status (Yes/No)")
    model_year: int = Field(..., ge=1900, le=CURRENT_YEAR, description="Car model year")
    selling_price: float = Field(..., gt=0, description="Original selling price (before inflation adjustment)")
    current_price: float = Field(..., gt=0, description="Current market value")
    transaction_year: Optional[int] = Field(None, ge=1900, le=CURRENT_YEAR, description="Year when car was originally sold (optional, defaults to model year)")
    
    @field_validator('fuel_type')
    @classmethod
    def validate_fuel_type(cls, v):
        allowed_types = ["Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel", "Others"]
        if v not in allowed_types:
            raise ValueError(f'fuel_type must be one of: {allowed_types}')
        return v
    
    @field_validator('transmission')
    @classmethod
    def validate_transmission(cls, v):
        allowed_types = ['Automatic', 'Manual', 'CVT', 'Dual Switch', 'Overdrive Switch', 'Others']
        if v not in allowed_types:
            raise ValueError(f'transmission must be one of: {allowed_types}')
        return v
    
    @field_validator('accident')
    @classmethod
    def validate_accident(cls, v):
        if v not in ['Yes', 'No', 'Unknown']:
            raise ValueError('Must be "Yes", "No", or "Unknown"')
        return v
    
    @field_validator('clean_title')
    @classmethod
    def validate_clean_title(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Must be "Yes" or "No"')
        return v

class PredictionResponse(BaseModel):
    depreciation_constant: float = Field(..., description="Predicted depreciation constant")
    model_version: str = Field(..., description="Version of the model used")

class FuturePredictionResponse(BaseModel):
    data: str = Field(..., description="String summary of future price predictions")

class InsertDataResponse(BaseModel):
    success: bool = Field(..., description="Whether the insert operation was successful")
    message: str = Field(..., description="Status message")
    calculated_age: int = Field(..., description="Calculated age from model year")
    calculated_depreciation_constant: float = Field(..., description="Calculated depreciation constant")
    inflation_multiplier: float = Field(..., description="Applied inflation multiplier")
    adjusted_selling_price: float = Field(..., description="Inflation-adjusted selling price")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str
    cpi_data_loaded: bool

# Global variables
depreciation_system = None
model_version = "1.0.0"

def calculate_confidence_score(prediction: float, input_features: Dict[str, Any]) -> float:
    """
    Calculate a confidence score based on prediction value and input features
    """
    try:
        # Base confidence on how close prediction is to expected range
        expected_range = (0.02, 0.15)  # Typical depreciation range
        
        if expected_range[0] <= prediction <= expected_range[1]:
            base_confidence = 0.9
        else:
            # Penalize predictions outside expected range
            distance = min(abs(prediction - expected_range[0]), abs(prediction - expected_range[1]))
            base_confidence = max(0.3, 0.9 - distance * 5)
        
        # Adjust based on input features
        age = input_features.get('age', 0)
        mileage = input_features.get('mileage', 0)
        
        # Higher confidence for typical age/mileage combinations
        if 1 <= age <= 15 and 0 <= mileage <= 200000:
            age_mileage_confidence = 1.0
        else:
            age_mileage_confidence = 0.7
        
        # Combine confidences
        final_confidence = (base_confidence + age_mileage_confidence) / 2
        return min(1.0, max(0.1, final_confidence))
        
    except Exception as e:
        logger.warning(f"Error calculating confidence score: {str(e)}")
        return 0.5

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global depreciation_system
    try:
        logger.info("Starting up FastAPI application...")
        
        # Check CPI data
        if not CPI_DATA:
            logger.error("CPI data not loaded! Depreciation calculations will be inaccurate.")
        else:
            logger.info(f"CPI data loaded for years {min(CPI_DATA.keys())}-{max(CPI_DATA.keys())}")
        
        # Get configuration from environment variables
        bucket_name = os.getenv('S3_BUCKET_NAME')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not bucket_name:
            logger.warning("S3_BUCKET_NAME environment variable not set.")
            bucket_name = "default-bucket"
        
        # Initialize the depreciation system
        depreciation_system = CarDepreciationSystem(bucket_name=bucket_name, aws_region=aws_region)
        
        # Test model loading
        try:
            test_features = {
                'mileage': 50000,
                'fuel_type': 'Gasoline',
                'transmission': 'Automatic',
                'accident': 'No',
                'clean_title': 'Yes',
                'age': 5
            }
            _ = depreciation_system.predict_depreciation(test_features)
            logger.info("Model loaded and tested successfully on startup")
        except Exception as e:
            logger.warning(f"Model test failed on startup: {str(e)}")
        
    except Exception as e:
        logger.error(f"Failed to initialize system on startup: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")

# Initialize FastAPI app
app = FastAPI(
    title="Car Depreciation Prediction API",
    description="API for predicting car depreciation constants and future values using machine learning",
    version=model_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service and model status"""
    global depreciation_system
    
    if depreciation_system is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message="Depreciation system not initialized",
            cpi_data_loaded=bool(CPI_DATA)
        )
    
    # Test if model can make predictions
    model_loaded = True
    try:
        test_features = {
            'mileage': 50000,
            'fuel_type': 'Gasoline',
            'transmission': 'Automatic',
            'accident': 'No',
            'clean_title': 'Yes',
            'age': 5
        }
        _ = depreciation_system.predict_depreciation(test_features)
    except Exception as e:
        model_loaded = False
        logger.warning(f"Health check model test failed: {str(e)}")
    
    return HealthResponse(
        status="healthy" if (model_loaded and CPI_DATA) else "unhealthy",
        model_loaded=model_loaded,
        message="Service is running" if (model_loaded and CPI_DATA) else "Service issues detected",
        cpi_data_loaded=bool(CPI_DATA)
    )

# Prediction endpoint for depreciation constant
@app.post("/api/predict/constant", response_model=PredictionResponse)
async def predict_depreciation_constant(request: PredictionRequest):
    """
    Predict car depreciation constant based on vehicle features
    
    - **mileage**: Current mileage of the vehicle
    - **fuel_type**: Type of fuel (Gasoline, Hybrid, Diesel, Plug-In Hybrid, E85 Flex Fuel, Others)
    - **transmission**: Transmission type (Automatic, Manual, CVT, Dual Switch, Overdrive Switch, Others)
    - **accident**: Whether the car has accident history (Yes/No/Unknown)
    - **clean_title**: Whether the car has a clean title (Yes/No)
    - **model_year**: Model year of the vehicle (age will be calculated automatically)
    """
    global depreciation_system
    
    if depreciation_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Depreciation system not initialized"
        )
    
    if not CPI_DATA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CPI data not available for calculations"
        )
    
    try:
        # Calculate age from model year
        age = calculate_age_from_model_year(request.model_year)
        
        # Prepare input features for prediction
        input_features = {
            'mileage': request.mileage,
            'fuel_type': request.fuel_type,
            'transmission': request.transmission,
            'accident': request.accident,
            'clean_title': request.clean_title,
            'age': age
        }
        
        logger.info(f"Received prediction request: {input_features}")
        
        # Make prediction using the CarDepreciationSystem
        prediction = depreciation_system.predict_depreciation(input_features)
        
        if prediction is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed. Please check if the model is properly trained and available."
            )
        
        logger.info(f"Prediction successful: {prediction}")
        
        return PredictionResponse(
            depreciation_constant=round(prediction, 6),
            model_version=model_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Future price predictions for 5 years
@app.post("/api/predict/future", response_model=FuturePredictionResponse)
async def predict_future_values(request: FuturePredictionRequest):
    """
    Predict future car values for the next 5 years based on selling price and depreciation constant
    
    - **mileage**: Current mileage of the vehicle
    - **fuel_type**: Type of fuel (Gasoline, Hybrid, Diesel, Plug-In Hybrid, E85 Flex Fuel, Others)
    - **transmission**: Transmission type (Automatic, Manual, CVT, Dual Switch, Overdrive Switch, Others)
    - **accident**: Whether the car has accident history (Yes/No/Unknown)
    - **clean_title**: Whether the car has a clean title (Yes/No)
    - **model_year**: Model year of the vehicle (age will be calculated automatically)
    - **selling_price**: Original selling price (e.g., $25,000 in 2020)
    - **selling_year**: Year when car was originally sold (e.g., 2020)
    """
    global depreciation_system
    
    if depreciation_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Depreciation system not initialized"
        )
    
    try:
        # Calculate age from model year
        age = calculate_age_from_model_year(request.model_year)
        
        # Prepare input features for prediction
        input_features = {
            'mileage': request.mileage,
            'fuel_type': request.fuel_type,
            'transmission': request.transmission,
            'accident': request.accident,
            'clean_title': request.clean_title,
            'age': age
        }
        
        logger.info(f"Received future prediction request: {input_features}")
        
        # Get depreciation constant prediction
        depreciation_constant = depreciation_system.predict_depreciation(input_features)
        
        if depreciation_constant is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Depreciation constant prediction failed. Please check if the model is properly trained and available."
            )
        
        # Calculate current value (2025) from selling price using depreciation formula
        # V(t) = V_0 * e^(-k*t) where t is years since selling year
        years_since_selling = CURRENT_YEAR - request.selling_year
        current_value = request.selling_price * np.exp(-depreciation_constant * years_since_selling)
        
        # Calculate future values for next 5 years
        prediction_lines = []
        prediction_lines.append(f"Future Value Predictions:")
        prediction_lines.append(f"Original Price ({request.selling_year}): ${request.selling_price:,.2f}")
        prediction_lines.append(f"Depreciation Constant: {depreciation_constant:.6f}")
        prediction_lines.append(f"Current Value ({CURRENT_YEAR}): ${current_value:,.2f}")
        prediction_lines.append("")
        
        for year_offset in range(1, 6):
            future_year = CURRENT_YEAR + year_offset
            years_from_selling = years_since_selling + year_offset
            future_value = request.selling_price * np.exp(-depreciation_constant * years_from_selling)
            percentage_of_original = (future_value / request.selling_price) * 100 if request.selling_price > 0 else 0
            
            prediction_lines.append(f"Year {future_year}: ${future_value:,.2f} ({percentage_of_original:.1f}% of original)")
        
        prediction_summary = "\n".join(prediction_lines)
        
        logger.info(f"Future prediction successful for depreciation constant: {depreciation_constant}")
        
        return FuturePredictionResponse(
            data=prediction_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during future prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Data insertion endpoint
@app.post("/api/donate", response_model=InsertDataResponse)
async def insert_car_data(request: InsertDataRequest):
    """
    Insert a new car record into the database
    
    This endpoint calculates age and depreciation constant automatically.
    
    - **mileage**: Current mileage of the vehicle
    - **fuel_type**: Type of fuel (Gasoline, Hybrid, Diesel, Plug-In Hybrid, E85 Flex Fuel, Others)
    - **transmission**: Transmission type (Automatic, Manual, CVT, Dual Switch, Overdrive Switch, Others)
    - **accident**: Whether the car has accident history (Yes/No/Unknown)
    - **clean_title**: Whether the car has a clean title (Yes/No)
    - **model_year**: Model year of the vehicle
    - **selling_price**: Original selling price (will be adjusted for inflation)
    - **current_price**: Current market value of the vehicle
    - **transaction_year**: Year when car was originally sold (optional, defaults to model year)
    """
    global depreciation_system
    
    if depreciation_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Depreciation system not initialized"
        )
    
    if not CPI_DATA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CPI data not available for calculations"
        )
    
    try:
        # Calculate age from model year
        age = calculate_age_from_model_year(request.model_year)
        
        # Use transaction_year if provided, otherwise use model_year
        transaction_year = request.transaction_year if request.transaction_year else request.model_year
        
        # Calculate inflation multiplier and adjusted selling price
        inflation_multiplier = calculate_inflation_multiplier(transaction_year)
        adjusted_selling_price = request.selling_price * inflation_multiplier
        
        # Calculate depreciation constant
        depreciation_constant = calculate_depreciation_constant(
            current_value=request.current_price,
            selling_price=request.selling_price,
            transaction_year=transaction_year,
            age=age
        )
        
        # Prepare record for insertion
        record_data = {
            'mileage': request.mileage,
            'fuel_type': request.fuel_type,
            'transmission': request.transmission,
            'accident': request.accident,
            'clean_title': request.clean_title,
            'age': age,
            'depreciation_constant': depreciation_constant
        }
        
        logger.info(f"Inserting record: {record_data}")
        logger.info(f"Calculated values - Age: {age}, Depreciation: {depreciation_constant:.6f}, Multiplier: {inflation_multiplier:.4f}")
        
        # Insert data using the CarDepreciationSystem
        success = depreciation_system.insert_data_record(record_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to insert data record. Check server logs for details."
            )
        
        logger.info("Thank you for your case donation!")
        
        return InsertDataResponse(
            success=True,
            message="Record inserted successfully",
            calculated_age=age,
            calculated_depreciation_constant=round(depreciation_constant, 6),
            inflation_multiplier=round(inflation_multiplier, 4),
            adjusted_selling_price=round(adjusted_selling_price, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data insertion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Root endpoint
@app.get("/api")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Car Depreciation Prediction API",
        "version": model_version,
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

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    # Validate required files
    if not os.path.exists('cpi.json'):
        logger.error("cpi.json file not found! This is required for inflation calculations.")
    
    if not os.getenv('S3_BUCKET_NAME'):
        logger.warning("S3_BUCKET_NAME environment variable not set!")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)