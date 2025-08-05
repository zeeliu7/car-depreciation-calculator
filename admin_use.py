"""
Admin script for Car Depreciation System
Usage examples:
    python3 admin_use.py --store --file data.csv --bucket my-bucket
    python3 admin_use.py --train --bucket my-bucket --epochs 200
    python3 admin_use.py --train --file initial_data.csv --bucket my-bucket --epochs 150 --lr 0.001
    python3 admin_use.py --evaluate --bucket my-bucket
    python3 admin_use.py --status --bucket my-bucket
"""

import argparse
import os
import sys
import logging
from typing import Optional
import json

from car_depreciation_system import CarDepreciationSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_system(bucket_name: str, aws_region: str = 'us-east-1') -> CarDepreciationSystem:
    """Initialize the CarDepreciationSystem"""
    logger.info(f"Initializing CarDepreciationSystem with bucket: {bucket_name}")
    return CarDepreciationSystem(bucket_name=bucket_name, aws_region=aws_region)

def store_csv_command(system: CarDepreciationSystem, csv_file: str) -> bool:
    """Execute the store CSV command"""
    logger.info(f"Storing CSV file: {csv_file}")
    
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return False
    
    success = system.store_csv_to_s3(csv_file)
    
    if success:
        logger.info("CSV file stored successfully in S3")
        
        # Show some statistics
        try:
            df = system._load_data_from_s3()
            logger.info(f"Total records in database: {len(df)}")
            logger.info(f"Data columns: {list(df.columns)}")
            if not df.empty:
                logger.info(f"Sample record: {df.iloc[0].to_dict()}")
        except Exception as e:
            logger.warning(f"Could not retrieve statistics: {str(e)}")
    else:
        logger.error("Failed to store CSV file")
    
    return success

def train_model_command(system: CarDepreciationSystem, csv_file: Optional[str] = None, 
                       epochs: int = 100, learning_rate: float = 0.001) -> bool:
    """Execute the train model command"""
    logger.info(f"Training model with epochs={epochs}, learning_rate={learning_rate}")
    
    if csv_file:
        logger.info(f"Using initial CSV file: {csv_file}")
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}")
            return False
    
    try:
        results = system.train_model(
            csv_file_path=csv_file,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        if results:
            logger.info("Model training completed successfully!")
            logger.info("Training Results:")
            logger.info(f"   - Final MSE: {results.get('final_mse', 'N/A'):.6f}")
            logger.info(f"   - Final MAE: {results.get('final_mae', 'N/A'):.6f}")
            logger.info(f"   - Training samples: {results.get('training_samples', 'N/A')}")
            logger.info(f"   - Validation samples: {results.get('validation_samples', 'N/A')}")
            
            # Save training results to file
            results_file = 'training_results.json'
            with open(results_file, 'w') as f:
                # Convert numpy floats to regular floats for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, list):
                        json_results[key] = [float(v) for v in value]
                    else:
                        json_results[key] = float(value) if isinstance(value, (int, float)) else value
                json.dump(json_results, f, indent=2)
            logger.info(f"Training results saved to: {results_file}")
            
            return True
        else:
            logger.error("Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        return False

def evaluate_model_command(system: CarDepreciationSystem) -> bool:
    """Execute the model evaluation command"""
    logger.info("Evaluating model on all stored data...")
    
    try:
        results = system.evaluate_all_predictions()
        
        if results:
            logger.info("Model evaluation completed successfully!")
            logger.info("Evaluation Results:")
            logger.info(f"   - Mean Absolute Error: {results.get('mean_absolute_error', 'N/A'):.6f}")
            logger.info(f"   - Mean Squared Error: {results.get('mean_squared_error', 'N/A'):.6f}")
            logger.info(f"   - Root Mean Squared Error: {results.get('root_mean_squared_error', 'N/A'):.6f}")
            logger.info(f"   - Mean Actual Value: {results.get('mean_actual_value', 'N/A'):.6f}")
            logger.info(f"   - Mean Predicted Value: {results.get('mean_predicted_value', 'N/A'):.6f}")
            logger.info(f"   - Prediction Bias: {results.get('prediction_bias', 'N/A'):.6f}")
            logger.info(f"   - Total Samples: {results.get('total_samples', 'N/A')}")
            
            # Save evaluation results to file
            results_file = 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to: {results_file}")
            
            return True
        else:
            logger.error("Model evaluation failed")
            return False
            
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        return False

def status_command(system: CarDepreciationSystem) -> bool:
    """Check system status"""
    logger.info("Checking system status...")
    
    try:
        # Check data availability
        logger.info("Checking data availability...")
        df = system._load_data_from_s3()
        data_available = not df.empty
        
        if data_available:
            logger.info(f"Data available: {len(df)} records")
            logger.info(f"Data columns: {list(df.columns)}")
            
            # Show some statistics
            if 'depreciation_constant' in df.columns:
                dep_stats = df['depreciation_constant'].describe()
                logger.info(f"Depreciation constant statistics:")
                logger.info(f"   - Mean: {dep_stats['mean']:.6f}")
                logger.info(f"   - Std: {dep_stats['std']:.6f}")
                logger.info(f"   - Min: {dep_stats['min']:.6f}")
                logger.info(f"   - Max: {dep_stats['max']:.6f}")
        else:
            logger.warning("No data available")
        
        # Check model availability
        logger.info("Checking model availability...")
        try:
            test_features = {
                'mileage': 50000,
                'fuel_type': 'Gasoline',
                'transmission': 'Automatic',
                'accident': 'No',
                'clean_title': 'Yes',
                'age': 5
            }
            prediction = system.predict_depreciation(test_features)
            if prediction is not None:
                logger.info(f"Model available and working (test prediction: {prediction:.6f})")
                model_available = True
            else:
                logger.warning("Model returned None prediction")
                model_available = False
        except Exception as e:
            logger.warning(f"Model not available or failed: {str(e)}")
            model_available = False
        
        # System configuration
        logger.info("System Configuration:")
        logger.info(f"   - S3 Bucket: {system.bucket_name}")
        logger.info(f"   - Data Key: {system.data_key}")
        logger.info(f"   - Model Key: {system.model_key}")
        logger.info(f"   - Feature Columns: {system.feature_columns}")
        logger.info(f"   - Target Column: {system.target_column}")
        
        # Overall status
        overall_status = data_available and model_available
        logger.info(f"Overall system status: {'READY' if overall_status else 'NOT READY'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return False

def test_prediction_command(system: CarDepreciationSystem) -> bool:
    """Test model prediction with sample data"""
    logger.info("Testing model prediction...")
    
    # Test cases
    test_cases = [
        {
            'name': 'Low mileage, new car',
            'features': {
                'mileage': 15000,
                'fuel_type': 'Gasoline',
                'transmission': 'Automatic',
                'accident': 'No',
                'clean_title': 'Yes',
                'age': 2
            }
        },
        {
            'name': 'High mileage, older car',
            'features': {
                'mileage': 150000,
                'fuel_type': 'Gasoline',
                'transmission': 'Manual',
                'accident': 'Yes',
                'clean_title': 'No',
                'age': 15
            }
        },
        {
            'name': 'Hybrid vehicle',
            'features': {
                'mileage': 30000,
                'fuel_type': 'Hybrid',
                'transmission': 'CVT',
                'accident': 'Unknown',
                'clean_title': 'Yes',
                'age': 3
            }
        },
        {
            'name': 'Plug-in Hybrid with accident',
            'features': {
                'mileage': 45000,
                'fuel_type': 'Plug-In Hybrid',
                'transmission': 'Automatic',
                'accident': 'Yes',
                'clean_title': 'Yes',
                'age': 4
            }
        }
    ]
    
    try:
        success_count = 0
        for test_case in test_cases:
            logger.info(f"Testing: {test_case['name']}")
            prediction = system.predict_depreciation(test_case['features'])
            
            if prediction is not None:
                logger.info(f"   Prediction: {prediction:.6f}")
                logger.info(f"   Features: {test_case['features']}")
                success_count += 1
            else:
                logger.warning(f"   Prediction failed for {test_case['name']}")
        
        logger.info(f"Test completed: {success_count}/{len(test_cases)} predictions successful")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Prediction test failed: {str(e)}")
        return False

def insert_sample_data_command(system: CarDepreciationSystem) -> bool:
    """Insert sample data for testing"""
    logger.info("Inserting sample data...")
    
    sample_records = [
        {
            'mileage': 79800,
            'fuel_type': 'Gasoline',
            'transmission': 'Automatic',
            'accident': 'No',
            'clean_title': 'Yes',
            'age': 12,
            'depreciation_constant': 0.0760892624813204
        },
        {
            'mileage': 300183,
            'fuel_type': 'Gasoline',
            'transmission': 'Automatic',
            'accident': 'Yes',
            'clean_title': 'Yes',
            'age': 19,
            'depreciation_constant': 0.0891249594673293
        }
    ]
    
    try:
        success_count = 0
        for i, record in enumerate(sample_records):
            logger.info(f"Inserting sample record {i+1}: {record}")
            success = system.insert_data_record(record)
            if success:
                success_count += 1
            else:
                logger.warning(f"Failed to insert record {i+1}")
        
        logger.info(f"Sample data insertion completed: {success_count}/{len(sample_records)} records inserted")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Sample data insertion failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Admin tool for Car Depreciation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 admin_use.py --store --file data.csv --bucket my-bucket
  python3 admin_use.py --train --bucket my-bucket --epochs 200 --lr 0.001
  python3 admin_use.py --train --file initial_data.csv --bucket my-bucket
  python3 admin_use.py --evaluate --bucket my-bucket
  python3 admin_use.py --status --bucket my-bucket
  python3 admin_use.py --test --bucket my-bucket
  python3 admin_use.py --insert-sample --bucket my-bucket
        """
    )
    
    # Main commands (mutually exclusive)
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('--store', action='store_true', 
                              help='Store CSV file to S3')
    command_group.add_argument('--train', action='store_true', 
                              help='Train the neural network model')
    command_group.add_argument('--evaluate', action='store_true', 
                              help='Evaluate model performance on all data')
    command_group.add_argument('--status', action='store_true', 
                              help='Check system status')
    command_group.add_argument('--test', action='store_true', 
                              help='Test model predictions with sample data')
    command_group.add_argument('--insert-sample', action='store_true',
                              help='Insert sample data for testing')
    
    # Required arguments
    parser.add_argument('--bucket', type=str, required=True,
                        help='S3 bucket name')
    
    # File arguments
    parser.add_argument('--file', type=str,
                        help='CSV file path (required for --store, optional for --train)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, dest='learning_rate',
                        help='Learning rate (default: 0.001)')
    
    # AWS configuration
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region (default: us-east-1)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.store and not args.file:
        logger.error("--file is required when using --store")
        sys.exit(1)
    
    # Initialize system
    try:
        system = setup_system(args.bucket, args.region)
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        sys.exit(1)
    
    # Execute commands
    success = False
    
    if args.store:
        success = store_csv_command(system, args.file)
    elif args.train:
        success = train_model_command(system, args.file, args.epochs, args.learning_rate)
    elif args.evaluate:
        success = evaluate_model_command(system)
    elif args.status:
        success = status_command(system)
    elif args.test:
        success = test_prediction_command(system)
    elif args.insert_sample:
        success = insert_sample_data_command(system)
    
    # Exit with appropriate code
    if success:
        logger.info("Operation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()