import pandas as pd
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import io
import pickle
import os
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple

load_dotenv()

class DepreciationNN(nn.Module):
    """Neural Network for depreciation prediction"""
    def __init__(self, input_size: int, hidden_sizes: list = [64, 32, 16]):
        super(DepreciationNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CarDepreciationSystem:
    def __init__(self, bucket_name: str = None, aws_region: str = None):
        # Use environment variables if not provided - IAM role handles credentials automatically
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        aws_region = aws_region or os.getenv('AWS_REGION', 'us-east-1')
        
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME must be provided or set in environment variables")
        
        # boto3 automatically uses IAM role attached to EC2 instance
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        self.data_key = 'car_depreciation_data.parquet'
        self.model_key = 'depreciation_model.pth'
        self.scaler_key = 'feature_scaler.pkl'
        self.encoders_key = 'label_encoders.pkl'
        
        self.feature_columns = ['mileage', 'fuel_type', 'transmission', 'accident', 'clean_title', 'age']
        self.target_column = 'depreciation_constant'
        
    def store_csv_to_s3(self, csv_file_path: str) -> bool:
        """
        Store CSV data in S3 as Parquet format (one-time operation)
        """
        try:
            df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_cols = self.feature_columns + [self.target_column]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.data_key,
                Body=parquet_buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            print(f"Successfully stored {len(df)} records to S3 as Parquet format")
            return True
            
        except Exception as e:
            print(f"Error storing CSV to S3: {str(e)}")
            return False
    
    def insert_data_record(self, record: Dict[str, Any]) -> bool:
        """
        Insert a single record into the stored data
        """
        try:
            df = self._load_data_from_s3()
            
            required_keys = set(self.feature_columns + [self.target_column])
            if not required_keys.issubset(set(record.keys())):
                missing_keys = required_keys - set(record.keys())
                raise ValueError(f"Missing keys in record: {missing_keys}")
            
            new_record_df = pd.DataFrame([record])            
            updated_df = pd.concat([df, new_record_df], ignore_index=True)
            parquet_buffer = io.BytesIO()
            updated_df.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.data_key,
                Body=parquet_buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            print(f"Successfully inserted record. Total records: {len(updated_df)}")
            return True
            
        except Exception as e:
            print(f"Error inserting record: {str(e)}")
            return False
    
    def _load_data_from_s3(self) -> pd.DataFrame:
        """Load data from S3 Parquet file"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.data_key)
            parquet_data = response['Body'].read()
            df = pd.read_parquet(io.BytesIO(parquet_data))
            return df
        except Exception as e:
            print(f"Error loading data from S3: {str(e)}")
            return pd.DataFrame()
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler, Dict]:
        """Preprocess data for neural network training"""
        X = df[self.feature_columns].copy()
        y = df[self.target_column].values
        
        # Handle categorical variables
        label_encoders = {}
        categorical_cols = ['fuel_type', 'transmission', 'accident', 'clean_title']
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        return X_tensor, y_tensor, scaler, label_encoders

    def train_model(self, csv_file_path: str = None, epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train PyTorch neural network model using all stored data
        """
        try:
            if csv_file_path:
                self.store_csv_to_s3(csv_file_path)
            
            df = self._load_data_from_s3()
            if df.empty:
                raise ValueError("No data available for training")
            
            print(f"Training on {len(df)} records")
            
            # Preprocess data
            X_tensor, y_tensor, scaler, label_encoders = self._preprocess_data(df)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=42
            )
            
            # Initialize model
            input_size = X_tensor.shape[1]
            model = DepreciationNN(input_size)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            train_losses = []
            val_losses = []
            
            model.train()
            for epoch in range(epochs):
                # Training
                optimizer.zero_grad()
                train_pred = model(X_train)
                train_loss = criterion(train_pred, y_train)
                train_loss.backward()
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = criterion(val_pred, y_val)
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
                
                model.train()
            
            # Save model and preprocessors to S3
            self._save_model_to_s3(model, scaler, label_encoders)
            
            # Calculate final metrics
            model.eval()
            with torch.no_grad():
                final_pred = model(X_val)
                mse = criterion(final_pred, y_val).item()
                mae = torch.mean(torch.abs(final_pred - y_val)).item()
            
            results = {
                'final_mse': mse,
                'final_mae': mae,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            print(f"Training completed. Final MSE: {mse:.6f}, MAE: {mae:.6f}")
            return results
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return {}
    
    def evaluate_all_predictions(self) -> Dict[str, Any]:
        """Evaluate model performance on all stored data"""
        try:
            df = self._load_data_from_s3()
            if df.empty:
                print("No data available for evaluation")
                return {}
            
            model, scaler, label_encoders = self._load_model_from_s3()
            if model is None:
                print("Model not found")
                return {}
            
            # Prepare features
            X_tensor, y_tensor, _, _ = self._preprocess_data(df)
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor)
            
            # Calculate metrics
            y_actual = y_tensor.numpy().flatten()
            y_pred = predictions.numpy().flatten()
            
            mae = np.mean(np.abs(y_actual - y_pred))
            mse = np.mean((y_actual - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            return {
                'mean_absolute_error': float(mae),
                'mean_squared_error': float(mse),
                'root_mean_squared_error': float(rmse),
                'mean_actual_value': float(np.mean(y_actual)),
                'mean_predicted_value': float(np.mean(y_pred)),
                'prediction_bias': float(np.mean(y_pred - y_actual)),
                'total_samples': len(y_actual)
            }
        except Exception as e:
            print(f"Error evaluating predictions: {str(e)}")
            return {}
    
    def _save_model_to_s3(self, model: DepreciationNN, scaler: StandardScaler, label_encoders: Dict):
        """Save model and preprocessors to S3"""
        try:
            # Save model
            model_buffer = io.BytesIO()
            torch.save(model.state_dict(), model_buffer)
            model_buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.model_key,
                Body=model_buffer.getvalue()
            )
            
            # Save scaler
            scaler_buffer = io.BytesIO()
            pickle.dump(scaler, scaler_buffer)
            scaler_buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.scaler_key,
                Body=scaler_buffer.getvalue()
            )
            
            # Save label encoders
            encoders_buffer = io.BytesIO()
            pickle.dump(label_encoders, encoders_buffer)
            encoders_buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.encoders_key,
                Body=encoders_buffer.getvalue()
            )
            
        except Exception as e:
            print(f"Error saving model to S3: {str(e)}")
    
    def _load_model_from_s3(self) -> Tuple[DepreciationNN, StandardScaler, Dict]:
        """Load model and preprocessors from S3"""
        try:
            # Load preprocessors first to determine input size
            scaler_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.scaler_key)
            scaler = pickle.load(io.BytesIO(scaler_response['Body'].read()))
            
            encoders_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.encoders_key)
            label_encoders = pickle.load(io.BytesIO(encoders_response['Body'].read()))
            
            # Load model
            model_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.model_key)
            
            # Initialize model with correct input size
            input_size = len(self.feature_columns)
            model = DepreciationNN(input_size)
            model.load_state_dict(torch.load(io.BytesIO(model_response['Body'].read())))
            model.eval()
            
            return model, scaler, label_encoders
            
        except Exception as e:
            print(f"Error loading model from S3: {str(e)}")
            return None, None, None
    
    def predict_depreciation(self, input_features: Dict[str, Any]) -> float:
        """
        Predict depreciation constant for given features
        """
        try:
            # Load model and preprocessors
            model, scaler, label_encoders = self._load_model_from_s3()
            if model is None:
                raise ValueError("Model not found. Please train the model first.")
            
            # Validate input features
            missing_features = set(self.feature_columns) - set(input_features.keys())
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Prepare input data
            input_df = pd.DataFrame([input_features])
            
            # Apply label encoding for categorical variables
            categorical_cols = ['fuel_type', 'transmission', 'accident', 'clean_title']
            for col in categorical_cols:
                if col in input_df.columns and col in label_encoders:
                    try:
                        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        print(f"Warning: Unknown category in {col}. Using most frequent category.")
                        input_df[col] = 0  # Default to first encoded value
            
            # Scale features
            input_scaled = scaler.transform(input_df[self.feature_columns])
            input_tensor = torch.FloatTensor(input_scaled)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(input_tensor).item()
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None