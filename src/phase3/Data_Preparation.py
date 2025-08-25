# Data_Preparation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FlowDataPreprocessor:
    """
    Preprocessor for Flow-Level Data (Phase 3)
    """
    
    def __init__(self, data_path: str = "D:/iot_project/flow_data/processed_flows/sampled_flow_data.csv"):
        self.data_path = Path(data_path)
        self.processed_path = self.data_path.parent / "preprocessed_flows"
        self.processed_path.mkdir(exist_ok=True)
        
        self.scalers = {}
        self.encoders = {}
        self.feature_info = {}
        
        print("🤖 FlowDataPreprocessor Initialized")
        print(f"📁 Input data: {self.data_path}")
        print(f"📁 Output directory: {self.processed_path}")
    
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load and analyze flow data"""
        print("\n" + "="*50)
        print("🔍 FLOW DATA ANALYSIS")
        print("="*50)
        
        df = pd.read_csv(self.data_path)
        print(f"Flow dataset shape: {df.shape}")
        
        # Check class distribution
        if 'attack_type' in df.columns:
            class_dist = df['attack_type'].value_counts()
            print("\n📊 Class Distribution:")
            for attack_type, count in class_dist.items():
                percentage = count / len(df) * 100
                print(f"  {attack_type}: {count:,} ({percentage:.2f}%)")
        
        # Feature analysis
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove non-feature columns
        non_feature_cols = ['attack_type', 'flow_id']
        categorical_features = [col for col in categorical_features if col not in non_feature_cols]
        
        print(f"\n🔢 Feature Analysis:")
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  Categorical features: {len(categorical_features)}")
        
        # Show first 10 features of each type
        print(f"\n  First 10 numeric features: {numeric_features[:10]}")
        print(f"  First 10 categorical features: {categorical_features[:10]}")
        
        self.feature_info = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'all_features': numeric_features + categorical_features
        }
        
        return df
    
    def handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite and extremely large values"""
        print("\n" + "="*50)
        print("🔍 HANDLING INFINITE VALUES")
        print("="*50)
        
        df_processed = df.copy()
        numeric_features = self.feature_info['numeric_features']
        
        infinite_count = 0
        large_value_count = 0
        
        for column in numeric_features:
            # Replace infinite values with NaN
            infinite_mask = np.isinf(df_processed[column])
            infinite_count += infinite_mask.sum()
            df_processed[column] = df_processed[column].replace([np.inf, -np.inf], np.nan)
            
            # Identify extremely large values (beyond reasonable network traffic range)
            # For network data, we can cap at 99.9th percentile
            if df_processed[column].notna().any():
                q999 = df_processed[column].quantile(0.999)
                large_mask = df_processed[column] > q999
                large_value_count += large_mask.sum()
                df_processed.loc[large_mask, column] = q999
        
        print(f"  Replaced {infinite_count} infinite values")
        print(f"  Capped {large_value_count} extremely large values")
        
        return df_processed
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in flow data"""
        print("\n" + "="*50)
        print("🔧 HANDLING MISSING VALUES")
        print("="*50)
        
        df_processed = df.copy()
        missing_info = df_processed.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) == 0:
            print("✅ No missing values found")
            return df_processed
        
        print(f"Missing values found in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            percentage = (count / len(df_processed)) * 100
            print(f"  {col}: {count:,} missing ({percentage:.2f}%)")
        
        print("\nHandling missing values...")
        for column in missing_cols.index:
            if df_processed[column].dtype in ['object']:
                # For categorical: use mode
                mode_value = df_processed[column].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                df_processed[column].fillna(fill_value, inplace=True)
            else:
                # For numeric: use median
                median_value = df_processed[column].median()
                df_processed[column].fillna(median_value, inplace=True)
        
        print("✅ Missing values handled")
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for flow data"""
        print("\n" + "="*50)
        print("🔤 CATEGORICAL ENCODING")
        print("="*50)
        
        df_processed = df.copy()
        categorical_features = self.feature_info['categorical_features']
        
        if not categorical_features:
            print("✅ No categorical features to encode")
            return df_processed
        
        print(f"Encoding {len(categorical_features)} categorical features...")
        
        for column in categorical_features:
            unique_count = df_processed[column].nunique()
            
            if unique_count <= 15:  # Increased threshold for flow data
                # One-hot encoding for low cardinality
                one_hot = pd.get_dummies(df_processed[column], prefix=column)
                df_processed = pd.concat([df_processed, one_hot], axis=1)
                df_processed.drop(column, axis=1, inplace=True)
                self.encoders[column] = {'type': 'onehot', 'columns': one_hot.columns.tolist()}
                print(f"  One-hot encoded: {column} ({unique_count} categories)")
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                df_processed[column] = le.fit_transform(df_processed[column].astype(str))
                self.encoders[column] = {'type': 'label', 'encoder': le}
                print(f"  Label encoded: {column} ({unique_count} categories)")
        
        print("✅ Categorical encoding completed")
        return df_processed
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features using RobustScaler (handles outliers better)"""
        print("\n" + "="*50)
        print("⚖️ FEATURE SCALING")
        print("="*50)
        
        df_processed = df.copy()
        
        if 'attack_type' in df_processed.columns:
            target = df_processed['attack_type'].copy()
            features = df_processed.drop('attack_type', axis=1)
        else:
            target = None
            features = df_processed.copy()
        
        # Remove non-feature columns
        non_feature_cols = ['flow_id']
        for col in non_feature_cols:
            if col in features.columns:
                features = features.drop(col, axis=1)
        
        # Use RobustScaler instead of StandardScaler (better for data with outliers)
        scaler = RobustScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        self.scalers['main'] = scaler
        
        # Add target back if exists
        if target is not None:
            scaled_features['attack_type'] = target.values
        
        print("✅ Feature scaling completed")
        return scaled_features
    
    def remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove low variance features"""
        print("\n" + "="*50)
        print("🧹 REMOVING LOW VARIANCE FEATURES")
        print("="*50)
        
        df_processed = df.copy()
        
        if 'attack_type' in df_processed.columns:
            target = df_processed['attack_type'].copy()
            features = df_processed.drop('attack_type', axis=1)
        else:
            target = None
            features = df_processed.copy()
        
        original_feature_count = features.shape[1]
        
        # Remove zero/low variance features
        selector = VarianceThreshold(threshold=0.01)
        features_filtered = selector.fit_transform(features)
        
        # Get remaining feature names
        remaining_features = features.columns[selector.get_support()].tolist()
        features_df = pd.DataFrame(features_filtered, columns=remaining_features, index=features.index)
        
        # Add target back if exists
        if target is not None:
            features_df['attack_type'] = target.values
        
        print(f"✅ Feature filtering completed:")
        print(f"  Original features: {original_feature_count}")
        print(f"  Final features: {len(features_df.columns) - (1 if target is not None else 0)}")
        
        return features_df
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Create train-test split for flow data"""
        print("\n" + "="*50)
        print("📊 CREATING TRAIN-TEST SPLIT")
        print("="*50)
        
        if 'attack_type' not in df.columns:
            raise ValueError("Target variable 'attack_type' not found")
        
        # Separate features and target
        X = df.drop('attack_type', axis=1)
        y = df['attack_type']
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        # Show class distribution in splits
        print(f"\nTraining class distribution:")
        train_dist = y_train.value_counts()
        for cls, count in train_dist.items():
            print(f"  {cls}: {count:,}")
        
        print(f"\nTest class distribution:")
        test_dist = y_test.value_counts()
        for cls, count in test_dist.items():
            print(f"  {cls}: {count:,}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data and components"""
        print("\n" + "="*50)
        print("💾 SAVING PROCESSED DATA")
        print("="*50)
        
        # Save datasets
        X_train.to_csv(self.processed_path / "X_train_flow.csv", index=False)
        X_test.to_csv(self.processed_path / "X_test_flow.csv", index=False)
        y_train.to_csv(self.processed_path / "y_train_flow.csv", index=False)
        y_test.to_csv(self.processed_path / "y_test_flow.csv", index=False)
        
        # Save preprocessing components
        joblib.dump(self.scalers['main'], self.processed_path / "flow_scaler.joblib")
        joblib.dump(self.encoders, self.processed_path / "flow_encoders.joblib")
        joblib.dump(self.feature_info, self.processed_path / "flow_feature_info.joblib")
        
        print("✅ All files saved successfully")
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("="*50)
        print("🤖 FLOW DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Step 1: Load and analyze
        df = self.load_and_analyze_data()
        
        # Step 2: Handle infinite values FIRST
        df = self.handle_infinite_values(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Scale features
        df = self.scale_features(df)
        
        # Step 6: Remove low variance features
        df = self.remove_low_variance_features(df)
        
        # Step 7: Create train-test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(df)
        
        # Step 8: Save everything
        self.save_processed_data(X_train, X_test, y_train, y_test)
        
        print("\n" + "="*50)
        print("🎉 FLOW PREPROCESSING COMPLETED!")
        print("="*50)

# Main execution
if __name__ == "__main__":
    preprocessor = FlowDataPreprocessor()
    preprocessor.run_preprocessing()