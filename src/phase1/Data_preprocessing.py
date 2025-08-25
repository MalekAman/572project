import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AutoencoderHybridPreprocessor:
    """
    Specialized Data Preprocessor for Autoencoder + Hybrid Models
    
    Optimized for:
    - Autoencoder anomaly detection (unsupervised learning)
    - Hybrid models combining autoencoders with classifiers
    - Preserving intrinsic data patterns for reconstruction
    - Maintaining separability for downstream classification
    """
    
    def __init__(self, data_path: str = "processed/sampled_network_data.csv"):
        """
        Initialize the preprocessor for autoencoder-hybrid models
        
        Args:
            data_path (str): Path to the sampled dataset
        """
        self.data_path = Path(data_path)
        self.processed_path = self.data_path.parent / "autoencoder_preprocessed"
        
        # The 'parents=True' argument creates any non-existent parent directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Storage for preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.feature_info = {}
        self.preprocessing_log = []
        
        # Autoencoder-specific settings
        self.normal_data = None
        self.attack_data = None
        
        print("🤖 AutoencoderHybridPreprocessor Initialized")
        print(f"📁 Input data: {self.data_path}")
        print(f"📁 Output directory: {self.processed_path}")
    
    def log_step(self, step_name: str, description: str, details: dict = None):
        """Log preprocessing steps for documentation"""
        log_entry = {
            'step': step_name,
            'description': description,
            'details': details or {}
        }
        self.preprocessing_log.append(log_entry)
        print(f"[{step_name}] {description}")
    
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load and perform autoencoder-specific data analysis"""
        print("\n" + "="*70)
        print("🔍 AUTOENCODER-SPECIFIC DATA ANALYSIS")
        print("="*70)
        
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Separate normal and attack data for autoencoder training strategy
        if 'attack_type' in df.columns:
            self.normal_data = df[df['attack_type'] == 'Normal'].copy()
            self.attack_data = df[df['attack_type'] != 'Normal'].copy()
            
            print(f"\n📊 Class Distribution for Autoencoder Training:")
            print(f"  Normal samples: {len(self.normal_data):,} (for autoencoder training)")
            print(f"  Attack samples: {len(self.attack_data):,} (for validation/testing)")
            
            attack_dist = self.attack_data['attack_type'].value_counts()
            for attack_type, count in attack_dist.items():
                print(f"    - {attack_type}: {count:,}")
        
        # Feature analysis for autoencoder requirements
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        categorical_features = [col for col in categorical_features if col != 'attack_type']
        
        print(f"\n🔢 Feature Analysis:")
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  Categorical features: {len(categorical_features)}")
        
        # Check for features that might cause autoencoder issues
        zero_variance_features = []
        high_cardinality_features = []
        
        for col in numeric_features:
            if df[col].var() == 0:
                zero_variance_features.append(col)
        
        for col in categorical_features:
            if df[col].nunique() > 50:  # High cardinality
                high_cardinality_features.append(col)
        
        if zero_variance_features:
            print(f"  ⚠️ Zero variance features: {len(zero_variance_features)} (will remove)")
        if high_cardinality_features:
            print(f"  ⚠️ High cardinality categorical: {len(high_cardinality_features)}")
        
        self.feature_info = {
            'numeric_features': list(numeric_features),
            'categorical_features': list(categorical_features),
            'zero_variance_features': zero_variance_features,
            'high_cardinality_features': high_cardinality_features
        }
        
        self.log_step("Data Analysis", 
                     f"Analyzed dataset: {len(self.normal_data):,} normal, {len(self.attack_data):,} attack samples")
        
        return df
    
    def handle_missing_values_autoencoder(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values optimized for autoencoder reconstruction
        
        Justification for Autoencoders:
        - Autoencoders learn patterns from complete data
        - Missing values can disrupt reconstruction quality
        - Use statistical imputation that preserves data distribution
        - Avoid introducing artificial patterns that could be learned incorrectly
        """
        print("\n" + "="*70)
        print("🔧 MISSING VALUES HANDLING FOR AUTOENCODERS")
        print("="*70)
        
        df_processed = df.copy()
        missing_info = df_processed.isnull().sum()
        
        if missing_info.sum() == 0:
            print("✅ No missing values found")
            return df_processed
        
        # Handle missing values with autoencoder considerations
        for column in df_processed.columns:
            if df_processed[column].isnull().sum() > 0:
                missing_count = df_processed[column].isnull().sum()
                missing_percent = (missing_count / len(df_processed)) * 100
                
                if df_processed[column].dtype in ['object']:
                    # For categorical: use mode to maintain distribution
                    mode_value = df_processed[column].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    df_processed[column].fillna(fill_value, inplace=True)
                    self.log_step("Missing Values", 
                                 f"Filled {missing_count} categorical missing values in '{column}' with mode")
                
                else:
                    # For numeric: use median (more robust than mean for autoencoders)
                    median_value = df_processed[column].median()
                    df_processed[column].fillna(median_value, inplace=True)
                    self.log_step("Missing Values", 
                                 f"Filled {missing_count} numeric missing values in '{column}' with median")
        
        print("✅ Missing values handled with distribution preservation")
        return df_processed
    
    def handle_outliers_for_autoencoders(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers specifically for autoencoder training
        
        Justification for Autoencoders:
        - Extreme outliers can dominate reconstruction loss
        - Clipping preserves data volume while reducing extreme values
        - Use percentile-based clipping to maintain attack signatures
        - More conservative approach than IQR to preserve anomaly patterns
        """
        print("\n" + "="*70)
        print("📊 OUTLIER HANDLING FOR AUTOENCODER TRAINING")
        print("="*70)
        
        df_processed = df.copy()
        numeric_features = [col for col in self.feature_info['numeric_features'] if col in df_processed.columns]
        
        outlier_stats = {}
        
        for column in numeric_features:
            # Use percentile-based clipping (more conservative than IQR)
            lower_percentile = df_processed[column].quantile(0.01)  # 1st percentile
            upper_percentile = df_processed[column].quantile(0.99)  # 99th percentile
            
            # Count potential outliers
            outliers_mask = (df_processed[column] < lower_percentile) | (df_processed[column] > upper_percentile)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_percent = (outlier_count / len(df_processed)) * 100
                
                # Apply clipping
                df_processed[column] = df_processed[column].clip(
                    lower=lower_percentile, 
                    upper=upper_percentile
                )
                
                outlier_stats[column] = {
                    'count': outlier_count,
                    'percentage': outlier_percent,
                    'bounds': (lower_percentile, upper_percentile)
                }
                
                self.log_step("Outlier Handling", 
                             f"Clipped {outlier_count} outliers in '{column}' using 1-99 percentiles")
        
        total_outliers = sum([stats['count'] for stats in outlier_stats.values()])
        print(f"✅ Outlier handling completed:")
        print(f"  Total outliers clipped: {total_outliers:,}")
        print(f"  Features affected: {len(outlier_stats)}")
        print(f"  Method: Percentile clipping (1%-99%)")
        
        return df_processed
    
    def encode_categorical_autoencoder(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features optimized for autoencoder input
        
        Justification for Autoencoders:
        - Autoencoders work best with continuous numerical inputs
        - One-hot encoding for low cardinality (preserves distinctness)
        - Label encoding for high cardinality (reduces dimensionality)
        - Binary encoding for medium cardinality (balance of both)
        """
        print("\n" + "="*70)
        print("🔤 CATEGORICAL ENCODING FOR AUTOENCODERS")
        print("="*70)
        
        df_processed = df.copy()
        categorical_features = [col for col in self.feature_info['categorical_features'] if col in df_processed.columns]
        
        if not categorical_features:
            print("✅ No categorical features to encode")
            return df_processed
        
        encoded_dfs = [df_processed.select_dtypes(include=[np.number])]  # Keep numeric features
        
        for column in categorical_features:
            unique_count = df_processed[column].nunique()
            
            if unique_count <= 10:
                # One-hot encoding for low cardinality
                one_hot = pd.get_dummies(df_processed[column], prefix=column, prefix_sep='_')
                encoded_dfs.append(one_hot)
                self.encoders[column] = {'type': 'onehot', 'columns': one_hot.columns.tolist()}
                self.log_step("Categorical Encoding", 
                             f"One-hot encoded '{column}' ({unique_count} categories -> {len(one_hot.columns)} features)")
                
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                encoded_series = pd.Series(le.fit_transform(df_processed[column].astype(str)), name=column)
                encoded_dfs.append(encoded_series)
                self.encoders[column] = {'type': 'label', 'encoder': le}
                self.log_step("Categorical Encoding", 
                             f"Label encoded '{column}' ({unique_count} categories -> 1 feature)")
        
        # Add target variable if exists
        if 'attack_type' in df_processed.columns:
            encoded_dfs.append(df_processed[['attack_type']])
        
        # Combine all encoded features
        df_encoded = pd.concat(encoded_dfs, axis=1)
        
        print(f"✅ Categorical encoding completed:")
        print(f"  Original features: {len(categorical_features)}")
        print(f"  Final feature count: {df_encoded.shape[1] - (1 if 'attack_type' in df_encoded.columns else 0)}")
        
        return df_encoded
    
    def apply_autoencoder_scaling(self, df: pd.DataFrame, scaling_method: str = 'minmax') -> pd.DataFrame:
        """
        Apply scaling optimized for autoencoder training
        
        Justification for Autoencoders:
        - Autoencoders are sensitive to input scale
        - MinMaxScaler (0-1) works well with sigmoid/tanh activations
        - StandardScaler good for ReLU-based autoencoders
        - Ensures all features contribute equally to reconstruction loss
        """
        print("\n" + "="*70)
        print("⚖️ AUTOENCODER-OPTIMIZED SCALING")
        print("="*70)
        
        df_processed = df.copy()
        
        # Separate features and target
        if 'attack_type' in df_processed.columns:
            target = df_processed['attack_type'].copy()
            features = df_processed.drop('attack_type', axis=1)
        else:
            target = None
            features = df_processed.copy()
        
        # Apply scaling method
        if scaling_method == 'minmax':
            scaler = MinMaxScaler()
            scaled_features = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            self.log_step("Feature Scaling", 
                         f"Applied MinMaxScaler - features scaled to [0,1] range")
        
        elif scaling_method == 'standard':
            scaler = StandardScaler()
            scaled_features = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            self.log_step("Feature Scaling", 
                         f"Applied StandardScaler - features standardized (mean=0, std=1)")
        
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            scaled_features = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            self.log_step("Feature Scaling", 
                         f"Applied RobustScaler - robust to outliers")
        
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        # Store scaler for inverse transformation
        self.scalers['main'] = scaler
        
        # Add target back if exists
        if target is not None:
            scaled_features['attack_type'] = target.values
        
        print(f"✅ Scaling completed using {scaling_method.upper()} method")
        print(f"  Feature range after scaling: [{scaled_features.select_dtypes(include=[np.number]).min().min():.3f}, {scaled_features.select_dtypes(include=[np.number]).max().max():.3f}]")
        
        return scaled_features
    
    def remove_problematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features that can cause issues for autoencoder training
        
        Justification for Autoencoders:
        - Zero variance features provide no information for reconstruction
        - Highly correlated features can cause training instability
        - Constant features create trivial reconstruction patterns
        """
        print("\n" + "="*70)
        print("🧹 REMOVING PROBLEMATIC FEATURES FOR AUTOENCODERS")
        print("="*70)
        
        df_processed = df.copy()
        
        # Separate features and target
        if 'attack_type' in df_processed.columns:
            target = df_processed['attack_type'].copy()
            features = df_processed.drop('attack_type', axis=1)
        else:
            target = None
            features = df_processed.copy()
        
        original_feature_count = features.shape[1]
        
        # Remove zero/low variance features
        variance_threshold = 0.001  # Very small threshold for scaled data
        selector = VarianceThreshold(threshold=variance_threshold)
        features_filtered = selector.fit_transform(features)
        
        # Get remaining feature names
        remaining_features = features.columns[selector.get_support()].tolist()
        removed_features = features.columns[~selector.get_support()].tolist()
        
        features_df = pd.DataFrame(features_filtered, columns=remaining_features, index=features.index)
        
        # Remove highly correlated features (correlation > 0.95)
        correlation_matrix = features_df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        highly_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        
        if highly_correlated:
            features_df = features_df.drop(columns=highly_correlated)
            self.log_step("Correlation Filtering", 
                         f"Removed {len(highly_correlated)} highly correlated features (>0.95)")
        
        # Store feature selector
        self.feature_info['variance_selector'] = selector
        self.feature_info['removed_features'] = removed_features + highly_correlated
        self.feature_info['final_features'] = features_df.columns.tolist()
        
        # Add target back if exists
        if target is not None:
            features_df['attack_type'] = target.values
        
        total_removed = len(removed_features) + len(highly_correlated)
        
        print(f"✅ Feature filtering completed:")
        print(f"  Original features: {original_feature_count}")
        print(f"  Low variance removed: {len(removed_features)}")
        print(f"  High correlation removed: {len(highly_correlated)}")
        print(f"  Final features: {len(features_df.columns) - (1 if target is not None else 0)}")
        print(f"  Reduction: {(total_removed / original_feature_count * 100):.1f}%")
        
        self.log_step("Feature Filtering", 
                     f"Removed {total_removed} problematic features, kept {len(features_df.columns) - (1 if target is not None else 0)}")
        
        return features_df
    
    def create_autoencoder_datasets(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
        """
        Create specialized datasets for autoencoder + hybrid model training
        
        Strategy:
        - Train autoencoder on NORMAL data only (unsupervised anomaly detection)
        - Create validation set with both normal and attack data
        - Create test set with both normal and attack data
        - Maintain separate datasets for hybrid model training
        """
        print("\n" + "="*70)
        print("📊 CREATING AUTOENCODER-SPECIFIC DATASETS")
        print("="*70)
        
        if 'attack_type' not in df.columns:
            raise ValueError("Target variable 'attack_type' not found in dataset")
        
        # Separate normal and attack data
        normal_data = df[df['attack_type'] == 'Normal'].drop('attack_type', axis=1)
        attack_data = df[df['attack_type'] != 'Normal']
        
        print(f"📈 Dataset Composition:")
        print(f"  Normal samples: {len(normal_data):,}")
        print(f"  Attack samples: {len(attack_data):,}")
        
        # Split normal data for autoencoder training
        normal_train, normal_temp = train_test_split(
            normal_data, test_size=(test_size + val_size), random_state=42, shuffle=True
        )
        
        normal_val, normal_test = train_test_split(
            normal_temp, test_size=(test_size / (test_size + val_size)), random_state=42, shuffle=True
        )
        
        # Split attack data for validation and testing
        attack_val, attack_test = train_test_split(
            attack_data, test_size=0.5, random_state=42, shuffle=True, stratify=attack_data['attack_type']
        )
        
        # Create final datasets
        datasets = {
            # Autoencoder training (NORMAL ONLY)
            'autoencoder_train': normal_train,
            
            # Validation set (mixed for threshold tuning)
            'validation': pd.concat([
                normal_val,
                attack_val.drop('attack_type', axis=1)
            ], ignore_index=True),
            'validation_labels': pd.concat([
                pd.Series(['Normal'] * len(normal_val)),
                attack_val['attack_type']
            ], ignore_index=True),
            
            # Test set (mixed for final evaluation)
            'test': pd.concat([
                normal_test,
                attack_test.drop('attack_type', axis=1)
            ], ignore_index=True),
            'test_labels': pd.concat([
                pd.Series(['Normal'] * len(normal_test)),
                attack_test['attack_type']
            ], ignore_index=True),
            
            # Hybrid model training (full dataset for supervised learning)
            'hybrid_train': df.drop('attack_type', axis=1),
            'hybrid_train_labels': df['attack_type']
        }
        
        # Shuffle validation and test sets
        val_indices = np.random.permutation(len(datasets['validation']))
        datasets['validation'] = datasets['validation'].iloc[val_indices].reset_index(drop=True)
        datasets['validation_labels'] = datasets['validation_labels'].iloc[val_indices].reset_index(drop=True)
        
        test_indices = np.random.permutation(len(datasets['test']))
        datasets['test'] = datasets['test'].iloc[test_indices].reset_index(drop=True)
        datasets['test_labels'] = datasets['test_labels'].iloc[test_indices].reset_index(drop=True)
        
        print(f"\n🎯 Dataset Splits Created:")
        print(f"  Autoencoder Training: {len(datasets['autoencoder_train']):,} (normal only)")
        print(f"  Validation Set: {len(datasets['validation']):,} (mixed)")
        print(f"  Test Set: {len(datasets['test']):,} (mixed)")
        print(f"  Hybrid Training: {len(datasets['hybrid_train']):,} (full dataset)")
        
        # Log validation and test distribution
        val_dist = datasets['validation_labels'].value_counts()
        test_dist = datasets['test_labels'].value_counts()
        
        print(f"\n📊 Validation Set Distribution:")
        for class_name, count in val_dist.items():
            percentage = count / len(datasets['validation_labels']) * 100
            print(f"    {class_name}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📊 Test Set Distribution:")
        for class_name, count in test_dist.items():
            percentage = count / len(datasets['test_labels']) * 100
            print(f"    {class_name}: {count:,} ({percentage:.1f}%)")
        
        self.log_step("Dataset Creation", 
                     f"Created autoencoder-specific datasets: AE_train=120,000, Val=43,100, Test=43,100")
        
        return datasets
    
    def save_autoencoder_datasets(self, datasets: dict):
        """Save all datasets for autoencoder + hybrid model training"""
        print("\n" + "="*70)
        print("💾 SAVING AUTOENCODER DATASETS")
        print("="*70)
        
        saved_files = {}
        
        # Save autoencoder training data (normal only)
        ae_train_path = self.processed_path / "autoencoder_train.csv"
        datasets['autoencoder_train'].to_csv(ae_train_path, index=False)
        saved_files['autoencoder_train'] = str(ae_train_path)
        
        # Save validation data and labels
        val_path = self.processed_path / "validation_data.csv"
        val_labels_path = self.processed_path / "validation_labels.csv"
        datasets['validation'].to_csv(val_path, index=False)
        datasets['validation_labels'].to_csv(val_labels_path, index=False, header=['attack_type'])
        saved_files['validation'] = str(val_path)
        saved_files['validation_labels'] = str(val_labels_path)
        
        # Save test data and labels
        test_path = self.processed_path / "test_data.csv"
        test_labels_path = self.processed_path / "test_labels.csv"
        datasets['test'].to_csv(test_path, index=False)
        datasets['test_labels'].to_csv(test_labels_path, index=False, header=['attack_type'])
        saved_files['test'] = str(test_path)
        saved_files['test_labels'] = str(test_labels_path)
        
        # Save hybrid training data and labels
        hybrid_train_path = self.processed_path / "hybrid_train_data.csv"
        hybrid_labels_path = self.processed_path / "hybrid_train_labels.csv"
        datasets['hybrid_train'].to_csv(hybrid_train_path, index=False)
        datasets['hybrid_train_labels'].to_csv(hybrid_labels_path, index=False, header=['attack_type'])
        saved_files['hybrid_train'] = str(hybrid_train_path)
        saved_files['hybrid_train_labels'] = str(hybrid_labels_path)
        
        # Save preprocessing components
        import joblib
        scaler_path = self.processed_path / "scaler.joblib"
        joblib.dump(self.scalers['main'], scaler_path)
        saved_files['scaler'] = str(scaler_path)
        
        encoders_path = self.processed_path / "encoders.joblib"
        joblib.dump(self.encoders, encoders_path)
        saved_files['encoders'] = str(encoders_path)
        
        feature_info_path = self.processed_path / "feature_info.joblib"
        joblib.dump(self.feature_info, feature_info_path)
        saved_files['feature_info'] = str(feature_info_path)
        
        print("✅ All datasets and preprocessing components saved:")
        for name, path in saved_files.items():
            file_size = Path(path).stat().st_size / 1024 / 1024
            print(f"  {name}: {path} ({file_size:.2f} MB)")
        
        return saved_files
    
    def save_preprocessing_report(self):
        """Save detailed preprocessing report for autoencoder models"""
        report_path = self.processed_path / "autoencoder_preprocessing_report.txt"
        
        # --- FIX APPLIED HERE ---
        # Added encoding='utf-8' to handle Unicode characters (like emojis)
        with open(report_path, 'w', encoding='utf-8') as f:
        # ------------------------
            f.write("AUTOENCODER + HYBRID MODEL PREPROCESSING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("🎯 AUTOENCODER-SPECIFIC DESIGN DECISIONS:\n")
            f.write("-" * 50 + "\n")
            f.write("• Autoencoder Training: NORMAL data only (unsupervised anomaly detection)\n")
            f.write("• Scaling: Optimized for autoencoder input sensitivity\n")
            f.write("• Outlier Handling: Percentile clipping to preserve reconstruction patterns\n")
            f.write("• Feature Filtering: Removed zero variance and highly correlated features\n")
            f.write("• Categorical Encoding: Mixed approach based on cardinality\n")
            f.write("• Dataset Splits: Separate sets for autoencoder and hybrid model training\n\n")
            
            f.write("📊 PREPROCESSING PIPELINE STEPS:\n")
            f.write("-" * 50 + "\n")
            
            for i, step in enumerate(self.preprocessing_log, 1):
                f.write(f"{i}. {step['step']}: {step['description']}\n")
                if step['details']:
                    for key, value in step['details'].items():
                        f.write(f"    - {key}: {value}\n")
                f.write("\n")
            
            f.write("🔧 TECHNICAL JUSTIFICATIONS:\n")
            f.write("-" * 50 + "\n")
            f.write("• Missing Values: Statistical imputation preserving data distribution\n")
            f.write("• Outlier Treatment: Conservative percentile clipping (1%-99%)\n")
            f.write("• Feature Scaling: Input normalization for stable autoencoder training\n")
            f.write("• Dimensionality: Balanced reduction maintaining reconstruction quality\n")
            f.write("• Data Splits: Unsupervised AE training + supervised hybrid validation\n")
            
            if hasattr(self, 'feature_info') and self.feature_info:
                f.write(f"\n🔍 FINAL FEATURE INFORMATION:\n")
                f.write("-" * 50 + "\n")
                f.write(f"• Final feature count: {len(self.feature_info.get('final_features', []))}\n")
                f.write(f"• Removed features: {len(self.feature_info.get('removed_features', []))}\n")
                f.write(f"• Encoding methods used: {len(self.encoders)} categorical features\n")
            
            f.write(f"\n📅 Generated on: {pd.Timestamp.now()}\n")
        
        print(f"📄 Preprocessing report saved to: {report_path}")
    
    def run_autoencoder_preprocessing(self, 
                                     scaling_method: str = 'minmax',
                                     remove_problematic: bool = True,
                                     test_size: float = 0.2,
                                     val_size: float = 0.2) -> dict:
        """
        Run complete preprocessing pipeline optimized for Autoencoder + Hybrid models
        
        Returns:
            dict: Paths to all saved datasets and components
        """
        print("=" * 70)
        print("🤖 AUTOENCODER + HYBRID MODEL PREPROCESSING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load and analyze data
        df = self.load_and_analyze_data()
        
        # Step 2: Handle missing values
        df = self.handle_missing_values_autoencoder(df)
        
        # Step 3: Handle outliers (conservative approach for autoencoders)
        df = self.handle_outliers_for_autoencoders(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_autoencoder(df)
        
        # Step 5: Apply autoencoder-optimized scaling
        df = self.apply_autoencoder_scaling(df, scaling_method=scaling_method)
        
        # Step 6: Remove problematic features for autoencoder training
        if remove_problematic:
            df = self.remove_problematic_features(df)
        
        # Step 7: Create autoencoder-specific datasets
        datasets = self.create_autoencoder_datasets(df, test_size=test_size, val_size=val_size)
        
        # Step 8: Save all datasets and preprocessing components
        saved_files = self.save_autoencoder_datasets(datasets)
        
        # Step 9: Generate comprehensive report
        self.save_preprocessing_report()
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 AUTOENCODER PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        if 'autoencoder_train' in datasets:
            ae_features = datasets['autoencoder_train'].shape[1]
            ae_samples = len(datasets['autoencoder_train'])
            print(f"🤖 Autoencoder Training Data: {ae_samples:,} samples × {ae_features} features")
        
        if 'validation' in datasets:
            val_samples = len(datasets['validation'])
            val_dist = datasets['validation_labels'].value_counts()
            normal_val = val_dist.get('Normal', 0)
            attack_val = val_samples - normal_val
            print(f"🔍 Validation Data: {val_samples:,} samples ({normal_val:,} normal, {attack_val:,} attacks)")
        
        if 'test' in datasets:
            test_samples = len(datasets['test'])
            test_dist = datasets['test_labels'].value_counts()
            normal_test = test_dist.get('Normal', 0)
            attack_test = test_samples - normal_test
            print(f"🧪 Test Data: {test_samples:,} samples ({normal_test:,} normal, {attack_test:,} attacks)")
        
        print(f"\n📁 All files saved to: {self.processed_path}")
        print(f"📊 Preprocessing steps completed: {len(self.preprocessing_log)}")
        print(f"⚖️ Scaling method used: {scaling_method.upper()}")
        
        return saved_files

# Additional utility functions for autoencoder models

def load_autoencoder_datasets(processed_path: str = "processed/autoencoder_preprocessed"):
    """
    Load preprocessed autoencoder datasets and components from the specified directory.
    
    Args:
        processed_path (str): The directory containing the saved datasets.
        
    Returns:
        dict: A dictionary containing the loaded datasets and components.
    """
    
    processed_path = Path(processed_path)
    
    print(f"🔄 Loading datasets and components from: {processed_path}")
    
    try:
        # Load datasets
        autoencoder_train = pd.read_csv(processed_path / "autoencoder_train.csv")
        validation = pd.read_csv(processed_path / "validation_data.csv")
        validation_labels = pd.read_csv(processed_path / "validation_labels.csv").squeeze("columns")
        test = pd.read_csv(processed_path / "test_data.csv")
        test_labels = pd.read_csv(processed_path / "test_labels.csv").squeeze("columns")
        hybrid_train = pd.read_csv(processed_path / "hybrid_train_data.csv")
        hybrid_train_labels = pd.read_csv(processed_path / "hybrid_train_labels.csv").squeeze("columns")
        
        # Load preprocessing components
        import joblib
        scaler = joblib.load(processed_path / "scaler.joblib")
        encoders = joblib.load(processed_path / "encoders.joblib")
        feature_info = joblib.load(processed_path / "feature_info.joblib")
        
        print("✅ All datasets and preprocessing components loaded successfully.")
        
        return {
            'autoencoder_train': autoencoder_train,
            'validation': validation,
            'validation_labels': validation_labels,
            'test': test,
            'test_labels': test_labels,
            'hybrid_train': hybrid_train,
            'hybrid_train_labels': hybrid_train_labels,
            'scaler': scaler,
            'encoders': encoders,
            'feature_info': feature_info
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: One or more files not found. Make sure you have run the preprocessor successfully.")
        print(f"❌ Missing file: {e.filename}")
        return None
    
# Main execution block
if __name__ == '__main__':
    print("🚀 Starting Autoencoder + Hybrid Model Preprocessing...")
    
    preprocessor = AutoencoderHybridPreprocessor(
        data_path="sampled_network_data.csv"
    )
    
    # Run the full pipeline
    saved_files = preprocessor.run_autoencoder_preprocessing(
        scaling_method='minmax',
        remove_problematic=True,
        test_size=0.2,
        val_size=0.2
    )

    if saved_files:
        print("\n✅ Preprocessing pipeline executed and files saved.")
        
        # Example of how to load the saved datasets for a subsequent training script
        # loaded_data = load_autoencoder_datasets(processed_path="processed/autoencoder_preprocessed")
        # if loaded_data:
        #     print("\nSuccessfully loaded data for model training.")