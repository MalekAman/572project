# flow_data_sampler.py
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from typing import Dict, List

class FlowDatasetSampler:
    """
    Flow-Level Dataset Sampler for Phase 3
    
    Loads flow-level CSV files and samples:
    - 200,000 benign traffic samples (flows)
    - Exactly 6,200 attack samples distributed randomly across attack types
    - Handles flow aggregation for flows split into 2-minute segments
    """
    
    def __init__(self, data_path: str = "D:/iot_project/flow_data", random_seed: int = 42):
        """
        Initialize the flow dataset sampler
        
        Args:
            data_path (str): Path to the directory containing flow CSV files
            random_seed (int): Seed for reproducible random sampling
        """
        self.data_path = Path(data_path)
        self.processed_path = self.data_path / "processed_flows"
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Ensure processed directory exists
        self.processed_path.mkdir(exist_ok=True)
        
        # Mapping of file names to attack types (UPDATED WITH CORRECT NAMES)
        self.file_attack_mapping = {
            'BenignTraffic.pcap_Flow.csv': 'Normal',
            'DDoS-HTTP_Flood-.pcap_Flow.csv': 'DDoS',  # Added extra hyphen
            'DictionaryBruteForce.pcap_Flow.csv': 'BruteForce',
            'DNS_Spoofing.pcap_Flow.csv': 'Spoofing',
            'DoS-HTTP_Flood.pcap_Flow.csv': 'DoS',
            'DoS-HTTP_Flood1.pcap_Flow.csv': 'DoS',
            'XSS.pcap_Flow.csv': 'XSS'
        }
        
        self.attack_types = ['DDoS', 'BruteForce', 'Spoofing', 'DoS', 'XSS']
        
        print(f"Flow Data path: {self.data_path}")
        print(f"Processed path: {self.processed_path}")
    
    def load_flow_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all flow CSV files from the data directory"""
        datasets = {}
        
        print("Loading flow datasets...")
        for file_path in self.data_path.glob("*.csv"):
            file_name = file_path.name
            if file_name in self.file_attack_mapping:
                try:
                    print(f"  Loading {file_name}...")
                    df = pd.read_csv(file_path)
                    df['attack_type'] = self.file_attack_mapping[file_name]
                    datasets[file_name] = df
                    print(f"    Shape: {df.shape}")
                    
                except Exception as e:
                    print(f"    Error loading {file_name}: {e}")
            else:
                print(f"  Skipping {file_name} (not in mapping)")
        
        return datasets
    
    def aggregate_flow_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate flow segments that were split due to 2-minute duration limit
        
        Args:
            df (pd.DataFrame): Raw flow data with potential segments
            
        Returns:
            pd.DataFrame: Aggregated flow data
        """
        # Check if flow ID columns exist
        flow_id_columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
        if not all(col in df.columns for col in flow_id_columns):
            print("Warning: Flow ID columns not found. Using original data.")
            return df
        
        # Create a flow identifier
        df['flow_id'] = df['src_ip'] + '-' + df['dst_ip'] + '-' + df['src_port'].astype(str) + '-' + df['dst_port'].astype(str)
        
        # Define which columns to average and which to take the first value
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove flow_id and attack_type from aggregation lists
        if 'flow_id' in numeric_cols:
            numeric_cols.remove('flow_id')
        if 'flow_id' in categorical_cols:
            categorical_cols.remove('flow_id')
        if 'attack_type' in categorical_cols:
            categorical_cols.remove('attack_type')
        
        # Group by flow_id and aggregate
        aggregation_dict = {}
        for col in numeric_cols:
            aggregation_dict[col] = 'mean'  # Average numeric features
        for col in categorical_cols:
            aggregation_dict[col] = 'first'  # Take first value for categorical features
        
        # Add attack_type - we'll use first since it should be consistent within a flow
        aggregation_dict['attack_type'] = 'first'
        
        aggregated_df = df.groupby('flow_id').agg(aggregation_dict).reset_index()
        
        print(f"  Aggregated {len(df)} segments into {len(aggregated_df)} unique flows")
        return aggregated_df
    
    def sample_flow_data(self, datasets: Dict[str, pd.DataFrame], 
                        total_benign: int = 200000, 
                        total_attacks: int = 6200) -> pd.DataFrame:
        """
        Sample data from loaded flow datasets with aggregation
        
        Args:
            datasets (Dict): Dictionary of loaded datasets
            total_benign (int): Number of benign flow samples
            total_attacks (int): Number of attack flow samples
            
        Returns:
            pd.DataFrame: Sampled and aggregated dataset
        """
        print(f"\nSampling: {total_benign:,} benign + {total_attacks:,} attack flows")
        
        # Separate benign and attack datasets
        benign_dfs = []
        attack_dfs = []
        
        for file_name, df in datasets.items():
            # First aggregate flow segments
            aggregated_df = self.aggregate_flow_segments(df)
            
            if self.file_attack_mapping[file_name] == 'Normal':
                benign_dfs.append(aggregated_df)
            else:
                attack_dfs.append(aggregated_df)
        
        # Combine benign data
        if benign_dfs:
            all_benign = pd.concat(benign_dfs, ignore_index=True)
            print(f"Total benign flows available: {len(all_benign):,}")
            
            # Sample benign data
            if len(all_benign) >= total_benign:
                sampled_benign = all_benign.sample(n=total_benign, random_state=self.random_seed)
            else:
                print(f"Warning: Only {len(all_benign):,} benign flows available")
                sampled_benign = all_benign
        else:
            sampled_benign = pd.DataFrame()
        
        # Combine attack data
        if attack_dfs:
            all_attacks = pd.concat(attack_dfs, ignore_index=True)
            print(f"Total attack flows available: {len(all_attacks):,}")
            
            # Show attack distribution
            attack_counts = all_attacks['attack_type'].value_counts()
            print("Available attack types:")
            for attack_type, count in attack_counts.items():
                print(f"  {attack_type}: {count:,}")
            
            # Randomly distribute attack samples
            available_types = attack_counts.index.tolist()
            sampled_attacks_list = []
            
            if len(available_types) > 0:
                # Generate random proportions
                random_props = np.random.dirichlet(np.ones(len(available_types)))
                target_counts = np.round(random_props * total_attacks).astype(int)
                
                # Adjust to ensure exact total
                diff = total_attacks - target_counts.sum()
                target_counts[0] += diff
                
                print(f"\nRandom attack distribution for {total_attacks} flows:")
                
                for i, attack_type in enumerate(available_types):
                    target_count = target_counts[i]
                    available_count = attack_counts[attack_type]
                    
                    # Sample what we can
                    actual_count = min(target_count, available_count)
                    
                    if actual_count > 0:
                        attack_subset = all_attacks[all_attacks['attack_type'] == attack_type]
                        sampled_subset = attack_subset.sample(n=actual_count, random_state=self.random_seed + i)
                        sampled_attacks_list.append(sampled_subset)
                        print(f"  {attack_type}: {actual_count:,} flows")
            
            # Combine all sampled attacks
            if sampled_attacks_list:
                sampled_attacks = pd.concat(sampled_attacks_list, ignore_index=True)
            else:
                sampled_attacks = pd.DataFrame()
        else:
            sampled_attacks = pd.DataFrame()
        
        # Combine final dataset
        if not sampled_benign.empty and not sampled_attacks.empty:
            final_dataset = pd.concat([sampled_benign, sampled_attacks], ignore_index=True)
        elif not sampled_benign.empty:
            final_dataset = sampled_benign
        elif not sampled_attacks.empty:
            final_dataset = sampled_attacks
        else:
            final_dataset = pd.DataFrame()
        
        # Shuffle the dataset
        if not final_dataset.empty:
            final_dataset = final_dataset.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Show final distribution
        if not final_dataset.empty:
            print(f"\nFinal sampled flow dataset: {len(final_dataset):,} flows")
            final_dist = final_dataset['attack_type'].value_counts()
            for attack_type, count in final_dist.items():
                percentage = count / len(final_dataset) * 100
                print(f"  {attack_type}: {count:,} ({percentage:.2f}%)")
            
            # Verify attack count
            attack_samples = len(final_dataset[final_dataset['attack_type'] != 'Normal'])
            print(f"\nVerification: {attack_samples:,} attack flows total")
        
        return final_dataset
    
    def save_sampled_dataset(self, df: pd.DataFrame, filename: str = "sampled_flow_data.csv"):
        """Save the sampled flow dataset"""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"\nFlow dataset saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return output_path
    
    def run_flow_sampling(self, total_benign: int = 200000, total_attacks: int = 6200) -> str:
        """
        Run the complete flow sampling process
        
        Returns:
            str: Path to saved dataset
        """
        print("=" * 50)
        print("FLOW DATASET SAMPLER - PHASE 3")
        print("=" * 50)
        
        # Load datasets
        datasets = self.load_flow_datasets()
        
        if not datasets:
            raise ValueError("No flow datasets found!")
        
        # Sample data
        sampled_df = self.sample_flow_data(datasets, total_benign, total_attacks)
        
        if sampled_df.empty:
            raise ValueError("No flow data was sampled!")
        
        # Save dataset
        output_path = self.save_sampled_dataset(sampled_df)
        
        print("\n" + "=" * 50)
        print("FLOW SAMPLING COMPLETED!")
        print("=" * 50)
        
        return str(output_path)

# Usage Example
if __name__ == "__main__":
    # Create sampler for flow data
    flow_sampler = FlowDatasetSampler(
        data_path="D:/iot_project/flow_data",  # Path to your flow data
        random_seed=42
    )
    
    # Run sampling for exactly 6200 attack flows
    try:
        output_file = flow_sampler.run_flow_sampling(
            total_benign=200000,
            total_attacks=6200
        )
        print(f"\nSuccess! Sampled flow dataset saved to: {output_file}")
        
    except Exception as e:
        print(f"Flow sampling failed: {e}")