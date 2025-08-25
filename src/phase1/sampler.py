import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from typing import Dict, List

class NetworkDatasetSampler:
    """
    Simple Network Traffic Dataset Sampler
    
    Loads existing network traffic CSV files and samples:
    - 200,000 benign traffic samples
    - Exactly 6,200 attack samples distributed randomly across attack types
    """
    
    def __init__(self, data_path: str = ".", random_seed: int = 42):
        """
        Initialize the dataset sampler
        
        Args:
            data_path (str): Path to the directory containing CSV files (default: current directory)
            random_seed (int): Seed for reproducible random sampling
        """
        self.data_path = Path(data_path)
        self.processed_path = self.data_path / "processed"
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Ensure processed directory exists
        try:
            self.processed_path.mkdir(exist_ok=True)
        except OSError as e:
            print(f"Error creating processed directory: {e}")
            print(f"Trying to create at: {self.processed_path.absolute()}")
            raise
        
        # Mapping of file names to attack types
        self.file_attack_mapping = {
            'BenignTraffic.csv': 'Normal',
            'BenignTraffic1.csv': 'Normal', 
            'BenignTraffic2.csv': 'Normal',
            'BenignTraffic3.csv': 'Normal',
            'DDoS-HTTP_Flood.csv': 'DDoS',
            'DictionaryBruteForce.csv': 'BruteForce',
            'DNS_Spoofing.csv': 'Spoofing',
            'DoS-HTTP_Flood.csv': 'DoS',
            'DoS-HTTP_Flood1.csv': 'DoS',
            'XSS.csv': 'XSS'
        }
        
        self.attack_types = ['DDoS', 'BruteForce', 'Spoofing', 'DoS', 'XSS']
        
        print(f"Data path: {self.data_path}")
        print(f"Processed path: {self.processed_path}")
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the data directory"""
        datasets = {}
        
        print("Loading datasets...")
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
        
        return datasets
    
    def sample_data(self, datasets: Dict[str, pd.DataFrame], 
                   total_benign: int = 200000, 
                   total_attacks: int = 6200) -> pd.DataFrame:
        """
        Sample data from loaded datasets
        
        Args:
            datasets (Dict): Dictionary of loaded datasets
            total_benign (int): Number of benign samples
            total_attacks (int): Number of attack samples (exactly 6200)
            
        Returns:
            pd.DataFrame: Sampled dataset
        """
        print(f"\nSampling: {total_benign:,} benign + {total_attacks:,} attack samples")
        
        # Separate benign and attack datasets
        benign_dfs = []
        attack_dfs = []
        
        for file_name, df in datasets.items():
            if self.file_attack_mapping[file_name] == 'Normal':
                benign_dfs.append(df)
            else:
                attack_dfs.append(df)
        
        # Combine benign data
        if benign_dfs:
            all_benign = pd.concat(benign_dfs, ignore_index=True)
            print(f"Total benign samples available: {len(all_benign):,}")
            
            # Sample benign data
            if len(all_benign) >= total_benign:
                sampled_benign = all_benign.sample(n=total_benign, random_state=self.random_seed)
            else:
                print(f"Warning: Only {len(all_benign):,} benign samples available")
                sampled_benign = all_benign
        else:
            sampled_benign = pd.DataFrame()
        
        # Combine attack data
        if attack_dfs:
            all_attacks = pd.concat(attack_dfs, ignore_index=True)
            print(f"Total attack samples available: {len(all_attacks):,}")
            
            # Show attack distribution
            attack_counts = all_attacks['attack_type'].value_counts()
            print("Available attack types:")
            for attack_type, count in attack_counts.items():
                print(f"  {attack_type}: {count:,}")
            
            # Randomly distribute 6200 samples across attack types
            available_types = attack_counts.index.tolist()
            sampled_attacks_list = []
            
            # Create random distribution
            if len(available_types) > 0:
                # Generate random proportions
                random_props = np.random.dirichlet(np.ones(len(available_types)))
                target_counts = np.round(random_props * total_attacks).astype(int)
                
                # Adjust to ensure exact total
                diff = total_attacks - target_counts.sum()
                target_counts[0] += diff
                
                print(f"\nRandom attack distribution for {total_attacks} samples:")
                
                for i, attack_type in enumerate(available_types):
                    target_count = target_counts[i]
                    available_count = attack_counts[attack_type]
                    
                    # Sample what we can
                    actual_count = min(target_count, available_count)
                    
                    if actual_count > 0:
                        attack_subset = all_attacks[all_attacks['attack_type'] == attack_type]
                        sampled_subset = attack_subset.sample(n=actual_count, random_state=self.random_seed + i)
                        sampled_attacks_list.append(sampled_subset)
                        print(f"  {attack_type}: {actual_count:,} samples")
                
                # If we couldn't get exactly 6200, fill from largest available
                total_sampled = sum([len(df) for df in sampled_attacks_list])
                if total_sampled < total_attacks:
                    remaining = total_attacks - total_sampled
                    print(f"\nNeed {remaining} more samples - taking from largest available groups...")
                    
                    for attack_type in available_types:
                        if remaining <= 0:
                            break
                        
                        available_count = attack_counts[attack_type]
                        already_sampled = sum([len(df[df['attack_type'] == attack_type]) for df in sampled_attacks_list])
                        can_take_more = available_count - already_sampled
                        
                        if can_take_more > 0:
                            take_count = min(remaining, can_take_more)
                            attack_subset = all_attacks[all_attacks['attack_type'] == attack_type]
                            
                            # Get indices already sampled
                            sampled_indices = pd.concat(sampled_attacks_list).index if sampled_attacks_list else []
                            remaining_subset = attack_subset[~attack_subset.index.isin(sampled_indices)]
                            
                            if len(remaining_subset) >= take_count:
                                additional = remaining_subset.sample(n=take_count, random_state=self.random_seed + 100)
                                sampled_attacks_list.append(additional)
                                remaining -= take_count
                                print(f"  Additional {attack_type}: {take_count:,} samples")
            
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
            print(f"\nFinal sampled dataset: {len(final_dataset):,} samples")
            final_dist = final_dataset['attack_type'].value_counts()
            for attack_type, count in final_dist.items():
                percentage = count / len(final_dataset) * 100
                print(f"  {attack_type}: {count:,} ({percentage:.2f}%)")
            
            # Verify attack count
            attack_samples = len(final_dataset[final_dataset['attack_type'] != 'Normal'])
            print(f"\nVerification: {attack_samples:,} attack samples total")
        
        return final_dataset
    
    def save_sampled_dataset(self, df: pd.DataFrame, filename: str = "sampled_network_data.csv"):
        """Save the sampled dataset"""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return output_path
    
    def run_sampling(self, total_benign: int = 200000, total_attacks: int = 6200) -> str:
        """
        Run the complete sampling process
        
        Returns:
            str: Path to saved dataset
        """
        print("=" * 50)
        print("NETWORK DATASET SAMPLER")
        print("=" * 50)
        
        # Load datasets
        datasets = self.load_datasets()
        
        if not datasets:
            raise ValueError("No datasets found!")
        
        # Sample data
        sampled_df = self.sample_data(datasets, total_benign, total_attacks)
        
        if sampled_df.empty:
            raise ValueError("No data was sampled!")
        
        # Save dataset
        output_path = self.save_sampled_dataset(sampled_df)
        
        print("\n" + "=" * 50)
        print("SAMPLING COMPLETED!")
        print("=" * 50)
        
        return str(output_path)

# Usage Example
if __name__ == "__main__":
    # Create sampler (uses current directory since you're already in the right folder)
    sampler = NetworkDatasetSampler(
        data_path=".",  # Current directory
        random_seed=42
    )
    
    # Run sampling for exactly 6200 attack samples
    try:
        output_file = sampler.run_sampling(
            total_benign=200000,
            total_attacks=6200
        )
        print(f"\nSuccess! Sampled dataset saved to: {output_file}")
        
    except Exception as e:
        print(f"Sampling failed: {e}")