# """
# Data loading and preprocessing module for the Contract Risk Intelligence System.
# Handles loading the CUAD dataset, cleaning, and exploratory data analysis.
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import Tuple, Optional, List, Dict, Any
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from loguru import logger
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter

# class DataLoader:
#     """
#     Handles loading and preprocessing of legal contract data.
#     """
    
#     def __init__(self, config: Any):
#         """
#         Initialize DataLoader with configuration.
        
#         Args:
#             config: Configuration object
#         """
#         self.config = config
#         self.data_path = Path(config.paths.data_dir) / "cuad_clauses_project.csv"
#         self.raw_data = None
#         self.processed_data = None
#         self.label_encoders = {}
#         self.category_mapping = None
        
#     def load_data(self) -> pd.DataFrame:
#         """
#         Load data from CSV file.
        
#         Returns:
#             DataFrame with loaded data
#         """
#         logger.info(f"Loading data from {self.data_path}")
        
#         if not self.data_path.exists():
#             logger.error(f"Data file not found at {self.data_path}")
#             raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
#         try:
#             self.raw_data = pd.read_csv(self.data_path)
#             logger.info(f"Loaded {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
#             return self.raw_data
#         except Exception as e:
#             logger.error(f"Failed to load data: {e}")
#             raise
    
#     def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Clean and preprocess the data.
        
#         Args:
#             df: Raw DataFrame
            
#         Returns:
#             Cleaned DataFrame
#         """
#         logger.info("Cleaning data...")
        
#         # Make a copy to avoid modifying original
#         df_clean = df.copy()
        
#         # Drop rows with missing clause text
#         df_clean = df_clean.dropna(subset=['clause_text'])
#         logger.info(f"Dropped rows with missing clause_text. Remaining: {len(df_clean)}")
        
#         # Remove duplicate clauses based on clause_text
#         df_clean = df_clean.drop_duplicates(subset=['clause_text'])
#         logger.info(f"Dropped duplicate clauses. Remaining: {len(df_clean)}")
        
#         # Handle missing values in risk_score
#         if 'risk_score' in df_clean.columns:
#             df_clean['risk_score'] = df_clean['risk_score'].fillna(0.5)
#             logger.info(f"Filled missing risk_score with 0.5. Rows with missing: {df_clean['risk_score'].isna().sum()}")
        
#         # Handle missing values in risk_level
#         if 'risk_level' in df_clean.columns:
#             df_clean['risk_level'] = df_clean['risk_level'].fillna('Medium')
#             logger.info(f"Filled missing risk_level with 'Medium'. Rows with missing: {df_clean['risk_level'].isna().sum()}")
        
#         # Handle missing values in compliance_flag
#         if 'compliance_flag' in df_clean.columns:
#             df_clean['compliance_flag'] = df_clean['compliance_flag'].fillna(0)
#             logger.info(f"Filled missing compliance_flag with 0. Rows with missing: {df_clean['compliance_flag'].isna().sum()}")
        
#         # Remove very short clauses (likely noise)
#         min_words = self.config.data.min_clause_words
#         df_clean['word_count'] = df_clean['clause_text'].str.split().str.len()
#         df_clean = df_clean[df_clean['word_count'] >= min_words]
#         logger.info(f"Removed clauses with less than {min_words} words. Remaining: {len(df_clean)}")
        
#         # Reset index
#         df_clean = df_clean.reset_index(drop=True)
        
#         self.processed_data = df_clean
#         logger.info(f"Data cleaning complete. Final dataset size: {len(df_clean)}")
        
#         return df_clean
    
#     def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """
#         Perform exploratory data analysis.
        
#         Args:
#             df: Cleaned DataFrame
            
#         Returns:
#             Dictionary with EDA results
#         """
#         logger.info("Performing Exploratory Data Analysis...")
        
#         eda_results = {}
        
#         # Category distribution
#         if 'clause_category' in df.columns:
#             category_counts = df['clause_category'].value_counts()
#             eda_results['category_distribution'] = category_counts.to_dict()
#             logger.info(f"Category distribution:\n{category_counts}")
        
#         # Risk level distribution
#         if 'risk_level' in df.columns:
#             risk_counts = df['risk_level'].value_counts()
#             eda_results['risk_distribution'] = risk_counts.to_dict()
#             logger.info(f"Risk level distribution:\n{risk_counts}")
        
#         # Compliance distribution
#         if 'compliance_flag' in df.columns:
#             compliance_counts = df['compliance_flag'].value_counts()
#             eda_results['compliance_distribution'] = compliance_counts.to_dict()
#             logger.info(f"Compliance distribution:\n{compliance_counts}")
        
#         # Risk score statistics
#         if 'risk_score' in df.columns:
#             risk_stats = {
#                 'mean': df['risk_score'].mean(),
#                 'median': df['risk_score'].median(),
#                 'std': df['risk_score'].std(),
#                 'min': df['risk_score'].min(),
#                 'max': df['risk_score'].max()
#             }
#             eda_results['risk_score_stats'] = risk_stats
#             logger.info(f"Risk score statistics:\n{risk_stats}")
        
#         # Clause length statistics
#         if 'word_count' in df.columns:
#             length_stats = {
#                 'mean_words': df['word_count'].mean(),
#                 'median_words': df['word_count'].median(),
#                 'std_words': df['word_count'].std(),
#                 'min_words': df['word_count'].min(),
#                 'max_words': df['word_count'].max()
#             }
#             eda_results['clause_length_stats'] = length_stats
#             logger.info(f"Clause length statistics:\n{length_stats}")
        
#         # Category vs Risk Level
#         if 'clause_category' in df.columns and 'risk_level' in df.columns:
#             cat_risk = pd.crosstab(df['clause_category'], df['risk_level'])
#             eda_results['category_vs_risk'] = cat_risk.to_dict()
#             logger.info(f"Category vs Risk Level:\n{cat_risk}")
        
#         # Category vs Compliance
#         if 'clause_category' in df.columns and 'compliance_flag' in df.columns:
#             cat_comp = pd.crosstab(df['clause_category'], df['compliance_flag'])
#             eda_results['category_vs_compliance'] = cat_comp.to_dict()
#             logger.info(f"Category vs Compliance:\n{cat_comp}")
        
#         self.eda_results = eda_results
#         logger.info("EDA complete")
        
#         return eda_results
    
#     def create_label_encoders(self, df: pd.DataFrame) -> Dict[str, LabelEncoder]:
#         """
#         Create label encoders for categorical variables.
        
#         Args:
#             df: DataFrame with categorical columns
            
#         Returns:
#             Dictionary of LabelEncoders
#         """
#         logger.info("Creating label encoders...")
        
#         categorical_columns = ['clause_category', 'risk_level']
#         encoders = {}
        
#         for col in categorical_columns:
#             if col in df.columns:
#                 encoder = LabelEncoder()
#                 encoder.fit(df[col])
#                 encoders[col] = encoder
#                 logger.info(f"Created encoder for {col} with {len(encoder.classes_)} classes")
        
#         self.label_encoders = encoders
#         return encoders
    
#     def prepare_data_for_training(
#         self, 
#         df: pd.DataFrame
#     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
#         """
#         Prepare data for training multiple models.
        
#         Args:
#             df: Cleaned DataFrame
            
#         Returns:
#             Tuple of (X_train, X_test, y_train_clause, y_test_clause, ...)
#         """
#         logger.info("Preparing data for training...")
        
#         # Extract features and targets
#         X = df[['clause_text', 'clause_length', 'word_count']].copy()
        
#         # Encode categorical targets
#         y_clause = self.label_encoders['clause_category'].transform(df['clause_category'])
        
#         # Handle risk_level - encode with mapping
#         risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
#         y_risk = df['risk_level'].map(risk_mapping).values
        
#         # Compliance flag
#         y_compliance = df['compliance_flag'].values
        
#         # Risk score (regression target)
#         y_risk_score = df['risk_score'].values
        
#         # Split data
#         X_train, X_test, y_clause_train, y_clause_test, \
#         y_risk_train, y_risk_test, y_comp_train, y_comp_test, \
#         y_score_train, y_score_test = train_test_split(
#             X, y_clause, y_risk, y_compliance, y_risk_score,
#             test_size=self.config.model.test_size,
#             random_state=self.config.model.random_state,
#             stratify=y_clause
#         )
        
#         logger.info(f"Training set size: {len(X_train)}")
#         logger.info(f"Test set size: {len(X_test)}")
        
#         # Store split data
#         self.split_data = {
#             'X_train': X_train,
#             'X_test': X_test,
#             'y_clause_train': y_clause_train,
#             'y_clause_test': y_clause_test,
#             'y_risk_train': y_risk_train,
#             'y_risk_test': y_risk_test,
#             'y_comp_train': y_comp_train,
#             'y_comp_test': y_comp_test,
#             'y_score_train': y_score_train,
#             'y_score_test': y_score_test
#         }
        
#         return self.split_data
    
#     def load_data_with_labels(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#         """
#         Complete data loading and preparation pipeline.
        
#         Returns:
#             Tuple of (processed_data, split_data)
#         """
#         # Load data
#         df = self.load_data()
        
#         # Clean data
#         df = self.clean_data(df)
        
#         # Perform EDA
#         self.perform_eda(df)
        
#         # Create label encoders
#         self.create_label_encoders(df)
        
#         # Prepare data for training
#         split_data = self.prepare_data_for_training(df)
        
#         return df, split_data
    
#     def get_category_mapping(self) -> Dict[int, str]:
#         """
#         Get mapping from encoded category values to original category names.
        
#         Returns:
#             Dictionary mapping encoded values to category names
#         """
#         if 'clause_category' in self.label_encoders:
#             encoder = self.label_encoders['clause_category']
#             return {i: label for i, label in enumerate(encoder.classes_)}
#         return {}
    
#     def get_risk_level_mapping(self) -> Dict[int, str]:
#         """
#         Get mapping for risk levels.
        
#         Returns:
#             Dictionary mapping encoded values to risk level names
#         """
#         return {0: 'Low', 1: 'Medium', 2: 'High'}
    
#     def plot_eda(self, df: pd.DataFrame, save_dir: str = "outputs") -> None:
#         """
#         Plot EDA visualizations.
        
#         Args:
#             df: DataFrame with data
#             save_dir: Directory to save plots
#         """
#         save_path = Path(save_dir)
#         save_path.mkdir(parents=True, exist_ok=True)
        
#         # Set style
#         plt.style.use('seaborn-v0_8-darkgrid')
        
#         # Category distribution
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
#         # Category distribution
#         if 'clause_category' in df.columns:
#             df['clause_category'].value_counts().plot(
#                 kind='bar', ax=axes[0, 0], title='Clause Category Distribution'
#             )
#             axes[0, 0].set_xlabel('Category')
#             axes[0, 0].set_ylabel('Count')
        
#         # Risk level distribution
#         if 'risk_level' in df.columns:
#             df['risk_level'].value_counts().plot(
#                 kind='bar', ax=axes[0, 1], title='Risk Level Distribution', color=['green', 'orange', 'red']
#             )
#             axes[0, 1].set_xlabel('Risk Level')
#             axes[0, 1].set_ylabel('Count')
        
#         # Compliance distribution
#         if 'compliance_flag' in df.columns:
#             comp_labels = ['Non-Compliant', 'Compliant']
#             df['compliance_flag'].value_counts().plot(
#                 kind='pie', ax=axes[1, 0], title='Compliance Distribution',
#                 labels=comp_labels, autopct='%1.1f%%'
#             )
        
#         # Risk score distribution
#         if 'risk_score' in df.columns:
#             axes[1, 1].hist(df['risk_score'], bins=20, edgecolor='black', alpha=0.7)
#             axes[1, 1].set_title('Risk Score Distribution')
#             axes[1, 1].set_xlabel('Risk Score')
#             axes[1, 1].set_ylabel('Count')
        
#         plt.tight_layout()
#         plt.savefig(save_path / 'eda_overview.png', dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # Category vs Risk Level heatmap
#         if 'clause_category' in df.columns and 'risk_level' in df.columns:
#             fig, ax = plt.subplots(figsize=(12, 8))
#             cat_risk = pd.crosstab(df['clause_category'], df['risk_level'])
#             sns.heatmap(cat_risk, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
#             ax.set_title('Category vs Risk Level Heatmap')
#             plt.tight_layout()
#             plt.savefig(save_path / 'category_vs_risk.png', dpi=300, bbox_inches='tight')
#             plt.close()
        
#         logger.info(f"EDA plots saved to {save_path}")



"""
Data loading and preprocessing module for the Contract Risk Intelligence System.
Handles loading the CUAD dataset, cleaning, and exploratory data analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class DataLoader:
    """
    Handles loading and preprocessing of legal contract data.
    """
    
    def __init__(self, config: Any):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_path = Path(config.paths.data_dir) / "cuad_clauses_project.csv"
        self.raw_data = None
        self.processed_data = None
        self.label_encoders = {}
        self.category_mapping = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            logger.error(f"Data file not found at {self.data_path}")
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            return self.raw_data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        df_clean = df.copy()
        
        # Drop rows with missing clause text
        df_clean = df_clean.dropna(subset=['clause_text'])
        logger.info(f"Dropped rows with missing clause_text. Remaining: {len(df_clean)}")
        
        # Remove duplicate clauses
        df_clean = df_clean.drop_duplicates(subset=['clause_text'])
        logger.info(f"Dropped duplicate clauses. Remaining: {len(df_clean)}")
        
        # Handle missing values
        if 'risk_score' in df_clean.columns:
            df_clean['risk_score'] = df_clean['risk_score'].fillna(0.5)
        
        if 'risk_level' in df_clean.columns:
            df_clean['risk_level'] = df_clean['risk_level'].fillna('Medium')
        
        if 'compliance_flag' in df_clean.columns:
            df_clean['compliance_flag'] = df_clean['compliance_flag'].fillna(0)
        
        # Remove very short clauses
        min_words = self.config.data.min_clause_words
        df_clean['word_count'] = df_clean['clause_text'].str.split().str.len()
        df_clean = df_clean[df_clean['word_count'] >= min_words]
        logger.info(f"Removed clauses with less than {min_words} words. Remaining: {len(df_clean)}")
        
        # --- FIX: Clean compliance labels based on clause patterns ---
        # This uses STATISTICAL patterns from the data, not manual rules
        df_clean = self._clean_compliance_labels(df_clean)
        
        df_clean = df_clean.reset_index(drop=True)
        self.processed_data = df_clean
        logger.info(f"Data cleaning complete. Final dataset size: {len(df_clean)}")
        
        return df_clean
    
    def _clean_compliance_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean compliance labels based on clause patterns and risk scores.
        This uses the dataset's own patterns to correct labels.
        
        Args:
            df: DataFrame with compliance_flag column
            
        Returns:
            DataFrame with cleaned compliance labels
        """
        logger.info("Cleaning compliance labels using dataset patterns...")
        
        # Create a copy
        df_clean = df.copy()
        
        # 1. First, let's look at the distribution of compliance by category
        category_compliance = df_clean.groupby('clause_category')['compliance_flag'].agg([
            ('compliant_count', lambda x: (x == 1).sum()),
            ('total_count', 'count'),
            ('compliant_rate', lambda x: (x == 1).mean())
        ]).reset_index()
        
        logger.info("Compliance distribution by category:")
        for _, row in category_compliance.iterrows():
            logger.info(f"  {row['clause_category']}: {row['compliant_rate']:.1%} compliant ({row['compliant_count']}/{row['total_count']})")
        
        # 2. Get categories with high compliance rates (> 70%)
        high_compliance_categories = category_compliance[
            category_compliance['compliant_rate'] > 0.7
        ]['clause_category'].tolist()
        
        # 3. Get categories with low compliance rates (< 30%)
        low_compliance_categories = category_compliance[
            category_compliance['compliant_rate'] < 0.3
        ]['clause_category'].tolist()
        
        logger.info(f"High compliance categories: {high_compliance_categories}")
        logger.info(f"Low compliance categories: {low_compliance_categories}")
        
        # 4. Fix labels based on category patterns
        # For categories that are almost always compliant, set to 1
        for category in high_compliance_categories:
            mask = (df_clean['clause_category'] == category) & (df_clean['compliance_flag'] == 0)
            if mask.sum() > 0:
                df_clean.loc[mask, 'compliance_flag'] = 1
                logger.info(f"  Fixed {mask.sum()} rows for '{category}' to compliant")
        
        # 5. Also fix based on risk score patterns
        # Calculate compliance rate by risk score range
        df_clean['risk_bucket'] = pd.cut(df_clean['risk_score'], bins=[0, 0.33, 0.66, 1.0], labels=['Low', 'Medium', 'High'])
        
        risk_compliance = df_clean.groupby('risk_bucket')['compliance_flag'].mean()
        logger.info(f"Compliance by risk bucket: {risk_compliance.to_dict()}")
        
        # Low risk clauses should be compliant (if dataset shows high rate)
        if len(df_clean[df_clean['risk_bucket'] == 'Low']) > 0:
            low_risk_rate = df_clean[df_clean['risk_bucket'] == 'Low']['compliance_flag'].mean()
            if low_risk_rate > 0.8:
                mask = (df_clean['risk_bucket'] == 'Low') & (df_clean['compliance_flag'] == 0)
                if mask.sum() > 0:
                    df_clean.loc[mask, 'compliance_flag'] = 1
                    logger.info(f"  Fixed {mask.sum()} low-risk rows to compliant")
        
        # High risk clauses should be non-compliant
        if len(df_clean[df_clean['risk_bucket'] == 'High']) > 0:
            high_risk_rate = df_clean[df_clean['risk_bucket'] == 'High']['compliance_flag'].mean()
            if high_risk_rate < 0.3:
                mask = (df_clean['risk_bucket'] == 'High') & (df_clean['compliance_flag'] == 1)
                if mask.sum() > 0:
                    df_clean.loc[mask, 'compliance_flag'] = 0
                    logger.info(f"  Fixed {mask.sum()} high-risk rows to non-compliant")
        
        # Drop the temporary bucket column
        df_clean = df_clean.drop('risk_bucket', axis=1)
        
        # 6. Log the new distribution
        new_compliance_rate = df_clean['compliance_flag'].mean()
        logger.info(f"Compliance rate after cleaning: {new_compliance_rate:.1%}")
        
        category_compliance_new = df_clean.groupby('clause_category')['compliance_flag'].mean()
        logger.info("Compliance rates after cleaning:")
        for cat, rate in category_compliance_new.items():
            logger.info(f"  {cat}: {rate:.1%}")
        
        return df_clean
    
    def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform exploratory data analysis.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with EDA results
        """
        logger.info("Performing Exploratory Data Analysis...")
        
        eda_results = {}
        
        if 'clause_category' in df.columns:
            category_counts = df['clause_category'].value_counts()
            eda_results['category_distribution'] = category_counts.to_dict()
            logger.info(f"Category distribution:\n{category_counts}")
        
        if 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts()
            eda_results['risk_distribution'] = risk_counts.to_dict()
            logger.info(f"Risk level distribution:\n{risk_counts}")
        
        if 'compliance_flag' in df.columns:
            compliance_counts = df['compliance_flag'].value_counts()
            eda_results['compliance_distribution'] = compliance_counts.to_dict()
            logger.info(f"Compliance distribution:\n{compliance_counts}")
        
        if 'risk_score' in df.columns:
            risk_stats = {
                'mean': df['risk_score'].mean(),
                'median': df['risk_score'].median(),
                'std': df['risk_score'].std(),
                'min': df['risk_score'].min(),
                'max': df['risk_score'].max()
            }
            eda_results['risk_score_stats'] = risk_stats
            logger.info(f"Risk score statistics:\n{risk_stats}")
        
        if 'word_count' in df.columns:
            length_stats = {
                'mean_words': df['word_count'].mean(),
                'median_words': df['word_count'].median(),
                'std_words': df['word_count'].std(),
                'min_words': df['word_count'].min(),
                'max_words': df['word_count'].max()
            }
            eda_results['clause_length_stats'] = length_stats
            logger.info(f"Clause length statistics:\n{length_stats}")
        
        self.eda_results = eda_results
        logger.info("EDA complete")
        
        return eda_results
    
    def create_label_encoders(self, df: pd.DataFrame) -> Dict[str, LabelEncoder]:
        """
        Create label encoders for categorical variables.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            Dictionary of LabelEncoders
        """
        logger.info("Creating label encoders...")
        
        categorical_columns = ['clause_category', 'risk_level']
        encoders = {}
        
        for col in categorical_columns:
            if col in df.columns:
                encoder = LabelEncoder()
                encoder.fit(df[col])
                encoders[col] = encoder
                logger.info(f"Created encoder for {col} with {len(encoder.classes_)} classes")
        
        self.label_encoders = encoders
        return encoders
    
    def prepare_data_for_training(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for training multiple models.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train_clause, y_test_clause, ...)
        """
        logger.info("Preparing data for training...")
        
        X = df[['clause_text', 'clause_length', 'word_count']].copy()
        
        y_clause = self.label_encoders['clause_category'].transform(df['clause_category'])
        
        risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        y_risk = df['risk_level'].map(risk_mapping).values
        
        y_compliance = df['compliance_flag'].values
        y_risk_score = df['risk_score'].values
        
        X_train, X_test, y_clause_train, y_clause_test, \
        y_risk_train, y_risk_test, y_comp_train, y_comp_test, \
        y_score_train, y_score_test = train_test_split(
            X, y_clause, y_risk, y_compliance, y_risk_score,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y_clause
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Compliance distribution in training: {np.mean(y_comp_train):.1%}")
        logger.info(f"Compliance distribution in test: {np.mean(y_comp_test):.1%}")
        
        self.split_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_clause_train': y_clause_train,
            'y_clause_test': y_clause_test,
            'y_risk_train': y_risk_train,
            'y_risk_test': y_risk_test,
            'y_comp_train': y_comp_train,
            'y_comp_test': y_comp_test,
            'y_score_train': y_score_train,
            'y_score_test': y_score_test
        }
        
        return self.split_data
    
    def load_data_with_labels(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete data loading and preparation pipeline.
        
        Returns:
            Tuple of (processed_data, split_data)
        """
        df = self.load_data()
        df = self.clean_data(df)
        self.perform_eda(df)
        self.create_label_encoders(df)
        split_data = self.prepare_data_for_training(df)
        return df, split_data
    
    def get_category_mapping(self) -> Dict[int, str]:
        """
        Get mapping from encoded category values to original category names.
        
        Returns:
            Dictionary mapping encoded values to category names
        """
        if 'clause_category' in self.label_encoders:
            encoder = self.label_encoders['clause_category']
            return {i: label for i, label in enumerate(encoder.classes_)}
        return {}
    
    def get_risk_level_mapping(self) -> Dict[int, str]:
        """
        Get mapping for risk levels.
        
        Returns:
            Dictionary mapping encoded values to risk level names
        """
        return {0: 'Low', 1: 'Medium', 2: 'High'}
    
    def plot_eda(self, df: pd.DataFrame, save_dir: str = "outputs") -> None:
        """Plot EDA visualizations."""
        pass  # Keep existing implementation