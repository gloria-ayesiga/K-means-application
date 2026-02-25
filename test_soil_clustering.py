# %%
import os
os.getcwd()
os.chdir("/")
os.getcwd()  #changing working directory
# %%
# test_soil_clustering.py
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Mock dataset mimicking the AfSIS wet chemistry + georef structure
mock_data = {
    'SSN': ['icr001', 'icr002', 'icr003', 'icr004', 'icr005'],
    'EC': [0.05, 0.20, 0.04, np.nan, 0.12],
    'ExAc': [0.3, 0.6, 1.2, 0.8, 0.45],
    'ExBas': [5.0, 15.0, 2.5, 4.0, 8.2],
    'M3 Al': [600, 800, 1200, 900, 750],
    'M3 B': [0.2, 1.5, 0.1, 0.3, 0.8],
    'M3 P': [10.0, 25.0, 5.0, 12.0, 18.0],
    'PH': [6.0, 7.5, 5.0, 5.8, 6.4],
    'Latitude': [0.31, 0.41, 0.21, np.nan, 0.35],
    'Longitude': [32.51, 32.71, 32.41, 32.60, 32.55],
}

mock_georef = {
    'SSN': ['icr001', 'icr002', 'icr003', 'icr005'],
    'Latitude': [0.310, 0.410, 0.210, 0.350],
    'Longitude': [32.510, 32.710, 32.410, 32.550],
}


class TestSoilClustering(unittest.TestCase):
    """Unit tests for AfSIS K-Means soil clustering application"""

    def setUp(self):
        """Create fresh mock data for each test"""
        self.df = pd.DataFrame(mock_data)
        self.georef = pd.DataFrame(mock_georef)
        self.feature_cols = ['EC', 'ExAc', 'ExBas', 'M3 Al', 'M3 B', 'M3 P', 'PH']

    def test_dataframe_has_required_columns(self):
        """Check that key columns exist"""
        required = ['SSN', 'PH', 'M3 P'] + self.feature_cols[:3]
        for col in required:
            with self.subTest(column=col):
                self.assertIn(col, self.df.columns)

    def test_missing_values_are_handled(self):
        """Test simple mean imputation"""
        x = self.df[self.feature_cols].copy()
        self.assertTrue(x.isna().any().any())  # has NaN
        x_filled = x.fillna(x.mean())
        self.assertFalse(x_filled.isna().any().any())
        self.assertEqual(x_filled.shape, x.shape)

    def test_feature_scaling_produces_standardized_data(self):
        """Scaled data should have ~mean=0, std=1"""
        x = self.df[self.feature_cols].fillna(self.df[self.feature_cols].mean())
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        self.assertAlmostEqual(x_scaled.mean(), 0.0, places=6)
        self.assertAlmostEqual(x_scaled.std(), 1.0, places=5)

    def test_kmeans_runs_and_assigns_labels(self):
        """K-Means fits and produces integer labels"""
        x_raw = self.df[self.feature_cols]
        x_prep = x_raw.fillna(x_raw.mean())
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_prep)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(x_scaled)
        labels = kmeans.labels_

        self.assertEqual(len(labels), x_scaled.shape[0])
        self.assertTrue(all(isinstance(l, (int, np.integer)) for l in labels))
        self.assertEqual(kmeans.n_clusters, 3)

    def test_merge_with_georeferences(self):
        """Merge keeps chemistry rows and adds lat/lon where available"""
        merged = pd.merge(self.df, self.georef, on='SSN', how='left', suffixes=('', '_geo'))
        self.assertEqual(merged.shape[0], self.df.shape[0])
        self.assertGreater(merged['Latitude_geo'].notna().sum(), 0)
        self.assertIn('Latitude_geo', merged.columns)  # or 'Latitude' after rename

    def test_cluster_column_is_added_correctly(self):
        """After clustering, 'Cluster' exists and is integer"""
        x_raw = self.df[self.feature_cols]
        x_prep = x_raw.fillna(x_raw.mean())
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_prep)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df['Cluster'] = kmeans.fit_predict(x_scaled)

        self.assertIn('Cluster', self.df.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(self.df['Cluster']))
        self.assertEqual(self.df['Cluster'].isna().sum(), 0)

    def test_empty_or_invalid_data_raises_appropriate_error(self):
        """Graceful handling of bad input"""
        df_empty = pd.DataFrame(columns=self.feature_cols)
        with self.assertRaises(ValueError):
            scaler = StandardScaler()
            scaler.fit(df_empty)  # should raise on empty


if __name__ == '__main__':
    unittest.main(verbosity=2)
# %%

# %%
