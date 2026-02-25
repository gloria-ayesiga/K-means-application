# AfSIS Soil Chemistry K-Means Clustering Application

A Python-based tool that applies K-Means clustering to soil chemistry data from the Africa Soil Information Service (AfSIS) to identify meaningful soil groups across Sub-Saharan Africa. The application helps reveal patterns in soil fertility, acidity, nutrient availability, and salinity—insights useful for agricultural planning, fertilizer recommendations, and sustainable farming practices.

## Features

- Loads and preprocesses wet chemistry soil data (pH, Mehlich-3 nutrients, EC, exchangeable bases, etc.)
- Performs feature scaling and K-Means clustering (with elbow + silhouette methods to choose optimal k)
- Interprets clusters into practical soil profiles (e.g., fertile high-base, acidic low-fertility, saline/high-Na)
- Joins georeferenced locations and visualizes clusters on an interactive Folium map
- Includes unit tests for core functionality (data prep, scaling, clustering, merging)

## Dataset

The project uses the **Africa Soil Information Service (AfSIS) Soil Chemistry** dataset, publicly available on AWS Open Data Registry:  
`s3://afsis` (no credentials required)

Focus: 2009–2013 wet chemistry measurements (e.g., `Wet_Chemistry/CROPNUTS/Wet_Chemistry_CROPNUTS.csv`)  
Georeferences: `2009-2013/Georeferences/georeferences.csv`

License: ODC Open Database License (ODbL) v1.0 – attribution to AfSIS required.

## Requirements

- Python 3.8+
- Libraries:
  pip install pandas numpy scikit-learn matplotlib seaborn folium 
