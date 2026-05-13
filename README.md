# UK Road Accident Analysis Using Machine Learning and Spatial Analytics

## Overview

This project presents a machine learning and geospatial analytics framework for analysing road traffic accidents in the United Kingdom using the 2024 UK Department for Transport (DfT) Road Casualty Statistics dataset.

The system predicts accident severity levels, identifies accident hotspot zones, and forecasts future accident-prone regions using:

* Decision Tree Classification
* Decision Tree Regression
* K-Means Clustering
* Spatial Grid Aggregation
* Interactive Heatmaps
* Temporal Trend Analysis

  This project demonstrates the use of machine learning and spatial analytics for road safety analysis.
---

# Project Objectives

The project aims to:

* Analyse key factors influencing accident severity
* Predict accident severity levels (Fatal, Serious, Slight)
* Identify accident hotspot regions across the UK
* Forecast future high-risk accident zones
* Visualise accident patterns using interactive maps and charts
* Support data-driven road safety analysis

---

# Technologies Used

## Programming Language

* Python 3.9+

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Folium
* IPython

---

# Machine Learning Models

## 1. Decision Tree Classifier

Used for accident severity prediction.

### Input Features

* Weather Conditions
* Light Conditions
* Road Type
* Speed Limit
* Road Surface Conditions
* Hour
* Month
* Urban or Rural Area
* Number of Vehicles
* Number of Casualties

### Target Variable

* Fatal
* Serious
* Slight

### Performance

* Accuracy: 75%

---

## 2. K-Means Clustering

Used for identifying spatial accident hotspots across the UK.

### Features Used

* Latitude
* Longitude

### Output

* Geographic hotspot clusters
* Heatmap visualisation

---

## 3. Decision Tree Regressor

Used for forecasting future accident-prone zones.

### Forecast Features

* Previous month accident count
* Rolling average accident count
* Month number

### Performance

* MAE: 0.710
* RMSE: 1.195

---

# Dataset

## Source

UK Department for Transport (DfT) Road Casualty Statistics Dataset 2024.

## Dataset Features

* Collision Severity
* Weather Conditions
* Speed Limit
* Road Type
* Light Conditions
* Number of Vehicles
* Number of Casualties
* Latitude
* Longitude
* Date and Time

---

# Project Structure

```text
uk-road-accident-analysis/
│
├── data/
│   └── dft-road-casualty-statistics-collision-2024.csv
│
├── src/
│   └── road_traffic_analysis.py
│
├── outputs/
│   ├── severity_distribution.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── decision_tree_plot.png
│   ├── uk_kmeans_clusters.png
│   └── uk_heatmap.html
│
├── docs/
│   └── road traffic.docx
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/uk-road-accident-analysis.git
cd uk-road-accident-analysis
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

```bash
python src/road_traffic_analysis.py
```

---

# Generated Outputs

The project generates:

* Severity distribution charts
* Monthly accident trend analysis
* Weather vs severity analysis
* Confusion matrix
* Feature importance graph
* Decision tree visualisation
* Actual vs predicted charts
* UK hotspot cluster plots
* Interactive Folium heatmaps

---

# Key Findings

* Slight accidents represent the majority of cases
* Traffic-related features strongly influence accident severity
* Urban regions show higher accident concentration
* The forecasting model successfully identifies future high-risk zones
* Spatial hotspot regions align with high traffic density areas

---

# Future Improvements

Potential future enhancements include:

* Random Forest and XGBoost implementation
* Deep learning models (LSTM/CNN)
* Multi-year accident forecasting
* Real-time traffic integration
* GIS road network analysis
* Live dashboard deployment

---


# References

* UK Department for Transport Road Safety Open Data
* Scikit-learn Documentation
* Folium Documentation
* Accident Analysis & Prevention Journal

---

# License

This project is developed for academic and educational purposes.
