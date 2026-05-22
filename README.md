# Building Energy Meter Reading Prediction

## Project Overview

This project aims to predict building energy meter readings using machine learning models. The prediction is based on a combination of weather conditions, building characteristics, and temporal features. This is a regression task that uses advanced feature engineering techniques and gradient boosting models to achieve accurate energy consumption forecasts.

## Dataset

The project utilizes four main datasets:

1. **train.csv** - Training data containing historical meter readings
2. **test.csv** - Test data for generating predictions
3. **building_metadata.csv** - Building characteristics including:
   - Building ID
   - Site ID
   - Primary use (building type)
   - Square footage
   - Year built
4. **weather_train.csv & weather_test.csv** - Weather data including:
   - Air temperature
   - Dew temperature
   - Relative humidity (calculated)
   - Sea level pressure
   - Wind speed
   - Precipitation depth

## Features

### Weather-Based Features

- **Relative Humidity**: Calculated from air and dew temperatures using Magnus formula
- **Heat Index**: Computed for hot conditions (≥ 27°C)
- **Wind Chill**: Calculated for cold conditions (≤ 10°C)
- **Feels-Like Temperature**: Composite measure combining heat index and wind chill
- **Precipitation Categories**: 
  - No Rain, Light Rain, Moderate Rain, Heavy Rain
  - Binary indicators for each category
- **Pressure Features**: Sea level pressure (winsorized)

### Temporal Features

- **Basic Time Features**:
  - Hour of day
  - Day of week
  - Month
  - Day of year
  - Weekend indicator

- **Cyclical Encoding** (sine/cosine encoding):
  - Hour cyclical encoding
  - Day of week cyclical encoding
  - Month cyclical encoding
  - Season encoding (Winter, Spring, Summer, Fall)

### Building Features

- Building ID and Site ID
- Primary use type (label encoded)
- Square footage (log-transformed)

## Data Preprocessing

### Cleaning Steps

1. **Missing Value Handling**:
   - Interpolation for precipitation data (6-hour gap limit)
   - Linear interpolation for temperature, humidity, and wind speed

2. **Outlier Treatment**:
   - Winsorization (0.5% on each end) for sea level pressure
   - Capping feels-like temperature between -10°C and 35°C

3. **Feature Engineering**:
   - Log transformation of meter readings (target variable)
   - Log transformation of building square footage
   - Label encoding for categorical variables (building type, precipitation category)

4. **Data Merging**:
   - Merged training data with building metadata
   - Merged with weather data on site ID and timestamp
   - Removed zero readings from training set

## Models

### LightGBM Regression Model

**Model Configuration**:
- Objective: Regression
- Boosting Type: GBDT (Gradient Boosting Decision Trees)
- Learning Rate: 0.05
- Number of Leaves: 51
- Subsample: 0.8
- Column Subsample: 0.8
- Number of Estimators: 1300
- Early Stopping: 50 rounds

**Performance Metrics**:
- Loss Function: RMSLE (Root Mean Squared Logarithmic Error)
- Evaluation: R² Score on training and validation sets
- Actual vs Predicted plots for model visualization

### Additional Models

- LSTM (Deep Learning approach) - implementation included

## Features Used in Model

```
'building_id', 'meter', 'site_id',
'type_enc', 'square_feet_log', 'relative_humidity', 'feels_like_capped',
'wind_speed', 'is_light_rain', 'is_moderate_rain',
'is_heavy_rain', 'hour', 'day_of_week', 'month', 'day_of_year',
'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin',
'month_cos', 'season'
```

## Visualizations

### Weather Analysis

#### Monthly Temperature Trends
![Monthly feels_like Trends](Graphs/output.png)
Comparison of feels-like and actual temperatures across months, showing clear seasonal patterns with peak temperatures in summer months (July-August) and lowest in winter months (December-January).

#### Daily Temperature Patterns Over the Year
![Feels-Like Temperature Trends Over the Year](Graphs/output2.png)
Scatter plot showing daily feels-like temperature variations across the year, with distinct seasonal cycles and day-to-day variability.

#### Temperature Distribution by Site
![Feels-Like Temperature Distribution by Site](Graphs/output4.png)
Box plots showing temperature variations across different sites, revealing site-specific climate characteristics and temperature ranges.

#### Seasonal Wind Speed Variations
![Wind Speed Variations by Season](Graphs/output7.png)
Seasonal wind speed patterns showing consistency across seasons with occasional outliers and typical wind speed ranges.

### Feature Distribution & Correlations

#### Feature Correlation Heatmap (Weather Data)
![Feature Correlation Heatmap](Graphs/output5.png)
Comprehensive correlation matrix showing relationships between all features:
- Strong correlation between site_id and building characteristics
- Moderate correlation between precipitation features (light, moderate, heavy rain)
- Seasonal patterns visible in cyclical encoding correlations

#### Feature Correlation Heatmap (Final Dataset)
![Feature Correlation Heatmap - Final Dataset](Graphs/output10.png)
Correlation analysis of merged dataset with building and weather features:
- Site_id shows strong correlation with building_id
- Wind speed moderately correlates with dew temperature and precipitation
- Most temporal features show weak correlation as expected

#### Feature Distribution Box Plots
![Feature Distribution Box Plots](Graphs/output9.png)
Distribution analysis of key features:
- Meter readings show skewed distribution (transformed to log scale)
- Square footage log shows clustered distribution with outliers
- Relative humidity spans wide range (0-100%)
- Feels-like temperature shows expected seasonal variation
- Wind speed concentrates in lower ranges with outliers

### Energy Consumption Patterns

#### Hourly Energy Usage Pattern
![Hourly Energy Usage Pattern](Graphs/output8.png)
Clear diurnal pattern in energy consumption:
- Minimum consumption during late night hours (2-4 AM)
- Steady increase through morning hours
- Peak consumption during mid-day (12-15 hours)
- Gradual decline in evening hours

#### Monthly Average Meter Reading
![Monthly Average Meter Reading](Graphs/output13.png)
Seasonal energy consumption trends:
- Peak consumption in summer months (July-August)
- Secondary peak in winter months (November-December)
- Lower consumption in spring and early fall
- Distinct pattern indicating climate-dependent usage

#### Yearly Energy Consumption Trends
![Yearly Energy Consumption Trends](Graphs/output7.png)
Daily energy consumption throughout the year showing:
- Clear seasonal patterns with summer peaks
- Day-to-day volatility influenced by weather and usage patterns
- Consistent baseline consumption levels

#### Energy Consumption by Meter Type
![Total Energy Consumption vs. Efficiency by Meter Type](Graphs/output11.png)
Meter type analysis showing:
- Electricity dominates total consumption
- Chilled water and steam usage varies by season
- Energy efficiency (per sq ft) varies significantly by meter type
- Hot water shows lowest overall consumption

#### Meter Type Distribution
![Distribution of Meter Types](Graphs/output12.png)
Data distribution showing:
- Electricity readings comprise ~60% of dataset
- Chilled water readings ~25%
- Steam readings ~15%
- Hot water readings ~5%

## Key Insights

### Monthly Trends
- Monthly average meter readings show distinct seasonal patterns
- Peak energy consumption visible during summer months (July-August) and winter months (November-December)
- Clear correlation between temperature extremes and energy demand

### Weather Relationships
- Strong correlation between temperature features (feels-like) and energy consumption
- Precipitation and humidity levels impact building energy usage
- Wind speed influences heating/cooling demands
- Site-specific weather patterns affect building energy profiles

### Temporal Patterns
- Hourly variations in energy consumption follow predictable patterns with peaks at midday
- Day-of-week effects evident in consumption patterns (weekdays vs weekends)
- Seasonal variations significantly affect overall consumption with summer/winter peaks
- Meter type shows different consumption patterns (electricity dominates)

## Project Structure

```
ML_Project/
├── Final_file.ipynb          # Main project notebook with complete pipeline
├── EDA*.ipynb                # Exploratory data analysis notebooks
├── README.md                 # This file
├── data.txt                  # Additional data/notes
├── Graphs/                   # Visualization outputs
└── Datast/                   # Data directory (train.csv, test.csv, etc.)
```

## Libraries Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, lightgbm
- **Deep Learning**: LSTM (Keras/TensorFlow)
- **Visualization**: matplotlib, seaborn
- **Statistics**: scipy

## Usage

To run the complete pipeline:

1. Ensure all required datasets are in the `Datast/` directory
2. Open `Final_file.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially to:
   - Load and preprocess data
   - Perform exploratory data analysis
   - Train the LightGBM model
   - Evaluate model performance
   - Generate predictions

## Model Output

The model generates:
- Training R² Score
- Validation R² Score
- Actual vs Predicted scatter plots
- Feature importance analysis (from LightGBM)

## Future Enhancements

1. Hyperparameter tuning for improved performance
2. Ensemble methods combining multiple model predictions
3. Time series cross-validation strategies
4. Feature selection optimization
5. LSTM model refinement and optimization
6. External data sources (holidays, special events)

## Notes

- The target variable (meter_reading) is log-transformed to handle skewed distributions
- Site 0, Meter 0 readings were calibrated with a factor of 0.293071
- All temporal features are cyclically encoded to capture seasonal patterns
- Early stopping is used to prevent overfitting during model training

---

**Author**: Group 12  
**Project Type**: Regression / Energy Consumption Prediction  
**Date**: 2024
