# Predicting Bus Arrival Time at Traffic Light Stop Lines using Machine Learning Methods

**Bachelor Thesis** **Author:** Elsayed Fared Ibrahim Khalil Ibrahim  
**Supervisors:** Prof. Dr. Stefanie Schmidtner, Prof. Dr. Bernd Martin Ebert  
**Institution:** Technische Hochschule Ingolstadt, Faculty of Electrical Engineering and Computer Science  
**Date:** May 2024

---

## 📖 Abstract

Accurate and efficient prediction of estimated arrival time (EAT) at traffic lights is a core factor for public transportation systems. This project investigates the application of machine learning techniques—specifically **Linear Regression**, **Multi-Layer Perceptron (MLP)**, and **Long Short-Term Memory (LSTM)** networks—to predict bus arrival times using Floating Car Data (FCD).

The study focuses on bus line 70 in Ingolstadt, Germany, utilizing data collected over seven months. A key finding of this research is that removing dwell times (waiting times at bus stops) significantly increases prediction accuracy, with LSTM and MLP models outperforming traditional mean-based methods.

## 📂 Repository Structure

The repository is organized as follows:

- **`Data/`**: Contains raw and processed parquet files (e.g., `stop_lines_cut.parquet`, `bus_stops.parquet`).
- **`FCD_preprocessing.ipynb`**: Notebook for data cleaning and preprocessing.
    - Projects GPS points onto the route line.
    - Removes invalid runs and outliers.
    - Filters data to include points 50-400 meters before the traffic light.
    - **Dwell Time Removal**: Specific logic to identify and remove waiting times at bus stops to isolate travel time.
- **`LSTM.ipynb`**: Implementation of the Long Short-Term Memory model.
    - Data loading and feature normalization.
    - Model architecture: LSTM layer (16 units, tanh) -> Dropout -> Dense layers.
    - Training and evaluation using RMSE and R-squared metrics.
- **`MLP.ipynb`**: Implementation of the Multi-Layer Perceptron model.
- **`linear_model.ipynb`**: Implementation of the Linear Regression baseline.
- **`mean_model.ipynb`**: Implementation of the statistical Mean Model baseline.
- **`descriptive_statistics.ipynb`** & **`distribution_average.ipynb`**: Exploratory Data Analysis (EDA) and visualizations.
- **`Thesis.pdf`**: The complete Bachelor Thesis document.

## 🛠️ Methodology

### 1. Data Collection & Preprocessing
The dataset consists of **24,949 runs** collected from October 2023 to April 2024. 
- **Geospatial Processing**: Raw GPS coordinates are projected onto a reference `LineString` of the bus route to calculate the actual distance to the stop line, rather than Euclidean distance.
- **Filtering**: Only data points between 50m and 400m before the traffic light are considered to account for congestion queue lengths.

### 2. Feature Engineering
The following features are used to train the models:
- **Distance**: Distance to the stop line along the route.
- **Time of Day**: Hour of the day to capture daily traffic patterns.
- **Day of Week**: To differentiate between weekday and weekend traffic.
- **Month of Year**: To capture seasonal variations.

### 3. Model Architectures
- **LSTM**: Designed to capture sequential dependencies in the trajectory data. It uses 16 LSTM units with `tanh` activation, followed by a Dropout layer (0.1) and Dense layers.
- **MLP**: A feedforward network with two hidden layers (16 and 8 neurons) using `ReLU` activation.
- **Linear Regression**: Used as a baseline to evaluate the complexity of the data relationships.

## 📊 Results

The models were evaluated using **Root Mean Squared Error (RMSE)** and **R-squared ($R^2$)**. The study compared performance with and without including dwell times (waiting at stops).

| Model | RMSE (With Dwell Time) | $R^2$ (With Dwell Time) | RMSE (No Dwell Time) | $R^2$ (No Dwell Time) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Model** | 19.47s | 0.43 | 5.55s | 0.86 |
| **Linear Regression** | 12.40s | 0.67 | 5.50s | 0.86 |
| **MLP** | 11.80s | 0.70 | **5.20s** | **0.86** |
| **LSTM** | **11.70s** | **0.71** | 5.50s | 0.86 |


**Key Takeaway**: Removing dwell times significantly improves prediction accuracy across all models. The LSTM and MLP models show superior performance when handling the noise introduced by dwell times.

## 💻 Getting Started

### Prerequisites
The project uses Python. You will need the following libraries installed:

```bash
pip install pandas geopandas shapely numpy scikit-learn tensorflow matplotlib
```

### Usage

1.  **Data Preprocessing**
    * Open `FCD_preprocessing.ipynb`.
    * Update the file paths and placeholders (e.g., `'INSERT YOUR PARQUET FILE'`, `Point(X, Y)`) to point to your raw Floating Car Data and specific route coordinates.
    * Run the notebook to:
        * Project GPS points onto the bus route line.
        * Filter out invalid runs or those not within the 50m-400m range.
        * Remove dwell times (waiting periods at stops) to generate the "clean" dataset.

2.  **Model Training**
    * Open the desired model notebook (e.g., `LSTM.ipynb`, `MLP.ipynb`, or `linear_model.ipynb`).
    * Update the `file_paths` list in the second cell to point to the cleaned parquet files generated in the preprocessing step (e.g., `'new_cleaned/cleaned_november_without.parquet'`).
    * Run all cells to:
        * Load and concatenate the processed data.
        * Normalize features using `StandardScaler`.
        * Split data into training (80%) and testing (20%) sets based on unique weeks.
        * Train the model and output performance metrics (RMSE, $R^2$).

3.  **Visualization**
    * The notebooks automatically plot Training vs. Validation Loss/MAE curves inline.
    * For interactive logs, the notebooks include a cell to launch TensorBoard: `%tensorboard --logdir logs/fit`.
