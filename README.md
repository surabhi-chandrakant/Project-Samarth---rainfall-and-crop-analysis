# Crop Rainfall Analysis Project

A comprehensive analysis of the relationship between crop production and rainfall patterns in India using open data from data.gov.in.

## 📊 Project Overview

This project analyzes the correlation between rainfall patterns and crop production across different districts in India. The analysis helps understand how meteorological factors impact agricultural productivity.

## 🗂️ Data Sources

### 1. Crop Production Data
- **Source**: https://www.data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
- **Collection Method**: API Integration
- **Time Period**: 1997 onwards
- **Key Fields**: District, Season, Crop, Production Area, Yield

### 2. Rainfall Data
- **Source**: https://www.data.gov.in/resource/daily-district-wise-rainfall-data
- **Collection Method**: API Integration
- **Time Period**: Daily rainfall records
- **Key Fields**: District, Date, Rainfall (mm)

## 🔄 Data Processing Pipeline

### Step 1: Data Collection
- Fetched crop production data via data.gov.in API
- Retrieved daily district-wise rainfall data via API
- Initial crop dataset contained extensive historical records

### Step 2: Data Sampling
- Randomly sampled crop data to 4000 records for manageable analysis
- Maintained proportional representation across districts and seasons

### Step 3: Data Merging
- Merged crop production data with rainfall data using district and temporal keys
- Aligned seasonal crop data with corresponding rainfall periods

### Step 4: Preprocessing
- Handled missing values in both datasets
- Standardized district names across datasets
- Normalized rainfall data (monthly/seasonal aggregates)
- Cleaned production values and removed outliers
- Created derived features (seasonal rainfall, anomalies)

### Step 5: Final Dataset
- Output: `data.csv` - Cleaned, merged dataset ready for analysis
- Size: Optimized for analysis while maintaining statistical significance

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/surabhi-chandrakant/samrath-projects.git
   cd samrath-projects
   ```

2. **Create virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   - Copy `.env` file and configure API keys if needed
   - Update data source URLs if necessary

5. **Running the Project**
   ```bash
   python app3.py
   ```

6. **View results**
   - Open `index.html` in your browser to see visualizations
   - Check generated reports in the output folder

## 📁 Project Structure
```
samrath-projects/
├── app3.py                 # Main application script
├── data.csv                # Processed merged dataset
├── requirements.txt        # Python dependencies
├── .env                    # Environment configuration
├── index.html              # Dashboard/Visualization
└── README.md               # Project documentation
```

## 📊 Analysis Features
- Correlation analysis between rainfall and crop yield
- District-wise comparative analysis
- Seasonal pattern identification
- Visualization of rainfall-crop relationships
- Statistical summary reports

## 🚀 Usage
- The application processes the merged crop-rainfall data
- Generates interactive visualizations and insights
- Provides district-level and crop-specific analysis
- Outputs statistical summaries and correlation metrics

## 🤝 Contributing
Feel free to contribute to this project by:
- Adding new analysis methods
- Improving data processing pipelines
- Enhancing visualizations
- Expanding to include more agricultural parameters

## 📄 License
This project uses open data from data.gov.in and is intended for educational and research purposes.

## 📧 Contact
For questions or suggestions regarding this analysis, please open an issue in the repository.
