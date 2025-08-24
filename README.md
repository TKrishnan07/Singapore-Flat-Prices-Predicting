# Singapore Resale Flat Price Prediction

## Project Overview
This project predicts the resale prices of Housing Development Board (HDB) flats in Singapore using machine learning. It leverages historical resale transaction data and uses a Random Forest Regressor model to estimate flat prices based on features such as town, flat type, floor area, storey range, lease commencement date, and transaction year.

The model is deployed in an interactive **Streamlit** web application enabling users to input flat details and receive instant resale price predictions.

---

## Features
- Data preprocessing including encoding categorical variables and extracting numeric features.
- Exploratory Data Analysis (EDA) visualizations for better insight into the dataset.
- Trained Random Forest regression model with performance evaluation metrics (MAE, MSE, R²).
- Streamlit app with a user-friendly sidebar navigation and organized tabs:
  - Dataset Sample
  - Exploratory Data Analysis
  - Model Performance Metrics
  - Interactive Price Prediction

---

## Installation

1. Clone the repository or download the project files.
2. Install required Python packages:

pip install -r requirements.txt
*If `requirements.txt` is not provided, install manually:*

pip install streamlit pandas scikit-learn matplotlib seaborn numpy

---

## Usage

Run the Streamlit app with the following command:

streamlit run app.py

Open the browser window that appears (usually at `http://localhost:8501`). Use the sidebar to navigate between:

- **Data:** View the first few rows of the resale flat dataset.
- **EDA:** Explore visualizations of resale prices and feature correlations.
- **Model Performance:** Review accuracy metrics on the test data.
- **Prediction:** Enter flat details and predict resale price interactively.

---

## File Structure

- `app.py` — Main Streamlit app integrating data loading, visualization, model evaluation, and prediction.
- `flat_price_rf_model.pkl` — Trained Random Forest pickle model.
- `category_maps.pkl` — Pickle file storing category-to-code mappings for encoding.
- `ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv` — Dataset of resale transactions.

---

## How It Works

1. **Data Loading and Preprocessing**  
   The app loads the CSV data, converts dates, encodes categorical features, and computes storey range midpoint.

2. **Model Loading**  
   The pre-trained Random Forest model is loaded for prediction and evaluation.

3. **User Interaction**  
   The sidebar allows navigation. Within Prediction tab, users input property details, which get encoded and fed to the model for price prediction.

4. **Visualization and Metrics**  
   EDA tab provides histograms, bar charts, and correlation heatmaps. Model Performance shows MAE, MSE, and R².

---

## Future Improvements

- Add external factors such as proximity to MRT stations, amenities, or economic indicators.
- Include advanced models like XGBoost or LightGBM with hyperparameter tuning.
- Incorporate more granular temporal features (month or quarter).
- Handle unseen category values dynamically.
- Extend deployment with user authentication and logging.

---

## Contact

For questions or suggestions, please contact [Your Name] at [Your Email].

---

*This project is for educational and demonstration purposes.*  

 
If you want, I can also generate a requirements.txt file or help you create a README that includes instructions to deploy on Streamlit Cloud or other cloud services.

⁂
 
1.	https://pypi.org/project/streamlit-project-template/ 
2.	https://blog.streamlit.io/streamlit-app-starter-kit-how-to-build-apps-faster/ 
3.	https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/deploy-from-a-template 
4.	https://www.youtube.com/watch?v=AiIptnSahMs 
5.	https://www.youtube.com/watch?v=3XFdq9RDz6A 
6.	https://readme-ai.streamlit.app 
7.	https://discuss.streamlit.io/t/a-new-library-and-template-using-react-hooks-for-creating-streamlit-components/17296 
8.	https://raw.githubusercontent.com/streamlit/streamlit/master/README.md 
9.	https://docs.streamlit.io 
