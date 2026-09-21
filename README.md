# Vipul Shukla — House Price Prediction

## Project Overview
This project predicts residential property prices from two input features:
- **area** — house area in square feet
- **rooms** — number of rooms
- **price** — target house price

The workflow follows the IBM SkillsBuild/BharatCares project submission structure shown in the training videos: data loading, preprocessing, train/test split, scaling, Linear Regression, evaluation, and saving the trained model.

## Files
- `Vipul_HousePricePrediction.py` — complete Python training/evaluation code.
- `requirements.txt` — required Python libraries.
- `house_price.csv` — project dataset used by the code.
- `Vipul_HousePricePrediction_Report.docx` — project documentation/report.
- `README.md` — setup and project overview.
- `app.py` — optional Streamlit interface for trying predictions after the model has been trained.

## Dataset
The included `house_price.csv` is the dataset used for this submission and is intentionally kept small and easy to understand for an academic demonstration. It follows the `area, rooms, price` structure demonstrated in the training session.

## How to Run
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `python Vipul_HousePricePrediction.py`

The script prints the dataset summary, MAE, RMSE and R², then saves:
- `house_price_model.pkl`
- `house_price_scaler.pkl`

## Optional Web App
After running the training script:
`streamlit run app.py`

## Machine Learning Method
**Linear Regression** is used because the project is a simple supervised regression problem where the target is a continuous house price.

## Important Note
Predictions are estimates for educational use and should not be treated as real property valuations.
