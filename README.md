# California Housing Price Prediction using SGDRegressor

This project uses **Stochastic Gradient Descent (SGD) Regression** from `scikit-learn` to predict housing prices in California based on various features such as median income, average rooms, and location attributes.

---

## 📂 Project Overview

The goal is to train an SGD-based linear regression model on the **California Housing dataset** and visualize its performance over multiple iterations.

### Key Steps:
1. Load the California housing dataset.
2. Perform exploratory data analysis (EDA) using **Seaborn** and **Matplotlib**.
3. Normalize the data using **StandardScaler**.
4. Train the model iteratively using `SGDRegressor` with partial fitting.
5. Track model performance using MSE and R² score.
6. Visualize:
   - Actual vs Predicted values
   - MSE per iteration

---

## 🧠 Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🚀 How to Run the Project

### 1. Clone this repository
```bash
git clone https://github.com/HarshadBhutal/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the script
```bash
python house_price_prediction.py
```
