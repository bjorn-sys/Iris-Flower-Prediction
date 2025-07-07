# Iris-Flower-Prediction

# 🌸 Iris Flower Classification with Logistic Regression
* This project uses the Iris dataset to classify flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica. We use logistic regression for classification, conduct exploratory data analysis (EDA), visualize feature relationships, and evaluate model performance using a confusion matrix and classification report.
---
# 📁 Dataset Overview
* Source: UCI Machine Learning Repository

* Records: 150 rows

**Features:**

* SepalLengthCm

* SepalWidthCm

* PetalLengthCm

* PetalWidthCm

* Species (target)

* The dataset has no missing or duplicated records and contains numeric measurements along with a categorical target.
---
# 📊 Exploratory Data Analysis
**📌 Distribution of Features**
* Sepal Length: Ranges from 4.4 to 7.9 cm

* Sepal Width: Ranges from 2.0 to 4.4 cm

* Petal Length: Ranges from 1.0 to 6.9 cm

* Petal Width: Ranges from 0.1 to 2.5 cm

* Visualized using seaborn.histplot with KDE for all features.

**📌 Species Distribution**
* Each class contains 50 samples, confirming a balanced dataset.

**📌 Correlation Heatmap**
* The strongest positive correlation is between Petal Length and Petal Width. Sepal Width and Sepal Length are slightly negatively correlated.

**📌 Pairwise Relationships**
* Pair plots show clear class separation, especially between Setosa and the other two species based on petal dimensions.
---
# 🌿 Species Filtering
**Subset analyses were done by splitting the dataset into:**

* Setosa

* Versicolor

* Virginica

* Each group was individually inspected to confirm distinguishable patterns in petal and sepal sizes.
---

# 🔧 Data Preprocessing
**Mapped Species column to numeric labels:**

* Iris-setosa → 0

* Iris-versicolor → 1

* Iris-virginica → 2

* Dropped Id column

* Standardized the features using StandardScaler
---

# 🧠 Model Building
**📌 Model Used**
* Logistic Regression from sklearn.linear_model

**📌 Train-Test Split**
* 80% Training, 20% Testing

* Stratified split using random_state=42 for reproducibility

**📌 Performance Metrics**
* Accuracy: 100%

**Classification Report:**

* All classes scored 1.00 in precision, recall, and F1-score

**Confusion Matrix:**
* Perfect classification across all species
---
# 💾 Model Saving
* The trained logistic regression model is saved using pickle:

* with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(logistic_regression, file)
  
* This enables easy deployment or loading for inference.
---

# 📁 File Structure

.
├── Iris.csv
├── iris_analysis.ipynb
├── logistic_regression_model.pkl
├── confusion_matrix.png
└── README.md
---


