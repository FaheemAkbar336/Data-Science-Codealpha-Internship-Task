# Data Science Codealpha Internship Task

## 📖 Overview
Iris flower classification using K-Nearest Neighbors (KNN) algorithm. This project demonstrates end-to-end machine learning workflow including data exploration, visualization, model training, and hyperparameter tuning.

## 🎯 Project Objectives
- Load and explore the Iris dataset
- Visualize relationships between features
- Build and train a KNN classifier
- Evaluate model performance
- Optimize hyperparameters using GridSearchCV

## 📊 Dataset
- **Source**: Iris dataset (IRIS.csv)
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Target**: Iris species (3 classes)
- **Samples**: 150 records

## 🛠️ Technologies Used
- **Python 3.x**
- pandas - Data manipulation
- scikit-learn - Machine learning
- matplotlib & plotly - Data visualization
- numpy - Numerical computing

## 📦 Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Usage
1. Ensure `IRIS.csv` is in the project directory
2. Run the script:
```bash
python main.py
```

## 🔄 Workflow Steps

1. **Data Loading**: Import Iris dataset from CSV
2. **Exploratory Analysis**: View statistics and unique labels
3. **Visualization**: Scatter plots to identify patterns
4. **Data Preparation**: Split features and target (80/20 train-test)
5. **Model Training**: Train KNN classifier (k=3)
6. **Evaluation**: Accuracy, confusion matrix, classification report
7. **Hyperparameter Tuning**: GridSearchCV for optimal k value
8. **Prediction**: Make predictions on new samples

## 📈 Results & Performance Metrics
- **Accuracy Score**: Run `main.py` to see results
- **Confusion Matrix**: Generated in output
- **Best k value**: Determined by GridSearchCV

## 📝 Project Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
└── IRIS.csv
```

## 🚀 Future Improvements
- Try other algorithms (SVM, Random Forest, Neural Networks)
- Cross-validation and k-fold analysis
- Feature scaling/normalization
- ROC curve and AUC analysis
- Deploy model as API

## 👤 Author
Faheem Akbar

## 📄 License
MIT License - See LICENSE file for details
