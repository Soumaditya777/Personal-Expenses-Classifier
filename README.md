# Personal Expense Classification

This project applies Natural Language Processing (NLP) and Machine Learning (ML) to classify personal expense transactions into categories such as `food`, `transport`, `shopping`, `technology`, and `entertainment`. It also provides rich visualizations through exploratory data analysis (EDA).

---

## 📁 Dataset

- **Filename:** `personal_expense_classification.csv`
- **Columns:**
  - `expense_id`: Unique ID for each transaction
  - `amount`: Transaction amount (float)
  - `merchant`: Name of the vendor/service provider
  - `description`: Short text describing the expense
  - `category`: Target class (e.g., food, shopping, etc.)

---

## 🔍 Exploratory Data Analysis

The dataset was analyzed to derive insights before model training.

### ✅ Checks:
- Verified all 100 rows have no missing values.
- Combined `merchant` and `description` into a new `text` feature for NLP.

### 📊 Visuals:
- **Category distribution:** Bar plot showing frequency of each category.
- **Total spending by category:** Bar chart showing total amount spent per category.
- **Average transaction amount by category:** Bar chart showing average expense per category.
- **Top 10 merchants:** Bar plot for merchants with the highest number of transactions.
- **Spending share:** Pie chart showing category-wise percentage of total spend.
- **Correlation heatmap:** Shows relationship between transaction amount and category.

---

## 🧠 Machine Learning Pipeline

### 🔧 Preprocessing
- Created a new `text` column: `merchant + description`
- Applied `LabelEncoder` to convert category names to numeric labels.

### 🧪 Train/Test Split
```python
train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
```

### 🔤 Text Vectorization
```python
TfidfVectorizer()
```

### 📈 Model
- Used `LogisticRegression` for classification.
- Achieved **100% accuracy** on the test set. ( very small dataset of 100 rows, out of it 20 were for test samples )

### ✅ Evaluation
- Classification report with precision, recall, F1-score.
- Confusion matrix for visual accuracy analysis.

---

## 🔮 Prediction Function

```python
def predict_expense(text):
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)
    return le.inverse_transform(pred)[0]
```

**Usage Example:**
```
Enter your transaction description: Uber fuel purchase
Predicted Expense Category: transport
```

---

## 📊 Visual Outputs

- **Category distribution**
- **Spending trends**
- **Top merchants**
- **Correlation heatmap**
- **Pie chart of expenses**

---

## 🛠️ Requirements

Install required packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## 📂 File Structure

```
├── personal_expense_classification.csv
├── personal_expense_classifier.ipynb
├── README.md
```

