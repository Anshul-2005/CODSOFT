# 📥 Dataset Download Instructions

## Titanic Dataset

The dataset used in this project cannot be redistributed due to Kaggle's terms of service.

### How to Download

1. **Visit Kaggle:**
   - Go to: [https://www.kaggle.com/datasets/yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

2. **Sign In / Create Account:**
   - You need a free Kaggle account to download datasets

3. **Download the Dataset:**
   - Click the **"Download"** button on the dataset page
   - Extract the ZIP file if necessary

4. **Place the File:**
   - Rename the file to `Titanic-Dataset.csv` (if not already named)
   - Place it in this `dataset/` folder

### Expected File Structure

```
dataset/
├── download_dataset.md    ← You are here
└── Titanic-Dataset.csv    ← Place the downloaded file here
```

### Dataset Details

| Property | Value |
|----------|-------|
| **Filename** | `Titanic-Dataset.csv` |
| **Rows** | 891 |
| **Columns** | 12 |
| **Size** | ~60 KB |
| **Format** | CSV (comma-separated values) |

### Expected Columns

| Column | Type | Description |
|--------|------|-------------|
| PassengerId | int | Unique identifier |
| Survived | int | Target variable (0 = No, 1 = Yes) |
| Pclass | int | Ticket class (1, 2, 3) |
| Name | string | Passenger name |
| Sex | string | Gender (male/female) |
| Age | float | Age in years |
| SibSp | int | # siblings/spouses aboard |
| Parch | int | # parents/children aboard |
| Ticket | string | Ticket number |
| Fare | float | Passenger fare |
| Cabin | string | Cabin number |
| Embarked | string | Port of embarkation (C/Q/S) |

### Verification

After placing the file, verify by running:

```python
import pandas as pd
df = pd.read_csv('dataset/Titanic-Dataset.csv')
print(f"Shape: {df.shape}")  # Should output: Shape: (891, 12)
```

---

*CodSoft Data Science Internship — Task 1*
