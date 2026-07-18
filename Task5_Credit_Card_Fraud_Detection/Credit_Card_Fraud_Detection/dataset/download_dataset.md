# Dataset Download Instructions

## Credit Card Fraud Detection Dataset

### Source
- **Platform:** Kaggle
- **URL:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`

### Download Steps

1. Visit the [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download** (requires Kaggle account)
3. Extract the ZIP file
4. Place `creditcard.csv` inside this `dataset/` folder

### Using Kaggle CLI

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d dataset/
```

### Dataset Details

| Property | Value |
|----------|-------|
| **Rows** | 284,807 |
| **Columns** | 31 |
| **Target** | Class (0 = Legitimate, 1 = Fraud) |
| **Fraud %** | ~0.172% |
| **Features** | V1–V28 (PCA transformed), Time, Amount |

### Important Notes
- The dataset contains transactions made by European cardholders in September 2013
- Features V1–V28 are principal components obtained via PCA
- Only `Time` and `Amount` are original features
- The dataset is highly imbalanced (fraud represents only 0.172%)
