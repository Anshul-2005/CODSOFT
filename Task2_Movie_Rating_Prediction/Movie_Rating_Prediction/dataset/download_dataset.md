# 📥 Dataset Download Instructions

## IMDb India Movies Dataset

The dataset used in this project cannot be redistributed due to Kaggle's terms of service.

### How to Download

1. **Visit Kaggle:**
   - Go to: [https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)

2. **Sign In / Create Account:**
   - You need a free Kaggle account to download datasets

3. **Download the Dataset:**
   - Click the **"Download"** button on the dataset page
   - Extract the ZIP file

4. **Place the File:**
   - Rename the file to `IMDb_Movies_India.csv` (if not already named)
   - Place it in this `dataset/` folder

### Expected File Structure

```
dataset/
├── download_dataset.md    ← You are here
└── IMDb_Movies_India.csv  ← Place the downloaded file here
```

### Dataset Details

| Property | Value |
|----------|-------|
| **Filename** | `IMDb_Movies_India.csv` |
| **Rows** | ~15,000+ |
| **Columns** | 10 |
| **Size** | ~500 KB |
| **Format** | CSV (comma-separated values) |
| **License** | CC0: Public Domain |

### Expected Columns

| Column | Type | Description |
|--------|------|-------------|
| Name | string | Movie name |
| Year | string | Release year (may contain ranges like "2020-2021") |
| Duration | string | Runtime (e.g., "120 min") |
| Genre | string | Movie genre(s), comma-separated |
| Rating | float | **TARGET** - IMDb rating (0-10 scale) |
| Votes | string | Number of votes (may contain commas) |
| Director | string | Director name(s) |
| Actor 1 | string | Lead actor |
| Actor 2 | string | Supporting actor |
| Actor 3 | string | Supporting actor |

### Data Quality Notes

- **Year**: May contain date ranges or special characters
- **Duration**: Needs extraction of numeric minutes
- **Votes**: Contains comma separators, needs cleaning
- **Rating**: Target variable for regression
- **Missing Values**: Present in several columns

### Verification

After placing the file, verify by running:

```python
import pandas as pd
df = pd.read_csv('dataset/IMDb_Movies_India.csv', encoding='latin-1')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

---

*CodSoft Data Science Internship — Task 2*
