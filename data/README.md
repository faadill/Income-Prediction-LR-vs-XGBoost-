# Data

This project uses the **Census Income ("Adult") dataset** — 10,000 observations, 11 variables, with `Income` (`<=50K` / `>50K`) as the binary target.

The raw dataset isn't redistributed in this repo. You can obtain an equivalent public version here:

- [UCI Machine Learning Repository — Adult Data Set](https://archive.ics.uci.edu/dataset/2/adult)
- [Kaggle — Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)

Download the CSV and place it at `data/income.csv` with a target column named `Income` to run `scripts/income_prediction_analysis.R` as-is.

### Variables

| Feature | Type | Values / Range |
|---|---|---|
| age | Integer | 17 – 90 (mean: 38.75) |
| workclass | Categorical | 8 levels (e.g. Private, Gov) |
| education | Categorical | 16 levels |
| education_num | Integer | 1 – 16 (mean: 10.07) |
| marital-status | Categorical | 7 levels |
| occupation | Categorical | 14 levels |
| relationship | Categorical | 6 levels |
| race | Categorical | 5 levels |
| sex | Categorical | Male / Female |
| hours-per-week | Integer | 1 – 99 (mean: 40.46) |
| Income | Binary (response) | <=50K / >50K |
