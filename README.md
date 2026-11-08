# PowerCo – Customer Churn Analysis

Work completed for the [BCG X Data Science Job Simulation](https://www.theforage.com/simulations/bcg/data-science-ccdz) on Forage (February 2026).

PowerCo, a gas and electricity supplier to small and medium-sized businesses, was losing customers and believed price sensitivity was the cause. The task was to test that hypothesis, build a model that predicts which customers will churn, and recommend what to do about it. The datasets provided by Forage are for simulation use only and are not included here.

**Notebook:** [`bcg_data_science_churn_analysis_sim.ipynb`](https://github.com/sravyam230/bcg-data-science-churn-analysis-sim/blob/main/bcg_data_science_churn_analysis_sim.ipynb)

## Data

- `client_data.csv`: 14,606 customers with consumption over the last 12 months and last month, forecast consumption and prices, margins, number of products, tenure, sales channel and a churn flag. Churn rate 9.7%.
- `price_data.csv`: 193,002 rows of monthly 2015 prices per customer, with variable and fixed components for off-peak, peak and mid-peak periods.

## Approach

1. **Framing.** Set out for the client which other churn drivers would need controlling for (contract terms, tariff structure, service, competitor offers), what data was needed, and the analysis plan.
2. **Exploratory data analysis.** Data types, distributions, missing values, churn rate, prices over time and a direct comparison of price between churners and non-churners. Forecast prices for the two groups were almost identical, the first sign that the hypothesis was incomplete.
3. **Feature engineering.** Change in off-peak price between January and December; average gaps between off-peak, peak and mid-peak prices per customer; the largest month-to-month jump in each gap; tenure and months since activation, modification and renewal; log transforms of the skewed consumption columns; one-hot encoding of sales channel and origin with rare categories grouped; removal of two highly correlated columns. Final table: 14,606 rows, 59 features.
4. **Modelling.** Class-weighted random forest (300 trees, depth 12, minimum leaf 20) evaluated with five-fold stratified cross-validation on accuracy, precision, recall and ROC-AUC, then on a held-out 25% test split. Accuracy was deliberately not the target, since predicting "stays" for everyone would score 90%.
5. **Findings and recommendations** for the client.

## Results

| | Cross-validation | Test split |
|---|---|---|
| Recall | 0.48 | 0.52 |
| Precision | 0.19 | 0.20 |
| ROC-AUC | 0.69 | 0.70 |

The model finds about half of the customers who churn at one-in-five precision: good enough to decide who receives a retention call, not good enough for automatic discounts.

**Price is not the main driver.** The strongest features are the power margins, 12-month consumption, meter rent and months active. The best price feature (the average off-peak to peak gap) ranks tenth and the January-to-December change fourteenth. Churners are slightly newer and slightly more profitable customers than those who stay: churn falls from about 18% at two years' tenure to about 7% at six.

**Recommendation.** Target retention offers at customers the model flags as high risk rather than discounting across the whole base, comparing the cost of each offer with the net margin it protects.

## Tools

Python (pandas, NumPy, scikit-learn, Matplotlib, seaborn)
