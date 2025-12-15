# Tree of Life Counseling: Client Appointment Prediction and Marketing Optimization

This project applies supervised machine learning techniques to help Tree of Life Counseling Center strategically target marketing efforts, increase morning appointment bookings, and enhance operational efficiency.

We analyze client demographics and booking patterns to identify which types of clients are most likely to book appointments during the hard-to-fill morning slots (9:00 a.m. – 12:00 p.m.).

---

### 🌟 Project Mission & Goals

Tree of Life Counseling Center is a family-owned, for-profit private mental health practice providing therapy and medication management services. Our mission is to promote accessible, high-quality mental health care.

#### **Core Objectives:**

* **Analyze Client Demographics:** Uncover patterns related to appointment times across Tree of Life Counseling’s locations.
* **Predict Morning Bookings:** Identify clients more likely to book hard-to-fill morning appointments (9:00 a.m. – 12:00 p.m.).
* **Apply Supervised Learning:** Utilize classification models, emphasizing multi-label classification, to understand these client-appointment relationships.
* **Provide Actionable Insights:** Deliver data-driven recommendations to improve marketing, increase appointment bookings, and enhance operational efficiency.

#### **Anticipated Business Impact:**

* ✅ **Increase Appointment Bookings:** Especially for underfilled morning time slots.
* ✅ **Optimize Marketing Strategies:** Focus efforts on high-probability morning clients.
* ✅ **Improve Operational Efficiency:** Enhance scheduling and decision-making through data insights.

---

### Technical Implementation and Usage

This project is implemented using Python and several key data science libraries.

#### **Prerequisites**

The following libraries are required to run the notebooks:

`Scikit Learn`
`NumPy`
`Pandas`
`MatPlotLib`
`Seaborn`
`Folium`
`GeoPy`

# Project Structure

The analysis and modeling are contained within three distinct Jupyter Notebooks:

| Notebook                                      | Description                                                                 |
|-----------------------------------------------|-----------------------------------------------------------------------------|
| `TOL_project_visualization.ipynb`             | Contains data processing (consistency, feature engineering, one-hot encoding) and visualization of crucial Key Performance Indicators (KPIs). |
| `TOL_project_knn_and_lr.ipynb`                | Implementation of the K-Nearest Neighbors (KNN) model and Logistic Regression model. |
| `TOL_project_gbdt.ipynb`                      | Implementation of the Gradient Boosting Decision Tree (GBDT) model.        |

## Visualization Highlights (KPIs)

The visualization notebook focuses on two key areas:

1. **Potential Morning Customers**  
   Charts created using `matplotlib` and `seaborn` to visualize client features correlated with morning appointment demand.

2. **New Location & Advertisement Analysis**  
   - `GeoPy` is used to extract latitudes and longitudes of client town locations.  
   - A heatmap is plotted on a New Jersey map using `folium` to analyze potential new Tree of Life locations and optimal highway selections for billboard advertisements.

## 📊 Model Performance and Key Findings

We implemented and compared three supervised learning models (with KNN emerging as superior). KNN was selected as the final model due to its better ability to identify morning bookings.

### Model Comparison (Classification Models)

| Model                  | Accuracy | ROC AUC | Recall (Morning) | Pros                                                                 | Cons                                                                 |
|-----------------------|----------|---------|------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Logistic Regression   | 0.75     | 0.67    | 0.09             | Simple, fast, clear feature influence.                               | Poor performance on imbalanced data, fails to capture complex non-linear relationships. |
| GBDT (Gradient Boosting) | 0.75  | 0.67    | 0.09             | Captures complex patterns, shows feature importance, performs well on structured data. | Biased toward non-morning class, requires more tuning.              |
| KNN (K-Nearest Neighbors) | -    | -       | **0.25**         | Recovered 25% of morning sessions.                                   | (Chosen for production)                                              |

### Key Insight: KNN Model Selection

KNN recovered **25%** of morning sessions versus approximately **9%** recovered by the other models. Optimizing for **Recall** (correctly identifying the positive class—a morning booking) was crucial given the business goal of filling these underbooked slots.

### Features Indicating Higher Morning Demand

- **Age Range**: Certain age groups showed a higher propensity for morning bookings.
- **Referral Source**: Physician and clinical referrals notably leaned toward morning sessions.
- **Office Location**: Specific Tree of Life locations had higher morning booking rates.
- **Intake Method**: Clients who used phone intake aligned more frequently with morning appointments.

## Challenges and Solutions

| Challenge                          | Solution Implemented                                                                 |
|------------------------------------|--------------------------------------------------------------------------------------|
| Imbalanced Labels                  | Compared results across different models (including simpler ones like Logistic Regression) to ensure robustness. |
| Low Recall Scores                  | Optimized model selection and tuning specifically for the Recall metric, leading to the selection of the KNN model. |
| Small Dataset                      | Focused on implementing simpler models (KNN, LR) to avoid overfitting and ensure reliable generalization. |
| Data Limitations                   | Limited information available (due to HIPAA-compliant cleaning) such as gender, exact age, or occupation (e.g., student, housewife) could have further enhanced predictions. |
