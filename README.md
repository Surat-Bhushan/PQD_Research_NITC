# Task 2 - Univariate Linear Regression using Gradient Descent

## Objective

The objective of this task is to implement **Univariate Linear Regression** from scratch using **Gradient Descent**, analyze the effect of different learning rates on convergence, and compare the obtained results with Scikit-Learn's `LinearRegression` model.

---

## Dataset

The following data points are used:

```python
X = [0, 0.18, 0.26, 0.57, 0.48,
     0.62, 0.44, 0.55, 0.89,
     1.0, 0.92]
```

The target values are generated using:

```python
Y = 0.5 * X + 1 + 0.001 * randn()
```

where a small amount of random noise is added to simulate real-world observations.

---

## Linear Regression Model

The univariate linear regression model is:

y = mx + b

where:

- **m** = slope (weight)
- **b** = intercept (bias)

---

## Loss Function

Mean Squared Error (MSE) is used as the loss function:

<img width="165" height="38" alt="Screenshot 2026-05-24 at 10 35 03 PM" src="https://github.com/user-attachments/assets/4e3a6de4-c96d-47f7-873e-c1072cb6911e" />
The objective of training is to find values of **m** and **b** that minimize this loss.

---

## Gradient Descent
<img width="298" height="221" alt="Screenshot 2026-05-24 at 10 35 48 PM" src="https://github.com/user-attachments/assets/d861f89c-4d40-43d4-8372-1710def6530c" />

---

## Training Procedure

1. Initialize:
   - Weight (m) = 0
   - Bias (b) = 0

2. Train using Gradient Descent for multiple learning rates:

```python
[0.01, 0.05, 0.1, 0.5]
```

3. Run Gradient Descent for:
   - Maximum 1000 iterations
   - Or until convergence

4. Record:
   - Loss at every iteration
   - Final weight and bias
   - Number of iterations required for convergence

---

## Visualization

### 1. Loss vs Iterations

Shows the convergence behavior of different learning rates.

Observations:

- Smaller learning rates converge slowly.
- Larger learning rates converge faster.
- Excessively large learning rates may overshoot the optimum.
- An optimal learning rate achieves fast and stable convergence.

### 2. Regression Line

A scatter plot of the dataset along with the learned regression line is plotted to visualize the fit.

---

## Prediction

Using the model parameters obtained from the best learning rate, predictions are made for:

```python
x = 0.30
x = 0.75
```

---

## Verification using Scikit-Learn

The implementation is verified using:

```python
from sklearn.linear_model import LinearRegression
```

The following are compared:

- Weight (m)
- Bias (b)
- Predicted values

The closeness of these values validates the correctness of the Gradient Descent implementation.

---

## Experimental Observations and Parameter Tuning

During implementation, multiple experiments were performed to understand the effect of learning rate, convergence tolerance, and iteration limits.

---

### Experiment 1

**Maximum Iterations:** 1000  
**Tolerance:** 1e-8

#### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000212471 | 1000 (did not converge) |
| 0.05 | 0.0000014215 | 442 |
| 0.1 | 0.0000010616 | 246 |
| 0.5 | 0.0000007697 | 61 |

#### Observation

- Learning rate 0.5 achieved the lowest loss and fastest convergence.
- Learning rate 0.01 did not converge within 1000 iterations.
- This raised the question of whether 0.01 was truly worse or simply required more training.

---

### Experiment 2

**Maximum Iterations:** 5000  
**Tolerance:** 1e-8

#### Observation

The iteration limit was increased to investigate the behavior of the smaller learning rate.

Even with 5000 iterations, learning rate 0.01 was still being stopped prematurely by the convergence criterion before reaching a loss comparable to the other learning rates.

This suggested that the tolerance value itself might be causing early termination.

---

### Experiment 3

**Maximum Iterations:** 5000  
**Tolerance:** 1e-12

#### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000007109 | 4948 |
| 0.05 | 0.0000007106 | 1104 |
| 0.1 | 0.0000007106 | 576 |
| 0.5 | 0.0000007105 | 125 |

#### Observation

- All learning rates converged to nearly the same minimum loss.
- Learning rate 0.01 was not inferior; it simply required substantially more iterations.
- The stricter tolerance prevented premature stopping and allowed optimization to approach the true optimum more closely.
- Learning rate 0.5 still reached the optimum much faster than the other learning rates.

---

### Experiment 4 (Final Configuration)

**Maximum Iterations:** 1000  
**Tolerance:** 1e-12

#### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000212471 | 1000 (did not converge) |
| 0.05 | 0.0000007108 | 1000 (did not converge) |
| 0.1 | 0.0000007106 | 576 |
| 0.5 | 0.0000007105 | 125 |

#### Observation

- This configuration follows the assignment requirement of running for **1000 iterations or until convergence**.
- Learning rate 0.01 still required more than 1000 iterations and therefore did not fully converge.
- Learning rate 0.05 reached a loss very close to the optimum but also required slightly more than 1000 iterations for full convergence.
- Learning rates 0.1 and 0.5 successfully converged within the allowed iteration budget.
- Learning rate 0.5 achieved the same optimum while requiring the fewest iterations.

---

## Final Model Comparison

#### Gradient Descent (Best Learning Rate = 0.5)

```text
Weight = 0.498645
Bias   = 1.001412
```

#### Scikit-Learn Linear Regression

```text
Weight = 0.498636
Bias   = 1.001417
```

The difference is only in the fifth and sixth decimal places, confirming the correctness of the implementation.

---

### Predictions

#### Gradient Descent

| x | Predicted y |
|---|------------|
| 0.30 | 1.151005 |
| 0.75 | 1.375395 |

#### Scikit-Learn

| x | Predicted y |
|---|------------|
| 0.30 | 1.151008 |
| 0.75 | 1.375394 |

The predictions are nearly identical.

---

## Final Conclusion

The final implementation uses:

- Maximum Iterations = 1000
- Tolerance = 1e-12

This configuration was chosen because:

1. It satisfies the assignment requirement of running for **1000 iterations or until convergence**.
2. It avoids the premature stopping observed with a tolerance of 1e-8.
3. It allows converging learning rates to approach the optimum more accurately.
4. It produces results that closely match Scikit-Learn's `LinearRegression` model.

Among the tested learning rates, **α = 0.5** was selected as the best learning rate because it achieved essentially the same optimum as the smaller learning rates while converging in the fewest iterations.

## Plots Generated (Final Configuration)
<img width="743" height="471" alt="Screenshot 2026-05-24 at 10 52 40 PM" src="https://github.com/user-attachments/assets/0ddf318b-a7aa-42d6-8fc7-bfe2b81f5377" />
<img width="739" height="473" alt="Screenshot 2026-05-24 at 10 52 35 PM" src="https://github.com/user-attachments/assets/f758101c-c80e-472f-bf60-2301929ef609" />


# Multivariate Linear Regression on California Housing Dataset

## Objective

The objective of this task is to build a Multivariate Linear Regression model using the California Housing dataset to predict the median house value based on housing and demographic features. The model is trained using Scikit-Learn's `LinearRegression` class and evaluated using both regression metrics and classification metrics.

---

## Dataset Description

The dataset contains information about California housing block groups collected during the 1990 Census.

Each record represents a **census block group**, not an individual house.

#### Features Available

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income

#### Target Variable

- median_house_value

#### Important Note

The dataset does **not** represent individual houses.

Each row represents a geographical block group containing multiple households.

Therefore, the model predicts the **median house value of a block group**, not the value of a single house.

---

## Data Preprocessing

#### Feature Selection

As specified in the problem statement, the following columns were excluded:

- longitude
- latitude

The remaining features were used as input variables:

- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income

#### Train-Test Split

The dataset was divided into:

- Training Set: 80%
- Testing Set: 20%

This allows the model to be trained on one portion of the data and evaluated on previously unseen data.

Training Samples = 13,600

Testing Samples = 3,400

---

## Multivariate Linear Regression

Unlike univariate regression, which uses a single feature, multivariate regression uses multiple input features.

The model can be represented as:

\[
y = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + w_5x_5 + w_6x_6 + b
\]

where:

- \(y\) = predicted median house value
- \(x_i\) = feature values
- \(w_i\) = learned coefficients
- \(b\) = intercept

---

## Learned Model Parameters

| Feature | Coefficient |
|----------|-------------:|
| housing_median_age | 1849.79 |
| total_rooms | -20.65 |
| total_bedrooms | 95.97 |
| population | -32.53 |
| households | 128.47 |
| median_income | 47705.63 |

#### Intercept

\[
b = -46224.84
\]

The intercept represents the predicted value when all feature values are zero.

Since a block group with all features equal to zero is not realistic, the intercept is not interpreted physically and primarily acts as a constant offset that helps the model fit the data.

---

## Model Evaluation

### Regression Metrics

Since house price prediction is a regression problem, the following regression metrics were used.

---

### Mean Absolute Error (MAE)

#### Formula

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat y_i|
\]

#### Meaning

Measures the average absolute prediction error.

#### Result

\[
MAE = 55065.78
\]

Interpretation:

On average, predictions differ from actual house values by approximately \$55,066.

---

### Mean Squared Error (MSE)

#### Formula

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat y_i)^2
\]

#### Meaning

Measures the average squared prediction error.

Large errors are penalized more heavily because errors are squared.

#### Result

\[
MSE = 5,558,804,420.35
\]

---

### Root Mean Squared Error (RMSE)

#### Formula

\[
RMSE = \sqrt{MSE}
\]

#### Meaning

Represents prediction error in the same units as house prices.

#### Result

\[
RMSE = 74,557.39
\]

Interpretation:

Typical prediction error is approximately \$74,557.

---

### R² Score

#### Formula

\[
R^2 = 1 - \frac{SS_{residual}}{SS_{total}}
\]

where

\[
SS_{residual} = \sum_{i=1}^{n}(y_i - \hat y_i)^2
\]

and

\[
SS_{total} = \sum_{i=1}^{n}(y_i - \bar y)^2
\]

#### Meaning

R² measures how much of the variation in house prices is explained by the model.

#### Result

\[
R^2 = 0.5966
\]

Interpretation:

The model explains approximately **59.66%** of the variation in median house values.

---

#3 Classification Metrics

### Why Were Classification Metrics Used?

The assignment explicitly required:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

However, these metrics are normally used for classification problems rather than regression problems.

To satisfy the assignment requirements, house prices were converted into two classes.

---

## Conversion from Regression to Classification

The median house value of the test set was used as a threshold.

#### Rule

If:

\[
Price > Median(y_{test})
\]

Class = 1 (High Price)

Otherwise:

Class = 0 (Low Price)

#### Why Median?

The median generally divides the dataset into two nearly equal groups and is less affected by extreme values than the mean.

---

## Confusion Matrix

| | Predicted Low | Predicted High |
|----|----|----|
| Actual Low | 1228 | 472 |
| Actual High | 238 | 1462 |

Definitions:

- TP = True Positive
- TN = True Negative
- FP = False Positive
- FN = False Negative

---

### Accuracy

#### Formula

\[
Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}
\]

#### Result

\[
Accuracy = 0.7912
\]

Interpretation:

Approximately 79.12% of houses were correctly classified.

---

##3 Precision

#### Formula

\[
Precision =
\frac{TP}
{TP + FP}
\]

#### Result

\[
Precision = 0.7559
\]

Interpretation:

Among all houses predicted as high-value, approximately 75.59% were actually high-value.

---

### Recall

#### Formula

\[
Recall =
\frac{TP}
{TP + FN}
\]

#### Result

\[
Recall = 0.8600
\]

Interpretation:

The model correctly identified approximately 86% of actual high-value houses.

---

### F1 Score

#### Formula

\[
F1 =
\frac{2 \times Precision \times Recall}
{Precision + Recall}
\]

#### Result

\[
F1 = 0.8046
\]

Interpretation:

The model achieves a good balance between Precision and Recall.

---

## Scatter Plot Analysis

A scatter plot was created using:

- X-axis: Actual House Prices
- Y-axis: Predicted House Prices

A reference line:

\[
y = x
\]

was added.

#### Interpretation

- Points close to the line indicate accurate predictions.
- Points far from the line indicate larger prediction errors.
- A tighter clustering around the line suggests better model performance.

---

## Predictions for New Data Points

### Data Point 1

| Feature | Value |
|----------|--------:|
| housing_median_age | 41 |
| total_rooms | 880 |
| total_bedrooms | 129 |
| population | 322 |
| households | 126 |
| median_income | 8.3252 |

#### Predicted Price

\[
\$426,695.76
\]

---

### Data Point 2

| Feature | Value |
|----------|--------:|
| housing_median_age | 28 |
| total_rooms | 960 |
| total_bedrooms | 160 |
| population | 310 |
| households | 150 |
| median_income | 4.5001 |

#### Predicted Price

\[
\$224,966.17
\]

---

## Observations

1. Median Income has the largest coefficient and appears to be the most influential feature.

2. Housing Median Age also contributes positively to house value.

3. The model achieved an R² score of 0.5966, indicating moderate predictive capability.

4. The model correctly classified approximately 79% of block groups into high-value and low-value categories.

5. The higher median income of Data Point 1 resulted in a significantly larger predicted house value compared to Data Point 2.

6. Since the dataset represents census block groups rather than individual houses, predictions correspond to the median house value of an area rather than the value of a single house.

---

## Conclusion

A Multivariate Linear Regression model was successfully trained on the California Housing dataset using six selected features. The model demonstrated moderate predictive performance with an R² score of 0.5966 and produced reasonable house value estimates for unseen data. Both regression and classification-based evaluation metrics were analyzed, and the results indicate that median income is the strongest predictor of median house value within the selected feature set.


