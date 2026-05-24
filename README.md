# Task 2- Univariate Regression Problem
## Problem Statement
(i) Implementing Linear Regression using Gradi-
ent Descent
Data Setup
You are given the following data points for the independent variable X and
dependent variable Y:
X = [0,0.18,0.26,0.57,0.48,0.62,0.44,0.55,0.89,1.0,0.92]
Y = 0.5· X+ 1 + 0.001· randn(11,1)
where some random noise is added to the Y values.
Linear Regression Model
Implement the univariate linear regression model:
y= mx+ b
where m is the slope (weight), and b is the intercept (bias).
Loss Function
Define the Mean Squared Error (MSE) loss function:

<img width="165" height="38" alt="Screenshot 2026-05-24 at 10 35 03 PM" src="https://github.com/user-attachments/assets/28b62ef7-9803-4b91-8a45-3d07563aef27" />

where yi are the observed values, xi are the feature values, and m, b are the
model parameters.

<img width="298" height="221" alt="Screenshot 2026-05-24 at 10 35 48 PM" src="https://github.com/user-attachments/assets/c6f4763c-e37a-4c83-a9df-14ef511ce2fe" />

Training Process
• Initialize m= 0 and b= 0, and choose several values for the learning rate
α (e.g., 0.01, 0.05, 0.1, 0.5, etc.).
• Run the gradient descent for a set number of iterations (e.g., 1000) or
until convergence.
• Record the loss at each iteration.
Convergence Visualization
• Plot the loss curve for each learning rate.
• Show how the convergence rate changes with different values of α. For
example:
– Small α may show slow convergence.
– Large α may cause overshooting.
– An optimal α will show faster convergence.
Print the Weight and Bias for Each Learning Rate
Print the learned weight and bias after running the gradient descent for each
learning rate.
(ii) Prediction at Specific Values of x
After training the model using gradient descent, predict the values of y corre-
sponding to:
x= 0.3
x= 0.75
Use the model parameters mand bobtained from the gradient descent with the
best learning rate to make these predictions.
(iii) Verify the Result using Sklearn Linear Re-
gression
Sklearn Linear Regression
• Use the LinearRegression model from the sklearn.linear model li-
brary.
• Train the linear regression model using the same X and Y data points.
• Extract the coefficients (weights) and intercept (bias) obtained from sklearn.linear model.LinearRegre
after fitting the model.
Comparison
• Compare the learned weight (m) and bias (b) obtained through gradient
descent with those from Sklearn’s linear regression.
• Verify if the predictions made in part (ii) using the gradient descent model
are similar to those made using Sklearn’s model.

# Experimental Observations and Parameter Tuning

During the implementation, multiple experiments were performed to understand the effect of the learning rate, convergence tolerance, and iteration limit on Gradient Descent.

---

## Experiment 1

**Maximum Iterations:** 1000  
**Tolerance:** 1e-8

### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000212471 | 1000 (did not converge) |
| 0.05 | 0.0000014215 | 442 |
| 0.1 | 0.0000010616 | 246 |
| 0.5 | 0.0000007697 | 61 |

### Observation

- Learning rate 0.5 produced the lowest loss and converged fastest.
- Learning rate 0.01 failed to converge within 1000 iterations.
- This raised the question of whether 0.01 was actually inferior or simply required more iterations.

---

## Experiment 2

**Maximum Iterations:** 5000  
**Tolerance:** 1e-8

### Observation

After increasing the iteration limit, the learning rate 0.01 still stopped before reaching a loss comparable to the other learning rates.

This suggested that:

- Increasing the iteration limit alone was not sufficient.
- The convergence criterion based on tolerance was causing optimization to terminate before reaching a sufficiently good optimum.
- The tolerance value of 1e-8 appeared to be too loose for smaller learning rates.

---

## Experiment 3

**Maximum Iterations:** 5000  
**Tolerance:** 1e-12

### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000007109 | 4948 |
| 0.05 | 0.0000007106 | 1104 |
| 0.1 | 0.0000007106 | 576 |
| 0.5 | 0.0000007105 | 125 |

### Observation

- All learning rates converged to nearly the same minimum loss.
- Learning rate 0.01 was not inherently worse; it simply required significantly more iterations.
- The stricter tolerance prevented premature stopping and allowed optimization to approach the optimum more closely.
- Learning rate 0.5 still achieved the optimum much faster than the other learning rates.

---

## Experiment 4 (Final Configuration)

**Maximum Iterations:** 1000  
**Tolerance:** 1e-12

### Results

| Learning Rate | Final Loss | Convergence Iterations |
|--------------|------------|-----------------------|
| 0.01 | 0.0000212471 | 1000 (did not converge) |
| 0.05 | 0.0000007108 | 1000 |
| 0.1 | 0.0000007106 | 576 |
| 0.5 | 0.0000007105 | 125 |

### Observation

- This configuration follows the assignment requirement of running for **1000 iterations or until convergence**.
- Learning rate 0.01 still required more than 1000 iterations and therefore did not fully converge.
- Learning rates 0.05, 0.1, and 0.5 all reached nearly identical optimum losses.
- Learning rate 0.5 achieved the same optimum while requiring the fewest iterations.

---

# Final Conclusion

The final implementation uses:

- **Maximum Iterations = 1000**
- **Tolerance = 1e-12**

This configuration was chosen because:

1. It satisfies the assignment requirement of running for **1000 iterations or until convergence**.
2. It avoids the premature stopping observed with a tolerance of 1e-8.
3. It allows converging learning rates to approach the optimum more accurately.
4. It produces results that closely match Scikit-Learn's `LinearRegression` model.

Among the tested learning rates, **0.5 was selected as the best learning rate** because it achieved essentially the same optimum as the smaller learning rates while converging in the fewest iterations.
