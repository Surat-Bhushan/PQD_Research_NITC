import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data Setup

X = np.array([0, 0.18, 0.26, 0.57, 0.48,
              0.62, 0.44, 0.55, 0.89,
              1.0, 0.92])

np.random.seed(0)

Y = 0.5 * X + 1 + 0.001 * np.random.randn(len(X))

# Gradient Descent Function

def gradient_descent(X, Y, learning_rate,
                     max_iterations=1000,
                     tolerance=1e-8):

    n = len(X)

    m = 0
    b = 0

    losses = []

    previous_loss = float('inf')
    convergence_iteration = max_iterations

    for iteration in range(max_iterations):

        # Predictions
        Y_pred = m * X + b

        # Mean Squared Error Loss
        loss = np.mean((Y - Y_pred) ** 2)
        losses.append(loss)

        # Check convergence
        if abs(previous_loss - loss) < tolerance:
            convergence_iteration = iteration + 1
            break

        previous_loss = loss

        # Gradients
        dm = (-2 / n) * np.sum(X * (Y - Y_pred))
        db = (-2 / n) * np.sum(Y - Y_pred)

        # Update parameters
        m = m - learning_rate * dm
        b = b - learning_rate * db

    return m, b, losses, convergence_iteration


# Train for Different Learning Rates

learning_rates = [0.01, 0.05, 0.1, 0.5]

results = {}

plt.figure(figsize=(8, 5))

for lr in learning_rates:

    m, b, losses, conv_iter = gradient_descent(
        X, Y, lr
    )

    final_loss = losses[-1]

    results[lr] = {
        "m": m,
        "b": b,
        "losses": losses,
        "final_loss": final_loss,
        "convergence_iteration": conv_iter
    }

    print(f"\nLearning Rate = {lr}")
    print(f"Weight (m) = {m:.6f}")
    print(f"Bias (b) = {b:.6f}")
    print(f"Final Loss = {final_loss:.10f}")
    print(f"Converged After = {conv_iter} iterations")

    plt.plot(losses, label=f"α = {lr}")

# Loss Curves

plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")
plt.legend()
plt.grid(True)


# Select Best Learning Rate

best_lr = min(
    results,
    key=lambda lr: (
        results[lr]["final_loss"],
        results[lr]["convergence_iteration"]
    )
)

print("\nBest Learning Rate =", best_lr)

m_best = results[best_lr]["m"]
b_best = results[best_lr]["b"]

print(f"Best Weight = {m_best:.6f}")
print(f"Best Bias = {b_best:.6f}")

# Predictions using Gradient Descent

x1 = 0.3
x2 = 0.75

y1 = m_best * x1 + b_best
y2 = m_best * x2 + b_best

print("\nPredictions using Gradient Descent")
print(f"x = 0.30 --> y = {y1:.6f}")
print(f"x = 0.75 --> y = {y2:.6f}")

# Scatter Plot + Regression Line

plt.figure(figsize=(8, 5))

plt.scatter(X, Y, label="Data Points")

plt.plot(
    X,
    m_best * X + b_best,
    label="Regression Line"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)


# Sklearn Verification

X_sklearn = X.reshape(-1, 1)

model = LinearRegression()

model.fit(X_sklearn, Y)

m_sk = model.coef_[0]
b_sk = model.intercept_

print("\nSklearn Results")
print(f"Weight = {m_sk:.6f}")
print(f"Bias = {b_sk:.6f}")

# Predictions using sklearn

y1_sk = model.predict([[0.3]])[0]
y2_sk = model.predict([[0.75]])[0]

print("\nPredictions using Sklearn")
print(f"x = 0.30 --> y = {y1_sk:.6f}")
print(f"x = 0.75 --> y = {y2_sk:.6f}")

# Comparison

print("\nComparison")

print(f"Gradient Descent Weight = {m_best:.6f}")
print(f"Sklearn Weight = {m_sk:.6f}")

print(f"\nGradient Descent Bias = {b_best:.6f}")
print(f"Sklearn Bias = {b_sk:.6f}")

plt.show()
