#!/usr/bin/env python
# coding: utf-8

# 1. Using a graph to illustrate slope and intercept, define basic linear regression.
# 

# The slope indicates the steepness of a line and the intercept indicates the location where it intersects an axis. The slope and the intercept define the linear relationship between two variables, and can be used to estimate an average rate of change.

# 2. In a graph, explain the terms rise, run, and slope.
# 

# Rise: The vertical distance between two points on a line or curve.
# Run: The horizontal distance between two points on a line or curve.
# Slope: The ratio of the vertical change (rise) to the horizontal change (run) between two points on a line or curve.

# 3. Use a graph to demonstrate slope, linear positive slope, and linear negative slope, as well as the different conditions that contribute to the slope.
# 

# In[2]:


import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y_positive = [0, 2, 4, 6, 8, 10]
y_negative = [0, -2, -4, -6, -8, -10]

plt.plot(x, y_positive, label='Positive Slope')
plt.plot(x, y_negative, label='Negative Slope')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Slope: Positive and Negative')
plt.grid(True)
plt.legend()
plt.show()


# 4. Use a graph to demonstrate curve linear negative slope and curve linear positive slope.
# 

# In[7]:


import numpy as np
x = np.linspace(-5, 5, 100)
y_positive = x ** 2
y_negative = -x ** 2
plt.plot(x, y_positive, label='Positive Slope')
plt.plot(x, y_negative, label='Negative Slope')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Linear Slope: Positive and Negative')
plt.grid(True)
plt.legend()
plt.show()


# 5. Use a graph to show the maximum and low points of curves.
# 

# In[8]:


x = np.linspace(-5, 5, 100)
y = x ** 2
max_point = (x[np.argmax(y)], np.max(y))
min_point = (x[np.argmin(y)], np.min(y))
plt.plot(x, y, label='Curve')
plt.plot(max_point[0], max_point[1], 'ro', label='Maximum')
plt.plot(min_point[0], min_point[1], 'go', label='Minimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Maximum and Minimum Points of a Curve')
plt.grid(True)
plt.legend()
plt.show()


# 6. Use the formulas for a and b to explain ordinary least squares.
# 

# In[9]:


#if you're looking for a simpler approach to calculate the coefficients (a and b) using ordinary least squares (OLS) in Python, you can use the numpy library.

#for example
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
a, b = np.polyfit(x, y, 1)
print("Estimated coefficients:")
print("a =", a)
print("b =", b)


# 7. Provide a step-by-step explanation of the OLS algorithm.
# 

# Import the necessary libraries (numpy and statsmodels.api).
# Prepare the data by defining the independent variable x and the dependent variable y.
# Add a constant term to the independent variable using sm.add_constant(x).
# Fit the linear regression model using OLS by creating an OLS model with sm.OLS(y, x) and using model.fit().
# Retrieve the estimated coefficients using results.params, where index 0 corresponds to the y-intercept (a) and index 1 corresponds to the slope (b).

# 8. What is the regression's standard error? To represent the same, make a graph.
# 

# The regression's standard error, also known as the residual standard error (RSE) or root mean squared error (RMSE), measures the average distance between the observed values and the predicted values in a regression model. It quantifies the amount of variation that is not explained by the regression equation.
# 
# To represent the regression's standard error on a graph in Python, you can plot the residuals, which are the differences between the observed values and the predicted values. The spread of the residuals can give an indication of the standard error

# 9. Provide an example of multiple linear regression.
# 

#  Prediction of CO2 emission based on engine size and number of cylinders in a car.

# 10. Describe the regression analysis assumptions and the BLUE principle.
# 

# Regression Analysis Assumptions:
# 
# Linearity: The relationship between the variables is linear.
# Independence: The observations are independent of each other.
# Homoscedasticity: The variance of the errors is constant across all levels of the independent variable(s).
# Normality: The errors (residuals) follow a normal distribution.
# No Multicollinearity: There is no perfect or near-perfect linear relationship between the independent variables.
#     
#     
# BLUE Principle (Best Linear Unbiased Estimator):
# 
# The OLS method provides the best estimates among all unbiased linear estimators.
# It minimizes the variance of the estimated coefficients, making them efficient and optimal.
# Python's statsmodels library implements the OLS method, adhering to the BLUE principle.

# 11. Describe two major issues with regression analysis.
# 

# Assumption Violations:
# 
# Regression analysis assumes certain conditions like linearity, independence, and normality. If these assumptions are violated, the regression results may be unreliable or misleading.
# Examples of violations include nonlinear relationships between variables, correlated observations, and non-normal distribution of residuals.
# 
# 
# Outliers and Influential Points:
# 
# Outliers are extreme observations that deviate significantly from the rest of the data, while influential points strongly influence the regression results.
# Outliers and influential points can distort the regression line and impact the accuracy and generalizability of the model.

# 12. How can the linear regression model's accuracy be improved?
# 

# emove irrelevant or redundant features.
# Handle outliers by using statistical techniques or visualizations.
# Normalize the input features to ensure they have similar scales.
# Consider using nonlinear transformations for features or the target variable.
# Handle missing data appropriately through removal or imputation.

# 13. Using an example, describe the polynomial regression model in detail.
# 

# Imports the necessary libraries.
# Creates the dataset with hours studied (X) and exam scores (Y).
# Uses PolynomialFeatures to transform the features into polynomial terms.
# Fits a linear regression model on the transformed features.
# Visualizes the original data points and the polynomial regression line.

# 14. Provide a detailed explanation of logistic regression.
# 

# Imports the necessary library LogisticRegression from sklearn.linear_model.
# Creates the dataset with input features (X) and binary target variable (Y).
# Fits the logistic regression model using the fit() function.
# Predicts probabilities for new instances using the predict_proba() function.
# Predicts classes (0 or 1) for new instances using the predict() function.

# 15. What are the logistic regression assumptions?
# 

# Binary Outcome: The dependent variable should be binary or dichotomous.
# 
# Linearity of the Logit: The logit transformation of the probability of the positive outcome should have a linear relationship with the input features.
# 
# Independence of Observations: Observations should be independent of each other.
# 
# No Multicollinearity: There should be little to no multicollinearity among the input features.
# 
# Large Sample Size: A relatively large sample size is preferred for accurate parameter estimates and reliable inferences.
# 
# No Outliers: The presence of extreme outliers should be minimized as they can impact the results.

# 16. Go through the details of maximum likelihood estimation.
# 

# Maximum Likelihood Estimation (MLE) is a statistical method used to find the best parameter values for a model that maximize the likelihood of observing the given data. In Python, you can use libraries like SciPy or specialized packages like StatsModels and scikit-learn to perform MLE for various models, including logistic regression. The process involves defining a likelihood function, taking the logarithm to simplify calculations, using optimization algorithms to find the parameter values that maximize the log-likelihood, and obtaining the estimated parameter values. MLE helps find the most likely parameter values that explain the observed data.

# In[ ]:




