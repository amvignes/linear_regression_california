California Housing Price Prediction with Machine Learning
This project explores two approaches to predict California housing prices using machine learning: a basic Linear Regression and an optimized Random Forest Regressor.
1. linearregressioncalifornia.py: 
This script implements a Linear Regression model to predict house prices based on the California Housing dataset. Linear regression is a straightforward method to model the relationship between predictors and the target variable.
Key Steps:
* Load and preprocess the dataset.
* Split data into training and test sets.
* Train a linear regression model.
* Evaluate the model using Mean Squared Error (MSE).
Result:
* MSE: 0.5559 — The linear model provides a basic understanding but lacks accuracy due to its simplicity.
2. linearregressionforestregressor.py
This script uses a Random Forest Regressor to improve predictions by capturing more complex relationships within the data.
Key Steps:
* Preprocess data.
* Train a Random Forest model with 100 trees.
* Evaluate the model’s performance with MSE.
Result:
* MSE: 0.2556 — A significant improvement over the linear model, as the Random Forest captures more complexity.
Conclusion
This project demonstrates how a simple Linear Regression model can be enhanced by using a Random Forest Regressor. The optimized model reduces prediction error by leveraging the strength of ensemble learning.


Learning and Optimization
* Linear Regression: A quick and effective approach to understand the basics of modeling in machine learning.
* Random Forest Regressor: A step towards optimization by harnessing the power of ensemble models to capture data complexity.
* Continuous Optimization: By experimenting with hyperparameters or more complex models like Gradient Boosting, we can continue to reduce error and achieve even better results.