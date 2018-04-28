# Random Forest Algorithm
The Random Forest Algorithm uses the Gini Index measure to analyse numerical data. Categorical data is handled by a one-hot encoding transformation, creating in this way a dummy variable for each category. This guarantees that the binary splits will always occur. The function returns: 1) The Forest - A set of decision trees and each one of their set of rules. 2) The list of observations used to plant a decision tree (about 2/3 of the total observations for each tree).

* Xdata = Dataset Attributes

* ydata = Dataset Target

* cat_missing = "none", "missing", "most", "remove" or "probability". If "none" is selected then nothing will be done if there are missing categorical values. If "missing" is selected then the missing categorical values will be replaced by a new category called Unkown. If "most" is selected then the categorical missing values will be replaced by the most popular category of the attribute. If "remove" is selected then the observation with missing categorical values will be deleted from the dataset. If "probability" is selected then the categorical missing values will be randomly replaced by a category based on the category distribution of the attribute.

* num_missing = "none", "mean", "median", "remove" or "probability". If "none" is selected then nothing will be done if there are missing numerical values. If "mean" is selected then the missin gnumerical values will be replaced by the attribute mean. If "median" is selected then the numerical missing values will be replaced by the attribute median. If "most" is selected then the numerical missing values will be replaced by the most popular value of the attribute. If "remove" is selected then the observation with missing numerical values will be deleted from the dataset. If "probability" is selected then the numerical missing values will be randomly replaced by a value based on the numerical distribution of the attribute.

* forest_size = The total number of decision trees that will be planted to form the forest. The default value is 5.

* m_attributes = Each decision tree is planted using a random subset of attributes. The default value of the subset size is the nearest integer of the square root of the total number of attributes.

* An out-of-bag (oob) function is given - oob_error_estimates(model, Xdata, ydata) - which calculates the Random Forest estimated oob error and also returns a confusion matrix of the target attribute.

* Finnaly a prediction function - prediction_dt_rf(model, Xdata) - is also included.
