# house_prices
 
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='center'>

<br><br>

# Project Title

Exploratory data analysis of the California housing dataset and fitting a Linear model for prediction of the house prices


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

## Evaluation and Results

#### Histogram analysis on the input features
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)

#### Visualizing the geographical data
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/corr2.png)

#### Correlation between the house price and median house income
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/corr1.png)


## Metrics of the Linear Regressor model used.

| Metric        | Value         |
| ------------- | ------------- |
| R2 Score      | 0.59          |
| MSE           | 0.54         |
                 

The above quant results show that we have an error of 54,000 USD betweeen the prediction and the actual house price.

## Key Takeaways

This task is evidently a classic example of supervised learning, as the model can be trained using labeled examples, with each instance providing the expected output, such as the median housing price of a district. It falls under the category of regression tasks since the model aims to predict a numerical value. More specifically, it constitutes a multiple regression problem as it utilizes multiple features, such as district population and median income, to make predictions. Additionally, it is a univariate regression problem since the objective is to predict a single value for each district. If the goal were to predict multiple values per district, it would be classified as a multivariate regression problem. Furthermore, there is no continuous stream of data entering the system, no urgent need to adapt to changing data quickly, and the dataset is small enough to be accommodated in memory, thus making plain batch learning sufficient.


## How to Run

The code is built on Jupyter notebook

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

We can do the following and try to get better results

- Try more models

- Wrapped Based Feature Selection


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

#### How does the linear regression model work?

 **Linear regression** is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (predictors) by fitting a linear equation to observed data. The model assumes that this relationship is approximately linear, meaning that changes in the independent variables are associated with a constant change in the dependent variable. **The goal is to find the best-fitting line that minimizes the difference between the actual and predicted values**, typically achieved by minimizing the sum of squared differences (least squares method). Once the model is trained, it can be used to predict the values of the **dependent** variable based on new values of the **independent** variables.

#### How do you train the model on a new dataset?

**Data Preparation**: Organize your dataset into features (independent variables) and the target variable (dependent variable).

**Split Data**: Divide the dataset into two subsets: training data and testing data. The training set is used to train the model, while the testing set is used to evaluate its performance.

**Model Training**: Use the training data to fit the linear regression model. This involves finding the coefficients (weights) that minimize the difference between the predicted values and the actual values of the target variable.

**Model Evaluation**: Assess the performance of the trained model using the testing data. Common evaluation metrics for linear regression include mean squared error (MSE), root mean squared error (RMSE), and coefficient of determination (R-squared).

**Fine-tuning (Optional)**: If the model performance is not satisfactory, you can fine-tune hyperparameters or consider feature engineering to improve its accuracy.

**Prediction**: Once the model is trained and evaluated satisfactorily, you can use it to make predictions on new data by inputting the values of the independent variables into the model equation.

**Deployment**: Finally, if the model performs well on new data, it can be deployed into production for making real-world predictions.

#### What is the California Housing Dataset?

The California Housing Dataset is a widely used dataset in machine learning and statistics. It contains data related to housing in California, particularly focusing on the state's census districts. The dataset typically includes features such as median house value, median income, housing median age, average number of rooms, average number of bedrooms, population, and geographical information like latitude and longitude.

The main objective of using this dataset is often to build predictive models, such as regression models, to predict the median house value based on other attributes present in the dataset. It's commonly used for practicing and learning regression techniques, particularly in the context of supervised learning.

This dataset has been used in various research studies, educational settings, and competitions due to its relevance to real-world problems and its accessibility for educational purposes.
## Acknowledgements


 - ![Hands on machine learning - by Geron](https://github.com/vasanthgx/house_prices/blob/main/images/bookcover.jpg)
 - [github repo for handsonml-3](https://github.com/ageron/handson-ml3)
 - [EDA on the California housing dataset - kaggle notebook](https://www.kaggle.com/code/olanrewajurasheed/california-housing-dataset)
 


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)