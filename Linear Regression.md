# Linear Regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task.Linear regression is used for finding linear relationship between target and one or more predictors.<br />
![linear](https://user-images.githubusercontent.com/45037048/118625998-1f476380-b7e8-11eb-9ee0-f761465fad45.png)
<br />
Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output).
## Explanation of Linear Regression using Housing Dataset
### Import Libraries :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
The very first step in any machine learning project is importing dataset.In this project we use Housing dataset .Import the dataset as shown below
```
df = pd.read_csv("USA_Housing.csv")
```
We can get more information about dataset and values and columns inside dataset by ``` head(), info(), describe() ``` methods

### Exploratory Data Ananlysis :
It is a way of visualizing, summarizing and interpreting the information that is hidden in rows and column format.It performs to define and refine our important features variable selection, that will be used in our model.We initially make several hypothesis by looking at the data before we hit the modelling.<br />
``` 
sns.pairplot(df)
```
<seaborn.axisgrid.PairGrid at 0x7fc4621c8950>
![download](https://user-images.githubusercontent.com/45037048/118628693-7bab8280-b7ea-11eb-91d9-38d02b674448.png)

```
sns.distplot(df['Price'])
```
![download (1)](https://user-images.githubusercontent.com/45037048/118628809-967df700-b7ea-11eb-911b-ef532be1500c.png)

```
df.corr()
```
![image](https://user-images.githubusercontent.com/45037048/118630212-f4f7a500-b7eb-11eb-84a8-e443cda9d236.png)


``` sns.heatmap(df.corr() , annot=True) ``` <br />
![download (2)](https://user-images.githubusercontent.com/45037048/118628888-a990c700-b7ea-11eb-8c8b-c856f782adf5.png)

## Split of DataSet
```train_test_split()``` is used for splitting the data points for testing and training. The ideal percentage for training and testing is 70% and 30 % respectively.
```x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']
```
Now import library ```from sklearn.model_selection import train_test_split```
We split the test and train data by using the code 
```x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101)``` Here we used 40% of data for testing and 60% for training.
### Training our data using linear regression 
First we need to import linear regression by 
```
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
```
Now we need to fit our model using fit()
```lm.fit(x_train,y_train)```
**LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)**
Here training and fitting our model is done.
### Predictions :
In this predictions we will find how good our model is we can predict values for our testing data (x_test).
```predictions = lm.predict(x_test)```
Now we can visualize and we can see if our model is performing well or not
```plt.scatter(y_test,predictions)```
![download (3)](https://user-images.githubusercontent.com/45037048/118632932-8d8f2480-b7ee-11eb-8b51-52918bd95e14.png)

### Evaluation :
Evaluation will describe how good our model is performing. In this evaluation part we will find 
```
mae = mean absolute error
mse = mean square error
rmse = root of mean square error
```
We need to import ```from sklearn import metrics```

```metrics.mean_absolute_error(y_test,predictions)```
82288.22251914928
```metrics.mean_squared_error(y_test,predictions)```
10460958907.208244
```np.sqrt(metrics.mean_squared_error(y_test,predictions))```
102278.82922290538







 









  
