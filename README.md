# Machine Learning
# Explination of Support Vector Machines:
Support vector machines is one of the most powerful algorithm,because it is mainly used for the classifying data and it also tries to find best possible boundaries by maintaining
largest distance from data points.
### There are two ways to calculate error in svms:
1.Classification error <br />
2.Margin error. <br />
Error = Classification error + Margin error <br />
![svm](https://user-images.githubusercontent.com/45037048/118471324-44bf6900-b725-11eb-8baf-f375422f145e.png)

1.Classification error : When a missclassification occurs, it is because a given point is on the wrong side of the separating hyperplane, and that's called a classification error.<br />
2.Margin error:Whenever a point is inside the margin, that counts as a margin error.
## Assignment Explination 
Libraries used 
```
1.Pandas
2.Numpy
3.Matplotlib
4.Seaborn 
5.Sklearn

```
The very first step in any project is loading dataset. In this Project we use inbuilt **Breast Cancer** dataset by importing **sklearn.datsets import load_breast_cancer** and some basic methods like **head()** shows top 5 rows in dataframe

**Exploratory Data Analysis (EDA)** <br /> 
It is a way of visualizing, summarizing and interpreting the information that is hidden in rows and column format.It performs to define and refine our important features variable selection, that will be used in our model.We initially make several hypothesis by looking at the data before we hit the modelling.

**Split of DataSet** <br /> 
``` train_test_split() ``` is used for splitting the data points for testing and training. The ideal percentage for training and testing is 70% and 30 % respectively.
In this project we implement **SVM** we need to import ``` from sklearn.svm import SVC ```
**Fitting the model**
Error is calculated by sum of Classification error and Margin error <br />
![svm2](https://user-images.githubusercontent.com/45037048/118478650-c3200900-b72d-11eb-85f6-6ed744757af3.png)



