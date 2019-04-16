#import the dataset
dataset=read.csv('Salary_Data.csv')

#split the dataset into training set and test set
library(caTools)
set.seed(123)
split=sample.split(dataset$salary,splitRatio = 2/3)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split(FALSE))
