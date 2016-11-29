#install.packages("caret", dependencies = TRUE)
#install.packages("randomForest")
#install.packages("fields")
library(caret)
library(randomForest)
library(fields)

trainingData <- read.table("train.csv", sep = ",", header = TRUE)
testingData <- read.table("test.csv", sep = ",", header = TRUE)

head(trainingData)
head(testingData)
summary(trainingData)
summary(testingData)

#including the Pclass, Sex, SibSp, Parch variables
table(trainingData[,c("Survived","Pclass")])
table(trainingData[,c("Survived","Sex")])
table(trainingData[,c("Survived","SibSp")])
table(trainingData[,c("Survived","Parch")])
table(trainingData[,c("Survived","Embarked")])

#excluding age as it doesn't have significant impact and it has many NAs
bplot.xy(trainingData$Survived, trainingData$Age)
summary(trainingData$Age)

#including fare as it doesn't have NAs and provides useful predictive capabilities
bplot.xy(trainingData$Survived, trainingData$Fare)
summary(trainingData$Fare)

# preparing the dataset
trainingData$Survived <- factor(trainingData$Survived)

set.seed(42)

# using the random forest classifier from the caret package
rf_model <- train(Survived ~ Pclass + Sex + SibSp + Embarked + Parch + Fare, 
                  data = trainingData,
                  method = "rf",
                  trControl = trainControl(method = "cv", number = 5))

#imputing values in the testSet which give errors due to NAs in prediction
testingData$Fare <- ifelse(is.na(testingData$Fare),mean(testingData$Fare, na.rm = T),testingData$Fare)

testingData$Survived <- predict(rf_model, newdata = testingData)

final_result <- testingData[,c("PassengerId","Survived")]

write.table(final_result, file="final_results.csv",col.names = TRUE, row.names = TRUE, sep = ",")
