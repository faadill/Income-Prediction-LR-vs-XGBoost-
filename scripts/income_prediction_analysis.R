## =============================================================
## Income Prediction using Machine Learning Models
## Author: Mohammed Fadil
## Description: Predicts whether an individual's annual income exceeds
##              $50,000 using the UCI/Census Income dataset, comparing
##              Logistic Regression against XGBoost.
## =============================================================

library(tidyverse)
library(caret)
library(pROC)
library(xgboost)
library(Matrix)
library(ggplot2)
library(corrplot)

## -------------------------------------------------------------
## 1. Load & inspect data
## -------------------------------------------------------------
indata <- read.csv("data/income.csv")

dim(indata)
str(indata)
head(indata, 10)
summary(indata)

## -------------------------------------------------------------
## 2. Exploratory Data Analysis
## -------------------------------------------------------------

# Income by education level
ggplot(indata, aes(x = education, fill = Income)) +
  geom_bar(position = "fill") +
  labs(title = "Income Distribution by Education Level",
       x = "Education", y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45))

# Class balance
table(indata$Income)
prop.table(table(indata$Income))

ggplot(indata, aes(x = Income, fill = Income)) +
  geom_bar() +
  labs(title = "Income Class Distribution",
       x = "Income Category", y = "Count") +
  theme_minimal()

# Missing value check — missing values are encoded as " ?" rather than NA
colSums(is.na(indata))
colSums(indata == " ?")

# Correlation among numeric predictors
num_data <- select_if(indata, is.numeric)
corr_matrix <- cor(num_data)
corrplot(corr_matrix, method = "color", addCoef.col = "red")

## -------------------------------------------------------------
## 3. Data Preprocessing
## -------------------------------------------------------------

# Recode "?" as NA, then explicitly label as its own "NA" factor level
# (kept rather than dropped, to avoid unnecessary data loss)
indata[indata == " ?"] <- NA
indata$workclass[is.na(indata$workclass)] <- "NA"
indata$occupation[is.na(indata$occupation)] <- "NA"

# Convert character columns to factors
indata[] <- lapply(indata, function(x) {
  if (is.character(x)) as.factor(x) else x
})

# One-hot encode categorical predictors
dummies <- dummyVars(Income ~ ., data = indata)
data_encoded <- data.frame(predict(dummies, newdata = indata))
data_encoded$Income <- indata$Income

# 80:20 train/test split
set.seed(123)
train_index <- createDataPartition(data_encoded$Income, p = 0.8, list = FALSE)
train_data <- data_encoded[train_index, ]
test_data  <- data_encoded[-train_index, ]

## -------------------------------------------------------------
## 4. Logistic Regression
## -------------------------------------------------------------

# education_num dropped: redundant with education, reduces multicollinearity
train_data$education_num <- NULL
test_data$education_num  <- NULL

log_model <- glm(Income ~ ., data = train_data, family = "binomial")

log_probs <- predict(log_model, newdata = test_data, type = "response")
log_pred  <- ifelse(log_probs > 0.5, ">50K", "<=50K")
log_pred  <- factor(log_pred, levels = levels(test_data$Income))

# Evaluation
confusionMatrix(log_pred, test_data$Income)
mean(log_pred == test_data$Income)                       # Accuracy

precision <- posPredValue(log_pred, test_data$Income)
recall    <- sensitivity(log_pred, test_data$Income)
f1_score  <- 2 * ((precision * recall) / (precision + recall))
precision; recall; f1_score

roc_curve <- roc(test_data$Income, log_probs)
auc(roc_curve)
plot(roc_curve, col = "green")

## -------------------------------------------------------------
## 5. XGBoost
## -------------------------------------------------------------

train_xgb <- train_data
test_xgb  <- test_data
train_xgb$Income <- as.numeric(train_xgb$Income == ">50K")
test_xgb$Income  <- as.numeric(test_xgb$Income == ">50K")

x_train <- as.matrix(train_xgb[, -ncol(train_xgb)])
y_train <- train_xgb$Income
x_test  <- as.matrix(test_xgb[, -ncol(test_xgb)])
y_test  <- test_xgb$Income

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)

xgb_model <- xgb.train(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 7,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  data = dtrain,
  nrounds = 250,
  verbose = 0
)

xgb_probs <- predict(xgb_model, dtest)
xgb_pred  <- ifelse(xgb_probs > 0.5, 1, 0)

# Evaluation
cm_xgb <- confusionMatrix(factor(xgb_pred), factor(y_test))
cm_xgb

accuracy_xgb  <- mean(xgb_pred == y_test)
precision_xgb <- posPredValue(factor(xgb_pred), factor(y_test))
recall_xgb    <- sensitivity(factor(xgb_pred), factor(y_test))
f1_xgb        <- 2 * ((precision_xgb * recall_xgb) / (precision_xgb + recall_xgb))
accuracy_xgb; precision_xgb; recall_xgb; f1_xgb

roc_xgb <- roc(y_test, xgb_probs)
auc(roc_xgb)
plot(roc_xgb, col = "red")

## -------------------------------------------------------------
## 6. Feature Importance
## -------------------------------------------------------------

# --- Logistic Regression: absolute coefficient magnitude ---
lr_importance <- summary(log_model)$coefficients
lr_importance <- data.frame(
  Feature    = rownames(lr_importance),
  Estimate   = lr_importance[, "Estimate"],
  Importance = abs(lr_importance[, "Estimate"])
)
lr_importance <- lr_importance[lr_importance$Feature != "(Intercept)", ]
lr_importance <- lr_importance[lr_importance$Importance < 5, ]   # drop unstable extreme coefficients

top_lr <- head(lr_importance[order(-lr_importance$Importance), ], 10)
top_lr$Direction <- ifelse(top_lr$Estimate > 0, "Positive", "Negative")
print(top_lr[, c("Feature", "Estimate", "Direction")])

ggplot(top_lr, aes(x = reorder(Feature, Estimate), y = Estimate, fill = Direction)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Positive" = "steelblue", "Negative" = "red")) +
  labs(title = "Logistic Regression Feature Importance",
       x = "Features", y = "Coefficient") +
  theme_minimal()

# --- XGBoost: gain-based importance ---
importance_matrix <- xgb.importance(feature_names = colnames(x_train), model = xgb_model)
top_xgb <- importance_matrix[1:10, ]

xgb.plot.importance(top_xgb, col = "steelblue", main = "XGBoost Feature Importance")
