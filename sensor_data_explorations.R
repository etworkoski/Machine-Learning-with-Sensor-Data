#Data read-in
library(tidyverse)
library(caret)
library(knitr)
library(parallel)
install.packages("doParallel")
library(doParallel)
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./training_data.csv", method = "curl")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "./testing_data.csv", method = "curl")

training <- read.csv(file = "training_data.csv", na.strings = c("","NA"))
testing <- read.csv(file = "testing_data.csv", na.strings = c("","NA"))
varnames <- colnames(training)

#Cleaning and exploration
find_missing <- function(var){
    sum(is.na(training[,var]))/length(training[,var])
}

missing_var_summary <- as.data.frame(cbind(varnames, prop_missing = sapply(varnames, find_missing)))
unique(missing_var_summary$prop_missing)
nonmissing_vars <- missing_var_summary %>% subset(prop_missing == 0) 
training_nonmissing <- training[,nonmissing_vars$varnames]

training_nonmissing$num_window <- as.factor(training_nonmissing$num_window)
training_nonmissing$classe <- as.factor(training_nonmissing$classe)
training_nonmissing$user_name <- as.factor(training_nonmissing$user_name)
training_model <- training_nonmissing[,-c(1,3:7)]

small_train <- subset(training_nonmissing, num_window == 1)

training_nonmiss_pivot <- training_nonmissing %>%
                            pivot_longer(cols = -c(1:7,60),
                                         names_to = "variable",
                                         values_to = "value")
var_subset_1 <- grepl("belt", training_nonmiss_pivot$variable)
training_belt <- training_nonmiss_pivot[var_subset_1,]
var_subset_2 <- grepl("forearm", training_nonmiss_pivot$variable)
training_forearm <- training_nonmiss_pivot[var_subset_2,]
var_subset_3 <- grepl("dumbbell", training_nonmiss_pivot$variable)
training_dumbbell <- training_nonmiss_pivot[var_subset_3,]
var_subset_4 <- grepl("_arm", training_nonmiss_pivot$variable)
training_arm <- training_nonmiss_pivot[var_subset_4,]

training_freq <- training_nonmissing %>% count(user_name, classe)
png(filename = "class_per_user.png")
ggplot(data = training_nonmissing, mapping = aes(user_name, fill = classe)) + 
    geom_bar(position = "dodge") +
    labs(x = "User Name", y = "Number of Observations")
dev.off()

png(filename = "roll_pitch_belt.png")
ggplot(data = training_nonmissing, mapping = aes(pitch_belt, roll_belt)) +
    geom_point(alpha = 0.2, aes(color = classe))
dev.off()
png(filename = "roll_belt_time.png")
ggplot(data = training_nonmissing, mapping = aes(raw_timestamp_part_2, roll_belt, group = num_window, color = classe)) +
    geom_point(alpha = 0.1) +
    geom_line()
dev.off()
png(filename = "roll_belt_user.png")
ggplot(data = training_nonmissing, mapping = aes(raw_timestamp_part_2, roll_belt, group = num_window, color = user_name)) +
    geom_point(alpha = 0.1) +
    geom_line()
dev.off()

png(filename = "all_var_time_class.png")
ggplot(data = training_nonmiss_pivot, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = classe)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable)
dev.off()

png(filename = "belt_time_class.png")
ggplot(data = training_belt, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = classe)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()
png(filename = "belt_time_user.png")
ggplot(data = training_belt, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = user_name)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()
png(filename = "belt_user_class.png")
ggplot(data = training_belt, mapping = aes(user_name, value, color = classe)) +
    geom_point(alpha = 0.1) +
    facet_wrap(~variable, scales = "free")
dev.off()


png(filename = "arm_time_class.png")
ggplot(data = training_arm, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = classe)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()
png(filename = "arm_time_user.png")
ggplot(data = training_arm, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = user_name)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()

png(filename = "forearm_time_class.png")
ggplot(data = training_forearm, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = classe)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()
png(filename = "forearm_time_user.png")
ggplot(data = training_forearm, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = user_name)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()

png(filename = "dumbbell_time_class.png")
ggplot(data = training_dumbbell, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = classe)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()
png(filename = "dumbbell_time_user.png")
ggplot(data = training_dumbbell, mapping = aes(raw_timestamp_part_2, value, group = num_window, color = user_name)) +
    geom_point(alpha = 0.1) +
    geom_line() +
    facet_wrap(~variable, scales = "free")
dev.off()

prop.table(table(training_nonmissing$classe,training_nonmissing$user_name))

#PCA
install.packages("ggbiplot")
library(ggbiplot)
nsv<- nearZeroVar(training_nonmissing, saveMetrics = T)
training_pca <- prcomp(training_nonmissing[,c(8:59)], center = T, scale = T)
summary(training_pca)

#Cross-validation partition
#Check for correlation
M <- abs(cor(training_nonmissing[,-c(1:7,60)]))
diag(M) <- 0
which(M>0.8, arr.ind=T)

#set.seed(1156)
#folds <- createFolds(y = training_nonmissing$classe, k=10, list = T, returnTrain = T)
#sapply(folds, length)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

control_param <- trainControl(method = "cv", number = 5, allowParallel = T)
metric_param <- "Accuracy"
y <- training_model[,54]
x <- training_model[,-54]
set.seed(1156)
lda_model <- train(classe~., data=training_model, method = 'lda', trControl = control_param, metric = metric_param)

set.seed(1156)
tree_model <- train(classe~., data=training_model, method = 'rpart', trControl = control_param, metric = metric_param)

set.seed(1156)
rf_model <- train(classe~., data=training_model, method = 'rf', trControl = control_param, metric = metric_param)
stopCluster(cluster)
registerDoSEQ()

lda_model$resample
print(lda_model)
all_results <- resamples(list(lda = lda_model, tree = tree_model, rf = rf_model))
summary(all_results)

predict(rf_model, testing)

#time_fold <- createTimeSlices(y = training_nonmissing, initialWindow = 20, horizon = 10)
#time_fold[2]

#Transpose data so have value and variable columns... facet grid plot all variables vs timestamp
#Identify features - with PCA? Need to include user, timestamp 2, and features from belt, arm, forearm, and dumbbell
#Set-up cross validation - kfold with bagging? Make kfolds based on num_window? Issue in that windows are of diff lengths?
#Run model - random forest? Parallel runs?
#Use combo of num_window and second raw timestamp to define activity?


#testing
data(iris)
# define training control
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Species~., data=iris, trControl=train_control, method="nb")
# summarize results
print(model)
