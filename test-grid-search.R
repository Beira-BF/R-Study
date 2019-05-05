library(h2o) 
localH2O = h2o.init()
# train_h2o <- h2o.importFile(localH2O, path ="E:/rcode/worktrain.csv")
# test_h2o <- h2o.importFile(localH2O, path ="E:/rcode/worktest.csv")

pathToFolder = "E:/rcode/training-antagonist.csv"
train_h2o.hex = h2o.importFile(path = pathToFolder, destination_frame = "train_h2o.hex") 
pathToFolder2 = "E:/rcode/test-antagonist.csv"
test_h2o.hex = h2o.importFile(path = pathToFolder2, destination_frame = "test_h2o.hex") 

# y_train <- as.factor(as.matrix(train_h2o[, 1]))
# y_test <- as.factor(as.matrix(test_h2o[, 1]))
# summary(train_h2o)  
# summary(test_h2o)  
# y <- "C1"  
# x <- setdiff(names(train_h2o), y)

y_train <- as.data.frame(train_h2o.hex[,1])
hidden_opt <- list(c(32,32), c(32,16,8), c(100))
l1_opt <- c(1e-4,1e-3)
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)
model_grid <- h2o.grid("deeplearning",
                        grid_id = "mygrid",
                        hyper_params = hyper_params,
                       x = 2:244, y = 1,train_h2o.hex, activation = "Maxout", distribution = "bernoulli", 
                       hidden = c(100, 200, 100), loss="CrossEntropy",
                        training_frame = train_h2o.hex,
                        validation_frame = test_h2o.hex,
                        score_interval = 2,
                        epochs = 100,
                        stopping_rounds = 3,
                        stopping_tolerance = 0.05,
                        stopping_metric = "misclassification")
yhat_train <- h2o.predict(model, train_h2o.hex)$predict
yhat_train <- as.data.frame(yhat_train)

y_test <- as.data.frame(test_h2o.hex[,1])
yhat_test <- h2o.predict(model, test_h2o.hex)$predict
yhat_test <- as.data.frame(yhat_test)

write.csv(yhat_test,file="E:/rcode/output_test-antagonist-6.csv",row.names=F,quote=F)


