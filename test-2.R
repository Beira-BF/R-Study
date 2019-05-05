library(h2o) 
localH2O = h2o.init()
# train_h2o <- h2o.importFile(localH2O, path ="E:/rcode/worktrain.csv")
# test_h2o <- h2o.importFile(localH2O, path ="E:/rcode/worktest.csv")

pathToFolder = "E:/rcode/training-antagonist.csv"
train_h2o.hex = h2o.importFile(path = pathToFolder, destination_frame = "train_h2o.hex") 


# y_train <- as.factor(as.matrix(train_h2o[, 1]))
# y_test <- as.factor(as.matrix(test_h2o[, 1]))
# summary(train_h2o)  
# summary(test_h2o)  
# y <- "C1"  
# x <- setdiff(names(train_h2o), y)

y_train <- as.data.frame(train_h2o.hex[,1])

model <- h2o.deeplearning(x = 2:244, y = 1,train_h2o.hex, activation = "Rectifier", distribution = "AUTO", hidden = c(100, 200, 100), epochs = 1000) 

yhat_train <- h2o.predict(model, train_h2o.hex)$predict
yhat_train <- as.data.frame(yhat_train)


pathToFolder2 = "E:/rcode/test-antagonist.csv"
test_h2o.hex = h2o.importFile(path = pathToFolder2, destination_frame = "test_h2o.hex") 


y_test <- as.data.frame(test_h2o.hex[,1])

x <- setdiff(names(test_h2o.hex), y_test)
x <- as.data.frame(x)
yhat_test <- h2o.predict(model, x)$predict

yhat_test <- as.data.frame(yhat_test)

#write.csv(yhat_test,file="E:/rcode/output_test-antagonist-5.csv",row.names=F,quote=F)


