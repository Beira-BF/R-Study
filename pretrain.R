#! /path/to/Rscript
library(h2o) 
localH2O = h2o.init(nthreads = -1)
# train_h2o <- h2o.importFile(localH2O, path ="E:/rcode/worktrain.csv")
# test_h2o <- h2o.importFile(localH2O, path ="E:/rcode/worktest.csv")

#pathToFolder = "/home/QSAR/Rcode/test/tr.csv"
pathToFolder = "E:/rcode/tr.csv"
train_h2o.hex = h2o.importFile(path = pathToFolder, destination_frame = "train_h2o.hex") 

resp =1

train_h20.hex[,resp] <- as.factor(train_h2o.hex[,resp])
test_h2o.hex[,resp] <- as.factor(test_h2o.hex[,resp])

# split data into two parts (first part for unsupervised training, second part for supervised training)
sid <- h2o.runif(train_h2o.hex, seed=0)
# split <- h2o.splitFrame(train_hex, 0.5)

# first part of the data, without labels for unsupervised learning (DL auto-encoder)
train_unsupervised <- train_h2o.hex[sid>=0.5,]
#summary(train_unsupervised)

# second part of the data, with labels for supervised learning (drf)
train_supervised <- train_h2o.hex[sid<0.5,]


# y_train <- as.factor(as.matrix(train_h2o[, 1]))
# y_test <- as.factor(as.matrix(test_h2o[, 1]))
# summary(train_h2o)  
# summary(test_h2o) 1111  
# y <- "C1"  
# x <- setdiff(names(train_h2o), y)

y_train <- as.data.frame(train_h2o.hex[,1])

ae_model <- h2o.deeplearning(x = 2:5235, y = 1,train_unsupervised, activation = "RectifierWithDropout", distribution = "AUTO", hidden = c(20), input_dropout_ratio = 0.2, epochs = 1) 

pretrained_model <- h2o.deeplearning(x = 2:5235,          
                                     y = 1, 
                                     train_supervised,
                                     hidden=c(20), 
                                     epochs=1, 
                                     reproducible=T,
                                     seed=1234,
                                     pretrained_autoencoder="ae_model")
#yhat_train <- h2o.predict(pretrained_model, train_h2o.hex)$predict
#yhat_train <- as.data.frame(yhat_train)


#pathToFolder2 = "/home/QSAR/Rcode/test/tst.csv"
pathToFolder2 = "E:/rcode/tst.csv"
test_h2o.hex = h2o.importFile(path = pathToFolder2, destination_frame = "test_h2o.hex") 
y_test <- as.data.frame(test_h2o.hex[,1])
yhat_test <- h2o.predict(pretrained_model, test_h2o.hex)$predict
yhat_test <- as.data.frame(yhat_test)

#write.csv(yhat_test,file="/home/QSAR/Rcode/test/output_test-CPU-2.csv",row.names=F,quote=F)

write.csv(yhat_test,file="E:/rcode/test/output_test-CPU-7.csv",row.names=F,quote=F)