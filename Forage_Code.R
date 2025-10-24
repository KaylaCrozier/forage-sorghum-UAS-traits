# Packages
library(dplyr)      
library(readxl)  
library(sp)
library(raster)
library(rgdal)        
library(FIELDimageR)  
library(caret)        
library(randomForest) 
library(gbm)          
library(rpart)      
library(brnn)         
library(e1071)        
library(xgboost)      
library(glmnet)     

# Load all required packages
library(readxl)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(elasticnet)
library(glmnet)
library(ggplot2)
library(lattice)
library(foreach)
library(doParallel)
library(tibble)

#################################################################################################### 
######################################### Data Extraction ########################################## 
####################################################################################################

################## Vegetative Indices ###################
setwd("C:/Desktop")
EX1 <- stack("Orthomosaic.tif")
plotRGB(EX1, r=1, g=2, b=3)
EX1.Shape<-readOGR("Shapefile.shp")
plotRGB(EX1)
plot(EX1.Shape, add=T)
EX1.RemSoil<-fieldMask(mosaic=EX1,
                       Red=1,
                       Green=2,
                       Blue=3,
                       index="HUE",
                       cropValue=0.7,
                       cropAbove=T)
EX1.Indices<-fieldIndex(mosaic=EX1.RemSoil$newMosaic,Red=1,Green=2,Blue=3,
                        index=c("BI","SCI", "GLI", "HI", "NGRDI", "SI", "VARI"),
                        myIndex=c("((2*Green)-Red-Blue)",#ExG
                                  "((3*Green)-(2.4*Red)-Blue)",#ExGR
                                  "((1.4*Red)-Green)",#EXR
                                  "((1.4*Blue-Green)/(Green+Blue+Red))",#ExB
                                  "(Green/(Red+Green+Blue))",#NG
                                  "((1.262*Green)-(0.884*Red)-(0.311*Blue))",#MExG
                                  "((Green^2)-(Red^2))/((Green^2)+(Red^2))",#MGVRI
                                  "(Red-Green)",#RGD
                                  "(((Green^2)-(Red*Blue))/((Green^2)+(Red*Blue)))",#RGBVI
                                  "Green/(Red^0.667*Blue^0.334)"#VEG
                        ))
EX1.Info<-fieldInfo(mosaic=EX1.Indices,fieldShape=EX1.Shape, n.core=11)
EX1.Info$fieldShape@data 
data1<- EX1.Info[["fieldShape"]]@data
write.csv(data1,"Indices.csv")

#################################################################################################### 
########################################### Plant Height ########################################### 
####################################################################################################

# extracted via CloudCompare and ArcGIS Pro 

#################################################################################################### 
####################################### Days to Mid-Anthesis ####################################### 
####################################################################################################
################## Create Data Files ###################
setwd("C:/Desktop")
data1 <- read_excel("/ForageData.xlsx", sheet= "DTA")
data1$Actual_DTA <- as.numeric(data1$Actual_DTA)
data1$TestName1 <- as.character(data1$TestName1)
train <- filter(data1, TestName1 %in% c("D", "F", "G", "I")) # data used to train and test the models
validate <- filter(data1, TestName1 %in% c("E", "H")) # data used to validate the models

################## Build Model ###################
predictor_columns <- colnames(train)[c(12:74)] # vegetative indices and percent pixels 
response_variables <- c( "Actual_DTA")
trControl <- trainControl(method = 'repeatedcv', 
                          number = 5, repeats = 20, 
                          returnData = TRUE, 
                          savePredictions = "final",
                          classProbs = TRUE, 
                          seeds = NULL)
best_models <- list()
results_df <- data.frame(Response_Variable = character(),
                         Method = character(),
                         RMSE = numeric(),
                         stringsAsFactors = FALSE)
for (response_var in response_variables) {
  best_models[[response_var]] <- list()
  for (method in c("rf", "svmRadial", "xgbTree", "enet")) {
    formula <- as.formula(paste(response_var, "~", paste(predictor_columns, collapse = " + ")))
    model <- train(formula, 
                   data = train, 
                   method = method,
                   trControl = trControl,
                   metric = "RMSE")
    resamples <- model$resample
    resamples <- resamples %>%
      mutate(Response_Variable = response_var,
             Method = method)
    results_df <- bind_rows(results_df, resamples)
    best_models[[response_var]][[method]] <- model
    saveRDS(model, paste("best_", method, "_model_", response_var, ".rds", sep = ""))
  }
}
write.csv(results_df, "Train22_model_resamples.csv", row.names = FALSE) # Save all the resample results

################## Apply Model ###################
validate1 <- validate # data frame that includes validation data 
unique_loc2 <- unique(validate1$TestName1)
response_variables <- c("Actual_DTA")
overall_test_results_df <- data.frame(TestName1 = character(),
                                      Response_Variable = character(),
                                      Method = character(),
                                      RMSE = numeric(),
                                      MAE = numeric(),
                                      stringsAsFactors = FALSE)
model_directory <- getwd()
# Iterate over each unique TestName1 to process models
for (current_test in unique(validate1$TestName1)) {
  message("Processing TestName1: ", current_test)
  test_subset <- filter(validate1, TestName1 == current_test)
  loaded_models <- list()
  for (response_var in response_variables) {
    for (method in c("rf", "svmRadial", "xgbTree", "enet")) {
      model_path <- file.path(model_directory, paste("best_", method, "_model_", response_var, ".rds", sep = ""))
      if (file.exists(model_path)) {
        loaded_models[[paste(method, response_var, sep = "_")]] <- readRDS(model_path)
        message("Model loaded: ", model_path)
      } else {
        message("Model not loaded: ", model_path)
      }
    }
  }
  # Data frame to store results for the current TestName
  test_results_df <- data.frame(Response_Variable = character(),
                                Method = character(),
                                RMSE = numeric(),
                                MAE = numeric(),
                                stringsAsFactors = FALSE)
  for (response_var in response_variables) {
    for (method in c("rf", "svmRadial", "xgbTree", "enet")) {
      model_key <- paste(method, response_var, sep = "_")
      if (model_key %in% names(loaded_models)) {
        model <- loaded_models[[model_key]]
        test_predictions <- predict(model, newdata = test_subset) # Make predictions
        rmse <- sqrt(mean((test_predictions - test_subset[[response_var]])^2))
        mae <- mean(abs(test_predictions - test_subset[[response_var]]))
        predictions_df <- data.frame(Predicted = test_predictions,
                                     Actual = test_subset[[response_var]],
                                     SN = test_subset$SN) # Save predictions in a CSV file
        write.csv(predictions_df, paste("test_predictions_", current_test, "_", method, "_", response_var, ".csv", sep = ""), row.names = FALSE)
        test_results_df <- rbind(test_results_df, data.frame(Response_Variable = response_var,
                                                             Method = method,
                                                             RMSE = rmse,
                                                             MAE = mae))
      } else {
        message("Model not loaded: ", model_key)
      }
    }
  }
  test_results_df$TestName <- current_test
  overall_test_results_df <- rbind(overall_test_results_df, test_results_df)
}
write.csv(overall_test_results_df, "overall_test_results.csv", row.names = FALSE)

#################################################################################################### 
########################################## Biomass Yield ########################################### 
####################################################################################################
################## Create Data Files ################## 
setwd("C:/Desktop")
data1 <- read_excel("/Users/kayla/Desktop/ForageData.xlsx", sheet= "Biomass")
data1$Actual_Biomass <- as.numeric(data1$Actual_Biomass)
data1$TestName1 <- as.character(data1$TestName1)
train <- filter(data1, TestName1 %in% c("D", "F", "J","C", "L")) # data used to train and test the models
validate <- filter(data1, TestName1 %in% c("A","E", "B", "K")) # data used to validate the models

################## Build Model ###################
response_variables <- c("Actual_Biomass")
predictor_columns <- names(train)[16:37] 
trControl <- trainControl(method = 'repeatedcv', 
                          number = 5, repeats = 20, 
                          returnResamp = "final",  
                          savePredictions = "final")
best_models <- list()
results_df <- data.frame(Response_Variable = character(),
                         Method = character(),
                         Resample = character(),
                         RMSE = numeric(),  
                         stringsAsFactors = FALSE)
for (response_var in response_variables) {
  best_models[[response_var]] <- list()
  for (method in c("rpart", "gbm", "rf", "brnn")) {
    formula <- as.formula(paste(response_var, "~", paste(predictor_columns, collapse = " + ")))
    cat("Training model for:", response_var, "using method:", method, "\n")
    cat("Formula:", deparse(formula), "\n")
    tryCatch({
      model <- train(formula, 
                     data = train, 
                     method = method,
                     trControl = trControl,
                     metric = "RMSE")
      resamples <- model$resample
      resamples <- resamples %>%
        mutate(Response_Variable = response_var, 
               Method = method)  
      results_df <- bind_rows(results_df, resamples)
      saveRDS(model, paste("best_", method, "_model_", response_var, ".rds", sep = "")) # Save the trained model
    }, error = function(e) {
      message("Error training model: ", e$message)
    })
  }
}
write.csv(results_df, "Train_Results.csv", row.names = FALSE)


################################# Validate Models ################################# 

unique_trial_names <- unique(validate$TestName1)
overall_test_results_df <- data.frame(TestName1 = character(),
                                      Response_Variable = character(),
                                      Method = character(),
                                      RMSE = numeric(),  
                                      stringsAsFactors = FALSE)
for (trial_name in unique_trial_names) {
  test <- filter(validate, TestName1 == trial_name)  
  loaded_models <- list()
  for (response_var in response_variables) {
    for (method in c("rpart", "gbm", "rf", "brnn")) {
      model_path <- paste("best_", method, "_model_", response_var, ".rds", sep = "")
      if (file.exists(model_path)) {
        loaded_models[[paste(method, response_var, sep = "_")]] <- readRDS(model_path)
      } else {
        message("Model not loaded: ", model_path)
      }
    }
  }
  test_results_df <- data.frame(Response_Variable = character(),
                                Method = character(),
                                RMSE = numeric(),  
                                stringsAsFactors = FALSE)
  for (response_var in response_variables) {
    for (method in c("rpart", "gbm", "rf", "brnn")) {
      model_key <- paste(method, response_var, sep = "_")
      if (model_key %in% names(loaded_models)) {
        model <- loaded_models[[model_key]]
        test_predictions <- predict(model, newdata = test)
        if (!is.numeric(test_predictions)) {
          message("Predictions are not numeric for model: ", model_key)}
        rmse <- sqrt(mean((test_predictions - test[[response_var]])^2))
        cat("Trial Name:", trial_name, "RMSE:", rmse, "\n")
        predictions_df <- data.frame(Predicted = test_predictions,
                                     Actual = test[[response_var]],
                                     SN = test$SN)
        write.csv(predictions_df, paste("test_predictions_", trial_name, "_", method, "_", response_var, ".csv", sep = ""), row.names = FALSE)
        test_results_df <- rbind(test_results_df, data.frame(Response_Variable = response_var,
                                                             Method = method,
                                                             RMSE = rmse))
      } else {
        message("Model not loaded: ", model_key)
      }
    }
  }
  test_results_df$TrialName1 <- trial_name  
  overall_test_results_df <- rbind(overall_test_results_df, test_results_df)
}
write.csv(overall_test_results_df, "Validation_results.csv", row.names = FALSE) #save final predictions

