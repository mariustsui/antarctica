
# Installing package
#install.packages("randomForest") 
#install.packages('kernelshap')
#install.packages('shapviz')
#install.packages("gridExtra")


# Loading package 
library(randomForest) 
library(raster)
library(ggplot2)
library(shapviz)
library(treeshap)
library(gridExtra)

remove(list = ls())
gc()

# Set the source data
source_data = read.csv("H:/My Drive/Colab Notebooks/PROJECT_ANTARCTICA/NZTAB_SURVEYPOINTS.csv")

# Select feature variable to be used in the training model
used_feature <- c("Relative.relief.1000M",
                  "Relative.relief.100M",
                  "Aspect",
                  "Wetness.Index",
                  "Elevation",
                  "Slope",
                  "Aspect.to.North",
                  "Distance.from.Ice",
                  "Distance.from.Coast")

feature_data <- source_data[names(source_data) %in% used_feature]

# Load feature raster for prediction
rr1000 <- raster("antarctica/SOURCE/rr1000.tif")
rr100 <- raster("antarctica/SOURCE/rr100.tif")
aspect <- raster("antarctica/SOURCE/aspect.tif")
cti <- raster("antarctica/SOURCE/cti.tif")
dem <- raster("antarctica/SOURCE/dem.tif")
slope <- raster("antarctica/SOURCE/slope.tif")
rel_aspect <- raster("antarctica/SOURCE/rel_aspect.tif")
edgedist <- raster("antarctica/SOURCE/edgedist.tif")
coastdist <- raster("antarctica/SOURCE/coastdist.tif")

# Stack and rename the raster as per used_feature
raster_stack <- stack(rr1000, rr100, aspect, cti, dem, slope, rel_aspect, edgedist, coastdist)
names(raster_stack) <- used_feature

# Features to be predicted
bio_attr <- c("CYANO", "MOSS", "LICHEN", "SPRINGTAILS", "MITES")

# Main loop for feature prediction
for (b in bio_attr){
  
  ####################################### REGRESSION ###############################################
  # Set the feature variable name
  bio = paste0('COV_',b)
  
  # Prepare a training data set including all features variable and the feature to be predicted
  training_data <- NULL
  training_data <- cbind(feature_data, bio = source_data[, bio])
  training_data <- droplevels(training_data)
  
  # Sample the data into training and validation, 75% for training and 25% for validation
  ind <- sample(2, nrow(training_data), replace = TRUE, prob = c(0.75, 0.25))
  train <- training_data[ind==1,]
  test <- training_data[ind==2,]
  
  # Set a random seed
  set.seed(99)
  
  # Configure the Random Forest model for regression
  rf_regression <-randomForest(bio~.,
                               data=train,
                               ntree=1000, 
                               type="regression", 
                               mtry=3) 
  
  # Prepare SHAP values and graph
  unified <- randomForest.unify(rf_regression, train)
  shaps <- treeshap(unified, train)
  sv <- shapviz(shaps)
  bee <- sv_importance(sv, kind = "bee")
  shap_g <- bee + ggtitle(paste0("SHAP value for ",b," regression")) + theme(plot.title = element_text(size=20, face="bold"))
  
  # Prepare the model performance as a ggplot object
  exp<-capture.output(rf_regression)
  text_box <- ggplot() + annotate("text", x=1, y=c(length(exp):1), label=exp, hjust=0, size=3) + theme_void() + xlim(0,10) + ylim(0, length(exp)*2)
  
  # Arrange two ggplot object into one plot
  g <- arrangeGrob(shap_g, text_box, nrow=1, widths=c(3,1)) #generates g
  
  # Save the result as an image file
  ggsave(paste0("antarctica/R_OUTPUT/",b,"_COV_SHAP_1000.png"), g)
  
  # Prediction on the Raster layer and save it to disk
  bio_cov <- raster::predict(model=rf_regression, object=raster_stack, filename=paste0("antarctica/R_OUTPUT/",bio,"_ByR.tif"), progress='text', overwrite=TRUE)
  
  # Remove the predicted raster layer to free memory
  remove(bio_cov)
  gc()
  
  
  
  #################################  CLASSIFICATION ###############################
  
  # Set the feature variable name
  bio = paste0('CAT_',b)
  
  # Prepare a training data set including all features variable and the feature to be predicted
  training_data <- NULL
  training_data <- cbind(feature_data, bio = source_data[, bio])
  training_data$bio = as.factor(training_data$bio)
  training_data <- droplevels(training_data)
  
  # Sample the data into training and validation, 75% for training and 25% for validation
  ind <- sample(2, nrow(training_data), replace = TRUE, prob = c(0.75, 0.25))
  train <- training_data[ind==1,]
  test <- training_data[ind==2,]

  # Set a random seed
  set.seed(129)
  
  # Configure the Random Forest model for classifcation
  rf_classification <-randomForest(bio~.,
                                   data=train, 
                                   ntree=1000, 
                                   type="classification", 
                                   mtry=3) 

  # Predict the training data with the model for performance
  c1 <- predict(rf_classification, train)
  c1_exp <- capture.output(confusionMatrix(c1, train$bio))
  
  # Predict the validation data with the model for performance
  c2 <- predict(rf_classification, test)
  c2_exp <- capture.output(confusionMatrix(c2, test$bio))
  
  # Save the performance to a text file on disk
  capture.output(confusionMatrix(c1, train$bio), file = paste0("antarctica/R_OUTPUT/",b,"_CAT_PERFORMANCE_1000.txt"), append = FALSE)
  capture.output(confusionMatrix(c2, test$bio), file = paste0("antarctica/R_OUTPUT/",b,"_CAT_PERFORMANCE_1000.txt"), append = TRUE)
  
  # Prepare SHAP values and graph
  unified <- randomForest.unify(rf_classification, train)
  shaps <- treeshap(unified, train)
  sv <- shapviz(shaps)
  bee <- sv_importance(sv, kind = "bee")
  shap_g <- bee + ggtitle(paste0("SHAP value for ",b," classification")) + theme(plot.title = element_text(size=20, face="bold"))
  
  exp <- capture.output(rf_classification)
  
  # Prepare the performance as a ggplot object
  c1_p <- ggplot() + annotate("text", x=0, y=c(length(c1_exp):1), label=c1_exp, hjust=0, size=2.5) + theme_void() + xlim(0,10) + ylim(0, length(c1_exp))
  c2_p <- ggplot() + annotate("text", x=0, y=c(length(c2_exp):1), label=c2_exp, hjust=0, size=2.5) + theme_void() + xlim(0,10) + ylim(0, length(c2_exp))
  rf_p <- ggplot() + annotate("text", x=0, y=c(length(exp):1), label=exp, hjust=0, size=2.5) + theme_void() + xlim(0,10) + ylim(0, length(exp))
  
  text_box <- arrangeGrob(rf_p, c1_p, c2_p, nrow=3, ncol=1)
  
  # Arrange the SHAP graph and performance to be plotted in one single plot
  g <- arrangeGrob(shap_g, text_box, nrow=1, widths=c(3,1)) 
  
  # Save the output as an image file to disk
  ggsave(paste0("antarctica/R_OUTPUT/",b,"_CAT_SHAP_1000.png"), g)
  
  # Prediction on the Raster layer and save it to disk
  bio_cat <- raster::predict(model=rf_classification, object=raster_stack, filename=paste0("antarctica/R_OUTPUT/",bio,"_ByR.tif"), progress='text', overwrite=TRUE)
  
  # Remove the predicted raster layer to free memory
  remove(bio_cat)
  gc()
}


