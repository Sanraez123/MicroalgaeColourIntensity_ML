## Colour_Intensity_with_ML
A documentation of my year 4 final MEng project.

### ***Project title***
Exploration of Machine Learning Models on Predicting Colour Intensity of Haematococcus Pluvialis.

#### ***Descriptions***
- Haematococcus Pluvialis, a type of microalgae, are currently highly focused upon due to it being a high value product and low environmental impact.
- However, much more research was required to deploy its production on larger scale due to various limitations (economics, water usage, etc.)
- Colour intensity is a method to monitor microalgae's growth, but requires specific lab machines, causing higher time and cost requirement.
- summary of all ML are documented in MLmodels_combined.pdf
                
#### ***Objectives***
1. To develop and improve machine learning model that can predict colour intensity based on an image taken with a smartphone.
2. To be able to both predict colour intensity and classify stress condition (red or green) of H.pluvialis.

#### ***Milestones***
- Obtain dataset by cultivating and monitoring H.pluvialis (both green and red phase) growth rate.
- 2 dataset will be obtained, 1 without light distraction, and another with light distraction, both should be labeled with the same colour intensity.
- Load the images without light distraction and link it with the analysed values recorded in a spreadsheet.
- Feature extraction of the images, in this case, Average RGB and HSV value was extracted.
- Feature scaling on the colour intensity data.
- Build a dummy model to determine the benchmark for MAE, MSE, and R^squared.
- Deploy different models to compare, in this case, LR, SVR, GB, RF, DNN, and CNN will be deployed.
- Optimise the models by hyperparameter tuning using grid search.
- Use K-fold cross validation to further optimize all models.
- Classification models will also be compared and deployed to classify the red and green colour of H.pluvialis.
- The classification model will also be optimized depending if the tradeoff between time and accuracy is worthy or not.
- Include the dataset with light distraction to look at how light distraction affect the accuracy of the model.
- Attempt to improve the accuracy of model with light distraction images.
