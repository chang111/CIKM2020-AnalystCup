# CIKM 2020 Analyticup COVID-19 Retweet Prediction Challenge

## Dependencies

* python 3.6
* Lightgbm

## The Framework structure
```directory
CIKM2020/
      EDA.ipynb ####can give you some insight about the data
      construct_feature.ipynb ####is all the features that we have tried
      logger.py ####a function we need in the model
      utils.py ####a function we need in the model
      config.py ####a function we need in the model
      select_feature.py ####the method that help us choose the best feature
      model_average_stack.py ####the best model
      prepocess.py ####you can get the used feature through this file
      README.md
```
## Code

### Demo

In this fold, we provide our code in the challenge:
  * At first, you strongly recommend you read the EDA.ipynb, you can get some insight of this challenge:
  * Then you can run the preprocess.py to get the used feature.
  * Finally you can the prediction by running:
  ```bash
  >>python model_average_stack.py
  ``` 
  * After the test finishes, we can get the prediction result under the fold.
