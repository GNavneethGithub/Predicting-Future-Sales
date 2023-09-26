This document provides an overview of my methodology and technical details regarding Kaggle's Predict Future Sales competition.

# Objective 
The objective of this project is to forecast future sales #DataScience, tackling the challenging time-series dataset comprised of daily sales data.

# Motivation
In this contest, I engaged with a demanding time-series dataset encompassing daily sales data, generously furnished by one of Russia's largest software enterprises - 1C Company. The aim was to forecast the total sales for each product and store in the forthcoming month. Through participating in this competition, I had the opportunity to apply and refine my data science skills.

# Data Overview
You have been supplied with daily historical sales data. The objective is to predict the total quantity of products sold in each shop for the test set. It's noteworthy that the roster of shops and products undergoes minor alterations monthly. Developing a resilient model capable of managing such scenarios is a facet of the challenge

## File descriptions

- sales_train.csv - the training set. Contains daily historical data spanning from January 2013 to October 2015.
- test.csv - the test set. The task is to project the sales for these shops and products for November 2015.
- sample_submission.csv - a sample submission file adhering to the correct format.
- items.csv - additional details regarding the items/products.
- item_categories.csv - additional details regarding the item categories.
- shops.csv - additional details regarding the shops.

## Data fields

- ID - an identifier representing a (Shop, Item) pair within the test set
- shop_id - distinct identifier for a shop
- item_id - distinct identifier for a product
- item_category_id - distinct identifier for an item category
- item_cnt_day - quantity of products sold. The goal is to predict a monthly total of this metric
- item_price - prevailing price of an item
- date - date formatted as dd/mm/yyyy
- date_block_num - a sequential month number, utilized for ease. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
- item_name - name of the item
- shop_name - name of the shop
- item_category_name - name of the item category

# Summary 

- Methods attempted but led to inferior RMSE: XGBoost, Stacking (both simple averaging and meta models like Linear Regression and shallow random forest)
- The pivotal features are lag features from preceding months, particularly the 'item_cnt_day' lag features. Some of these, accessible in my lag dataset, include:
-- target_lag_1, target_lag_2: item_cnt_day for each shop-item duo from the prior month and the month before that
-- item_block_target_mean_lag_1, item_block_target_sum_lag_1: sum and average of item_cnt_day per item from the last month. Significant features are derived from the LightGBM model.
- Tools utilized in this competition encompass: numpy, pandas, sklearn, XGBoost GPU, LightGBM (operating Pytorch)
- All models were fine-tuned on a Linux server equipped with an Intel i5 processor, 16GB RAM, NVIDIA 1080 GPU. Model tuning spanned roughly 8 to 10 hours, while training on the entire dataset was completed in less than or equal to 5 minutes.

 # Exploratory Data Analysis
Further details can be explored in the EDA notebook.

A basic analysis of the data has been conducted, which includes plotting the sum and average of item_cnt_day for each month to discern patterns, investigating missing values, examining the test set, and more.

Here are some intriguing insights garnered from the EDA:

The quantity of sold items diminishes as the year progresses.
There are noticeable spikes in November, along with a zig-zag pattern in item count observed during the months of June, July, and August. This prompted me to research Russian national holidays and formulate a Boolean holiday feature. More details on this can be found in the 'Feature Engineering' section.
The data is devoid of missing values.
Some notable findings from the analysis of the test set include:
Not all shop_id values present in the training set are utilized in the test set. The test set omits the following shops (but not the other way around): [0, 1, 8, 9, 11, 13, 17, 20, 23, 27, 29, 30, 32, 33, 40, 43, 51, 54]
Not every item in the training set is found in the test set and vice versa.
Within the test set, a fixed assortment of items (5100) is designated for each shop_id, with each item appearing only once per shop. This likely indicates that items are selected via a generator, which will lead to a multitude of zeros for item count. Hence, generating all possible shop-item pairs for each month in the training set and filling missing item counts with zero is a logical step.

# Feature Engineering
## Generation of All Shop-Item Pairs and Mean Encoding
Given that the objective of the competition is to forecast on a monthly basis, it's imperative to aggregate the data to a monthly level prior to executing any encodings.

The item counts for each shop-item pair per month (termed as 'target') were computed. Additionally, the sum and average of item counts for each shop per month ('shop_block_target_sum', 'shop_block_target_mean'), each item per month ('item_block_target_sum', 'item_block_target_mean'), and each item category per month ('item_cat_block_target_sum', 'item_cat_block_target_mean') were also calculated.

The procedure can be reviewed in this notebook, under the section titled 'Generating new_sales.csv'. The datasets produced from these steps will be stored with the name 'new_sales.csv'.

## Generation of Lag Features
Lag features represent values from previous time steps. I am creating lag features based on 'item_cnt', grouped by 'shop_id' and 'item_id'. The time steps considered are: 1, 2, 3, 5, and 12 months.

All sales records prior to 2014 are discarded, as there wouldn't be any lag features before 2014 given the 12-month lag.

These lag features emerged as the most crucial features in my dataset, as per the importance features identified by gradient boosting.

Additional details can be explored in this notebook, under the section 'Generate lag feature new_sales_lag_after12.pickle'.

# Cross-Validation
Given the time-series nature of this data, it's necessary to pre-specify the data segments for training and testing. I have a function named get_cv_idxs in utils.py that returns a list of tuples for cross-validation purposes. I opted for a 6-fold cross-validation, spanning from date_block_num 28 to 33, and fortuitously, the CV score aligns well with the leaderboard score.

The output from this function can be fed into sklearn's GridSearchCV for further analysis.

# Ensemble Modeling
Utilizing LightGBM, XGB model-1, and XGB model-2 out-of-fold features from preceding methods, I computed the pairwise differences among them, obtained the mean of all three LGB, XGB1, and XGB2 out-of-fold features, and incorporated the most significant features from feature importance: 'target_lag_1'.

Subsequently, I experimented with several ensemble methods:

1. Simple Average and Weighted Average
2. SKlearn Linear Regression and ElasticNet
3. Shallow Random Forest, fine-tuned with 5 folds (from 29 to 33)

All these methods yielded an RMSE score that was marginally higher than the best model from LightGBM, hence LightGBM continues to surpass them in performance.




 
