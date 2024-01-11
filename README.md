# Heritage Housing

Heritage housing is an simple ML system that predicts prices of houses in Ames, Iowa, US, by using gradient-boosting based ML regression technique that maps relationships between housing attributes to its sale prices. 

## Table of Content
- [Business Requirements](https://github.com/fokhrun/heritage_housing#business-Requirements-)
- [Dataset](https://github.com/fokhrun/heritage_housing#dataset-)
- [Hypothesis](https://github.com/fokhrun/heritage_housing#hypothesis-)
- [Mapping Business Case To ML Solution](https://github.com/fokhrun/mapping-business-case-to-ml-solution-)
- [Planning & Execution](https://github.com/fokhrun/heritage_housing#planning-executions-)
- [Software Development]
    - [Tech Stack]
    - [Testing]
    - [Deployment]
    - [Development Environment]
    - [Credits]

## Business requirements
The client is interested in a dashboard application that allows her to maximize sale price of her inherited houses in the Ames, Iowa area, as well as any other houses in that area. The dashboard should includes 
- prediction of house sale prices from her 4 inherited houses, and any other house in Ames, Iowa. The prediction should be generated by a reliable ML system.  
- data visualizations of training data that shows correlations of the properties of houses against the sale price. It is sufficient to use conventional data analysis techniques.
- A way to explain the predictions made by the estimator. The estimator should demonstrate an R2 score of at least 0.75 on the training and the test data.
- Hypothesis used in the project and how it has been validated

## Dataset
The dataset used in this project is a curated version of publicly available Ames Housing dataset. The dataset contains 24 explanatory variables describing key attributes of residential homes and their sale prices in Ames, Iowa. It contains unique records of 1430 houses. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data).

|  Num | Variables | Type | Description | Value |
|---|---|---|---|---|
| 1 | 1stFlrSF | numerical | First Floor square feet | 334 - 4692 |
| 2 | 2ndFlrSF | numerical | Second floor square feet |	0 - 2065 |
| 3	| BedroomAbvGr | categorical | Bedrooms above grade (does NOT include baseme... | 0 - 8 |
| 4	| BsmtExposure | categorical | Refers to walkout or garden level walls | Gd: Good Exposure, Av: Average Exposure, ...|
| 5	| BsmtFinType1 | categorical | Rating of basement finished area | GLQ: Good Living Quarters, ALQ: Average Living, ...|
| 6	| BsmtFinSF1 | numerical | Type 1 finished square feet | 0 - 5644 |
| 7	| BsmtUnfSF	| numerical	| Unfinished square feet of basement area |	0 - 2336 |
| 8	| TotalBsmtSF |	numerical |	Total square feet of basement area | 0 - 6110 |
| 9 | GarageArea | numerical | Size of garage in square feet | 0 - 1418 |
| 10 | GarageFinish | categorical | Interior finish of the garage |	Fin: Finished, RFn: Rough Finished, Unf: Unfinished, ...|
| 11 | GarageYrBlt | temporal |	Year garage was built |	1900 - 2010 |
| 12 | GrLivArea | numerical | Above grade (ground) living area square feet | 334 - 5642 |
| 13 | KitchenQual | categorical | Kitchen quality | Ex: Excellent, Gd: Good, TA: Typical/Average, ... |
| 14 | LotArea | numerical | Lot size in square feet |	1300 - 215245 |
| 15 | LotFrontage | numerical | Linear feet of street connected to property | 21 - 313 |
| 16 | MasVnrArea |	numerical | Masonry veneer area in square feet | 0 - 1600 |
| 17 | EnclosedPorch | numerical |	Enclosed porch area in square feet | 0 - 286 |
| 18 | OpenPorchSF | numerical | Open porch area in square feet | 0 - 547 |
| 19 | OverallCond | categorical |	Rates the overall condition of the house |	10: Very Excellent, 9: Excellent, 8: Very Good, ...|
| 20 | OverallQual | categorical |	Rates the overall material and finish of the house | 10: Very Excellent, 9: Excellent, 8: Very Good...|
| 21 | WoodDeckSF |	numerical |	Wood deck area in square feet |	0 - 736 |
| 22 | YearBuilt |	temporal |	Original construction date | 1872 - 2010 |
| 23 | YearRemodAdd | temporal | Remodel date |	1950 - 2010 |
| 24 | SalePrice | numerical | Sale Price |	34900 - 755000 |

## Hypothesis

The target variable in this project is SalePrice, which represent the sale price of houses. The factors that effect these prices are:

- size:
    - hypothesis: larger the property, higher should be the price
    - relevant variables: 1stFlrSF, 2ndFlrSF, BsmtFinSF1, BsmtUnfSF, TotalBsmtSF, GarageArea, GrLivArea, LotArea, LotFrontage, MasVnrArea, EnclosedPorch, WoodDeckSF
- condition:
    - hypothesis: better the condition, higher should be the price
    - relevant variables: BedroomAbvGr, BsmtExposure, KitchenQual, BsmtFinType1, GarageFinish, OverallCond, OverallQual
- age: newer the house, higher should be the price
    - hypothesis: newer the house, higher the price
    - relevant variables: GarageYrBlt,YearBuilt, YearRemodAdd

### Validation Approach
These hypothesis should be validated by the following observations:
- Both the actual and predicted (on data unused in training) SalePrice should be very strongly correlated
- The predicted (on data unused in training) SalePrice should generally increase with the that of the house size, condition, and age. It should show correlation to the columns mentioned above similarly to the actual sale price.

Note that location desirability and room count also have similar effect, but the dataset did not those variables.

## Mapping Business Case To ML Solution

### How the data is collected and cleansed

Before we handle any of the business case, we need to acquire the data. Even though the data can be acquired using a plain Python script, we have implemented it as a jupyter notebook titled [Data Collection](https://github.com/fokhrun/heritage_housing/blob/documentation/jupyter_notebooks/data_collection.ipynb). The notebook performs the following:

1. Download the data from kaggle. It is a zip file that is extracted. It provides three main data: 
    1. `house-metadata.txt`: contains description of the attributes 
    2. `house_prices_records.csv`: contains data to be used for ML model training
    3. `inherited_houses.csv`: contains data to be used for ML model prediction

2. Read the `house_prices_records.csv` and `inherited_houses.csv` files as pandas dataframes, validate the shapes, and get a basic understanding of the dataset. Apart from the number of rows (1460 for the former and 4 for the latter), the difference between these two dataframes is the `SalePrice` attribute, which is only available in `house_prices_records.csv`. This is the target attribute. 

3. Analyze the missing data. `inherited_houses.csv` does not have any. However, the `house_prices_records.csv` has some. Observe the following image for more details. 

![Missing Values](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/missing_data.png)

### How the House attributes correlate with the sale price

To study the housing attributes and its relationship with the target variable `SalePrice`, we implemented a Jupyter notebook titled [Exploratory Data Analysis](https://github.com/fokhrun/heritage_housing/blob/documentation/jupyter_notebooks/exploratory_data_analysis.ipynb). The notebook performs the following steps:

1. Read `house-metadata.txt` and prepares housing attribute descriptions as a table with the following columns `featureName`, `featureType` (inferred categorisation of features as numerical, categorical, or temporal), `featureDescription`, and `featureValues`.
2. Read `house_prices_records.csv`, analyse its missing data, and remove columns that have 10% or more data missing. 
3. Analyse numerical, temporal, and categorical columns as well target variable. 
4. Analyse correlation of the housing attributes to the target variable. 
5. Identify variables with strong and moderate correlations and preserve them to be used in the next steps. 

### Analyse correlation to target variable

#### Numerical and temporal attributes

We used scatter plot between each numerical and temporal variables with target variable. To measure the correlation we used [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) as score titled `corr`. We used the score to also identify each column to have `very weak correlation` (`corr` < 0.3), `weak correlation` (0.3 =< `corr` < 0.5),  `moderate correlation` (0.5 =< `corr` < 0.7), and `strong correlation` (`corr` >= 0.7). 

The analysis is demonstrated in the following figures. 

![Correlation to numerical variables (group 1)](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/correlation_numerical_1.png
![Correlation to numerical variables (group 2)](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/correlation_numerical_2.png)

From these variables, the following have a `corr` score 0.5 or higher. 

|  Variable | Score | Correlation Category |
|---|---|---|
| GrLivArea | 0.71 | strong |
| GarageArea | 0.62 | moderate |
| TotalBsmtSF | 0.61 | moderate |
| 1stFlrSF | 0.61 | moderate |
| YearBuilt | 0.52 | moderate |
| YearRemodAdd | 0.51 | moderate |

#### Categorical attributes

For categorical attributes, I used box plots of target variable per category for each attributes. The categorical variables are have orders as they fit likert-like scales. To evaluate the correlations, we visually inspected if category order indicates reasonable sale price improvement. Please see the following image for more details. The visual inspection indicated that the following variables have a moderate to strong correlation: `KitchenQual`, `OverallCond`, `OverallQual`.

![Correlation to categorical variables](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/correlation_categorical.png)

### How the trained model generates good predictions

The model training is process is implemented using the Jupyter notebook titled [Model Training, Optimization, and Validation](https://github.com/fokhrun/heritage_housing/blob/documentation/jupyter_notebooks/model_training.ipynb)

The process consists of the following steps:
1. Data loading
2. Feature Engineering
3. Hyperparameter Tuning
4. Model Training
5. Model Performance Validation 

#### Feature Engineering

Our approach for feature engineering is as follows:
1. Only use correlated attributes from the outcomes of exploratory data analysis: `GrLivArea`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `YearBuilt`, `YearRemodAdd`, `KitchenQual`, `OverallCond`, and `OverallQual`.
2. Instead of feeding attributes directly, we will create bins out of target variable, group and aggregate attributes on the bin values (a concept inspired by histogram), and expand aggregrated features with sale prices matching the bins. 

This way of feature engineering achieves the following:
1. Efficient model training process due to small number of variables (9 instead of 23), hopefully without sacrificing the performance (due to the strong correlation nature). 
2. A feature engineering process that would allow solving this ML problem both as a regression and classification problem. The `SalePrice` bins can also be treated as house price classes. 

##### Binning SalePrice

We followed a simple approach. We created histogram out of the `SalePrice`, picked the highest value from each histogram bin as the chosen `SalePrice` bin values, and mapped the `SalePrice` to these chosen values. There were 39 bins. 

##### Categorical features

There were three categorical attributes. We followed the approach below:
1. For each categorical variable, we created a dummy variable for each of its categories. There were three categorical variables with 10, 10, and 5 categories, which created 25 dummy variables. Some category values were missing. So, we ultimately had 23 dummy variables. 
2. Then for each 23 dummy variables, we grouped them based on `SalePrice` bin values and aggregated the group using `sum` and `mean` functions. These process ultimately gave us 46 features.  

##### Numerical features

There were four chosen numerical attributes. For each of them, we grouped them based on `SalePrice` bin values and aggregated the group using `count`, `mean`, `max`, `min`, and `sum` functions. These process ultimately gave us 20 features taking the total count of feature to 66.

##### Temporal features

There were two chosen temporal variables, each representing years. we picked the highest year in one of the column and used as the most recent year. Then we calculated number of years using each value in the temporal columns from the most recent date using that value. These gave us 2 features, taking the total count of features to 68.


##### Combining features

After combining features from all three threads, we got 69 variables, 68 from the features and 1 from the `SalePrice` bin. Then we joined that dataset with the `SalePrice` and `SalePrice` bins. That gave us 70 x 1460 matrix. 

##### Splitting the features for training and testing

We decided split the feature matrix for training and testing purposes. The training dataset has the 80% of the dataset. Although, during hyperparameter tuning, we sampled 50% of the training dataset again. We will use the remaining 20% for testing model on known but unseen data.

#### Hyperparameter tuning

We decided to use Random Search technique over an [XGBoost estimator](https://xgboost.readthedocs.io/en/stable/) for the following parameter configurations:

- `n_estimators`: 500, 550, 600, 650, ..., 2000 
- `learning_rate`: 0.05, 0.06, 0.07 
- `max_depth`: 3, 5, 7
- `min_child_weight`: 1, 1.5, 2

The parameter optimisation ran for about 1.12 minutes. It generated an r2 score of 0.99 and 0.97 on full training feature and testing feature respectively with its best parameter, which happens to be `n_estimators`: 550, `min_child_weight`: 1.5, `max_depth`: 5, and `learning_rate`: 0.06. 

The estimator with the best parameter had the following feature importance score for the top 10 contributing features:

![Feature Importance Optimisation](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/feature_importance_opt.png)

#### Model training

The model is trained using XGBoost using the best parameters of the hyper parameter tuning over full training feature matrix. It only took a few seconds and generated generated an r2 score of 1 and 0.99 on full training feature and testing feature matrix respectively. 

The estimator had the following feature importance score for the top 10 contributing features:

![Feature Importance Optimisation](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/feature_importance_ml.png)

#### Saving and using the model

We saved the model in the `joblib` serialized format. The serialized modeled is loaded in memory again and tested against `inherited_houses` dataset that generated good looking predictions. 

#### Evaluating if predicted and actual values have similar correlations to housing attributes

We wanted to validate that the size (numerical) attributes would yield higher `SalePrice` for its larger values and lower for its lower values. Similarly validation needs to happen on condition and age attributes. 

To validate the hypothesis, we compared the correlation of housing attributes to both actual and predicted `SalePrice` based on testing dataset. The following table demonstrated that the actual and predicted (on data unused in training) `SalePrice` are quite strongly correlated. 

![Correlation Predicted Actuals](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/correlation_predicted_actuals.png)

The following scatterplots demonstrates that the predicted (on data unused in trainin) `SalePrice` generally increase with the that of the house size, condition, and age. It shows correlation to the columns mentioned above similarly to the actual sale price.

![Correlation Predicted Actuals Scatter](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/correlation_predicted_actuals_scatter.png)

### How key results are demonstrated in a dashboard

## Planning & Execution

The project followed a simple agile method, with the 5 main epics:
- Bootstrap
- Data collection
- Data analysis
- Model training
- Dashboard
- Documentation

The epics were executed with one or more user stories in the GitHub Project [Heritage Housing](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=). The following image provides a snapshot of the project:

![Tracking user stories in GitHub Project](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/github_project_heritage_housing.png)

The project is designed as a kanban board with three columns: `To Do`, `In Progress`, `Done`.

Each user story is created by adding an item in the `To Do` column. When work starts for a user story, it is moved to `In Progress`. When the work regarding the user story finishes, it is moved to `Done` column. The following figure demonstrates a user story. 

![User story](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/user_story.png)

### Execution

Each user story is linked to a GitHub issue as demonstrated by the following figure. It allows connected the user story to relevant GitHub branc/pull request. 

![Issue connected to user story](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/issues_connected_to_user_stories.png)

The execution of the user story typically starts at the codebase, with a branch off the main branch. Once the development is done, the branch is merged to the main branch using a pull request (see below figure).

![Pull request](https://github.com/fokhrun/heritage_housing/blob/documentation/doc_images/pull_request.png)

### User Stories

|  Epic | User Stories |
|---|---|
| Bootstrap | [Starter repository](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=&pane=issue&itemId=46559138) |
|  | [Local development environment](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=epic%3ABootstrap&pane=issue&itemId=46693177) |
| Data collection | [Prepare for data collection](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=epic%3A%22Data+collection%22&pane=issue&itemId=46690864)
|  | [Data collection notebook](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=epic%3A%22Data+collection%22&pane=issue&itemId=46560003) |
| Data Analysis | [Exploratory data analysis](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=epic%3A%22Data+Analysis%22&pane=issue&itemId=46694465)
| Model Training | [Model training, optimization and validation](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=epic%3A%22Model+Training%22&pane=issue&itemId=46690073) |
| Dashboard | [Dashboard planning, designing, and development](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=&pane=issue&itemId=46690128) |
|  | [Dashboard deployment and release](https://github.com/fokhrun/heritage_housing/issues/12) |
| Documentation | [Documentation](https://github.com/users/fokhrun/projects/3/views/1?filterQuery=&pane=issue&itemId=49358304) |

## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into <a href="https://app.codeanywhere.com/" target="_blank" rel="noreferrer">CodeAnywhere</a> with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and <code>pip3 install -r requirements.txt</code>

1. In the terminal type <code>pip3 install jupyter</code>

1. In the terminal type <code>jupyter notebook --NotebookApp.token='' --NotebookApp.password=''</code> to start the jupyter server.

1. Open port 8888 preview or browser

1. Open the jupyter_notebooks directory in the jupyter webpage that has opened and click on the notebook you want to open.

1. Click the button Not Trusted and choose Trust.

Note that the kernel says Python 3. It inherits from the workspace so it will be Python-3.8.12 as installed by our template. To confirm this you can use <code>! python --version</code> in a notebook code cell.


## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In your Cloud IDE, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.

## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace. 
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|





## Business Requirements
As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them).


## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* In case you would like to thank the people that provided support through this project.

