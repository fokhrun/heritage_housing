# Heritage Housing

Heritage housing is an simple ML system that predicts prices of houses in Ames, Iowa, US, by using gradient-boosting based ML regression technique that maps relationships between housing attributes to its sale prices. 

## Table of Content
- [Business Requirements](https://github.com/fokhrun/heritage_housing#business-Requirements-)
- [Dataset](https://github.com/fokhrun/heritage_housing#dataset-)
- [Hypothesis](https://github.com/fokhrun/heritage_housing#hypothesis)
- [Planning & Execution](https://github.com/fokhrun/heritage_housing#planning-executions-)
- [ML System](https://github.com/fokhrun/heritage_housing#ml-system)
    - [Initial Setup]
    - [Data Gathering]
    - [Data Analysis]
    - [Feature Engineering]
    - [Model Training]
    - [Hypothesis Validation]
    - [ML Dashboard]
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

These hypothesis should be validated by the following observations:
- Both the actual and predicted (on data unused in training) SalePrice should be very strongly correlated
- The predicted (on data unused in training) SalePrice should generally increase with the that of the house size, condition, and age. It should show correlation to the columns mentioned above similarly to the actual sale price.

Note that location desirability and room count also have similar effect, but the dataset did not those variables.

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

