
# Dataset 

Income dataset available on the **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)**. The problem statement here is to predict whether the income exceeds 50k a year or not based on the census data.

The Census Income dataset has 32561 entries. Each entry contains the following information about an individual:

salary (target feature/label): whether or not an individual makes more than $50,000 annually. (<= 50K, >50K)
age: the age of an individual. (Integer greater than 0)
workclass: a general term to represent the employment status of an individual. (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
fnlwgt: this is the number of people the census believes the entry represents. People with similar demographic characteristics should have similar weights. There is one important caveat to remember about this statement. 
        That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.(Integer greater than 0)
education: the highest level of education achieved by an individual. (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.)
education-num: the highest level of education achieved in numerical form. (Integer greater than 0)
marital-status: marital status of an individual. Married-civ-spouse corresponds to a civilian spouse while Married-AF-spouse is a spouse in the Armed Forces. 
                Married-spouse-absent includes married people living apart because either the husband or wife was employed and living at a considerable distance from home (Married-civ-spouse, Divorced, Never-married, Separated, 
                Widowed, Married-spouse-absent, Married-AF-spouse)
occupation: the general type of occupation of an individual. (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, 
            Priv-house-serv, Protective-serv, Armed-Forces)
relationship: represents what this individual is relative to others. For example an individual could be a Husband. Each entry only has one relationship attribute. (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
race: Descriptions of an individual’s race. (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
sex: the biological sex of the individual. (Male, female)
capital-gain: capital gains for an individual. (Integer greater than or equal to 0)
capital-loss: capital loss for an individual. (Integer greater than or equal to 0)
hours-per-week: the hours an individual has reported to work per week. (continuous)
native-country: country of origin for an individual (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, 
               Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, 
               Holand-Netherlands)

Having examined the features above, it can be concluded that some of the variables, such as fnlwgt are not related directly to the target variable income and are not self-explanatory. 
Therefore, they can be removed from the Dataset for Machine Learning modeling. 
The continuous variable fnlwgt represents final weight, which is the number of units in the target population that the responding unit represents. 
The variable education_num stands for the number of years of education in total, which is a continuous representation of the discrete variable education. 
Similarly, education gives the same information as education_num does, but in a categorical manner. 
The variable relationship represents the responding unit’s role in the family. capital_gain and capital_loss are income from investment sources other than wage/salary.

# Aim of the Project

Applying Exploratory Data Analysis (EDA) and implement the Machine Learning Algorithms;
1. Analyzing the characteristics of individuals according to income groups
2. Preparing data to create a model that will predict the income levels of people according to their characteristics (So the "salary" feature is the target feature)
3. Feature importance analysis
4. Model selection process and parameter tuning
5. Training the final model
6. Saving it to a file with pickle
7. Loading the model
8. Simple prediction
9. Deployment

# How to run the Project

1. You may find the data with the name DATA in the folder.
2. In notebook:
	- Data preparation and data cleaning
	- EDA, feature importance analysis
	- Model selection process and parameter tuning
	- Training the final model
	- Saving it to a file with pickle
	- Loading the model
	- Simple prediction
3. train.py
	- Training the final model
	- SSaving it to a file with pickle
4. predict.py
	- Loading the model
	- Serving it via a web service 
		- with Flask
		- with Streamlit
5. mlzoomcamp
	- The folder for env
	- Env was created via venv

6. You may activate environment via below codes:
   - mlzoomcamp/Scripts/activate (vscode)
   - source mlzoomcamp/Scripts/activate (gitbash)
   - source mlzoomcamp/bin/activate (mac terminal use "bin" instead of Scripts) 

   and then just code "streamlit run mlzoomcamp_app.py" 

Dockerfile for running the service  ------> I haven't done it yet

7. Deployment

You may find the deployment on the below link. I did the deployment via streamlit.

https://mlzoomcampmidtermproject-a35spnhsuss2wmkqqgpedg.streamlit.app/



