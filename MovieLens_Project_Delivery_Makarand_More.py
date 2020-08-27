#!/usr/bin/env python
# coding: utf-8

# ### Project - MovieLens Data Analysis
# 
# The GroupLens Research Project is a research group in the Department of Computer Science and Engineering at the University of Minnesota. The data is widely used for collaborative filtering and other filtering solutions. However, we will be using this data to act as a means to demonstrate our skill in using Python to “play” with data.
# 
# ### Datasets Information:
# 
# - Data.csv: It contains information of ratings given by the users to a particular movie. Columns: user id, movie id, rating, timestamp
# 
# - item.csv: File contains information related to the movies and its genre.
# 
# - Columns: movie id, movie title, release date, unknown, Action, Adventure, Animation, Children’s, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
# 
# - user.csv: It contains information of the users who have rated the movies. Columns: user id, age, gender, occupation, zip code
# 
# ### Objective:
# 
# `To implement the techniques learnt as a part of the course.`
# 
# ### Learning Outcomes:
# - Exploratory Data Analysis
# 
# - Visualization using Python
# 
# - Pandas – groupby, merging 
# 
# 
# #### Domain 
# `Internet and Entertainment`
# 
# **Note that the project will need you to apply the concepts of groupby and merging extensively.**

# In[1070]:


#Student Name : Makarand More


# #### 1. Import the necessary packages - 2.5 marks

# In[1160]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# #### 2. Read the 3 datasets into dataframes - 2.5 marks

# In[1161]:


UR_Data=pd.read_csv('Data.CSV') # Import user ratings, the dataset file named 'Data.csv'
Movie_Data=pd.read_csv('item.csv') # Import movies info, the dataset file named 'item.csv'
User_Data=pd.read_csv('user.csv') # Import user info, the dataset file named 'user.csv'


# #### 3. Apply info, shape, describe, and find the number of missing values in the data - 5 marks
#  - Note that you will need to do it for all the three datasets seperately

# In[1162]:


#1st shape, info, describe and find missing values for UR_Data i.e for Data.csv
UR_Data.shape


# In[1163]:


UR_Data.info()


# In[1164]:


UR_Data.describe()


# In[1165]:


print(UR_Data.isnull().sum()) #missing value count in dataframe for each column
print('')
UR_Data.isnull().sum().sum() #missing value count in dataframe for both row and column


# In[1166]:


#1st shape, info, describe and find missing values for Movie_Data i.e for item.csv
Movie_Data.shape


# In[1167]:


Movie_Data.info()


# In[1168]:


Movie_Data.describe()


# In[1169]:


print(Movie_Data.isnull().sum()) #missing value count in dataframe for each column
print('')
Movie_Data.isnull().sum().sum() #missing value count in dataframe for both row and column


# In[1170]:


#1st shape, info, describe and find missing values for User_Data i.e for user.csv
User_Data.shape


# In[1171]:


User_Data.info()


# In[1172]:


User_Data.describe()


# In[1173]:


print(User_Data.isnull().sum()) #missing value count in dataframe for each column
print('')
User_Data.isnull().sum().sum() #missing value count in dataframe for both row and column


# #### 4. Find the number of movies per genre using the item data - 2.5 marks

# In[1229]:


# use sum on the default axis
Movie_Data.sum() # or we can mentioned axis=0, which is default


# #### 5. Find the movies that have more than one genre - 5 marks

# In[1175]:


#hint: use sum on the axis = 1
#Option 1 
Movie_Data_Result= Movie_Data[Movie_Data.iloc[:,2:].sum(axis=1)>1]
Movie_Data_Result['movie title']


# #### 6. Drop the movie where the genre is unknown - 2.5 marks

# In[1176]:


#option 1 using drop
Movie_Data.drop(Movie_Data.loc[Movie_Data.unknown == 1].index,inplace=False)  #not deleting permantely with inplace=False
#option 2 using simple filter to exclude data
#Movie_Data1=Movie_Data[Movie_Data.unknown == 0]
#Movie_Data1


# ### 7. Univariate plots of columns: 'rating', 'Age', 'release year', 'Gender' and 'Occupation' - 10 marks

# In[1177]:


# HINT: use distplot for age and countplot for gender,ratings,occupation, release year.
# HINT: Please refer to the below snippet to understand how to get to release year from release date. You can use str.split()
# as depicted below.


# In[1178]:


a = 'My*cat*is*brown'
print(a.split('*')[3])

#similarly, the release year needs to be taken out from release date

#also you can simply slice existing string to get the desired data, if we want to take out the colour of the cat

print(a[10:])
print(a[-5:])


# In[1179]:


#your answers here
sns.distplot(User_Data['age']) #creating displot for Age as per 1st Hint


# In[1180]:


sns.countplot(User_Data['gender']) #creating countplot for gender as per 1st Hint


# In[1181]:


sns.countplot(UR_Data['rating']) #creating countplot for rating as per 1st Hint


# In[1200]:


plt.figure(figsize=(40,20))
sns.countplot(User_Data['occupation']) #creating countplot for occupation as per 1st Hint


# In[1183]:



# Option 1 using pd.DatetimeIndex 
#sns.countplot(pd.DatetimeIndex(Movie_Data['release date']).year) # Option 1 using Convert DataTimeIndex

# Option 2 using Split function, 
plt.figure(figsize=(40,20))
sns.countplot(Movie_Data['release date'].str.split("-", expand=True)[2])
#Imp: in Split expend=True is require to get the right Index value


# ### 8. Visualize how popularity of genres has changed over the years - 10 marks
# 
# Note that you need to use the number of releases in a year as a parameter of popularity of a genre

# Hint 
# 
# 1: you need to reach to a data frame where the release year is the index and the genre is the column names (one cell shows the number of release in a year in one genre) or vice versa.
# Once that is achieved, you can either use multiple bivariate plots or can use the heatmap to visualise all the changes over the years in one go. 
# 
# Hint 2: Use groupby on the relevant column and use sum() on the same to find out the nuumber of releases in a year/genre.  

# In[1219]:


#Your answer here
MovieData_Map  = Movie_Data
MovieData_Map['release year'] = pd.DatetimeIndex(MovieData_Map['release date']).year
#MovieData_Map["no of release"]= MovieData_Map.loc[:,'unknown':'Western'].sum(axis=1)
MovieData_Map=MovieData_Map.groupby(['release year']).sum()

MovieData_Map.drop('movie id',axis=1,inplace=True)
plt.figure(figsize=(40,20))
sns.heatmap(MovieData_Map,annot=True,vmin=0, vmax=20)
#sns.barplot(x = "release year",y = "no of release", data = MovieData1)



# ### 9. Find the top 25 movies according to average ratings such that each movie has number of ratings more than 100 - 10 marks
# 
# Hint : 
# 
# 1. First find the movies that have more than 100 ratings(use merge, groupby and count). Extract the movie id in a list.
# 2. Find the average rating of all the movies and sort them in the descending order. You will have to use the .merge() function to reach to a data set through which you can get the ids and the average rating.
# 3. Use isin(list obtained from 1) to filter out the movies which have more than 100 ratings.
# 
# Note: This question will need you to research about groupby and apply your findings. You can find more on groupby on https://realpython.com/pandas-groupby/.

# In[1120]:


#your answer here
#merge one time to be use for mutiple 
#Option 1
Merge_UR_Movie_Data = pd.merge(UR_Data,Movie_Data)
#Movies_Size =Merge_UR_Movie_Data.groupby(['movie title','movie id'])['movie id'].count()>100
Movies_Size =Merge_UR_Movie_Data.groupby(['movie title','movie id']).agg({'rating':[np.size]})
List_MovieId_100R_1=List_MovieId_100R_1[Movies_Size['rating']['size'] >= 100]
Avg_RMovies = Merge_UR_Movie_Data.groupby(['movie title','movie id']).agg({'rating':[np.mean]}).sort_values([('rating', 'mean')], ascending=False)
Movies_Top25= pd.merge(List_MovieId_100R_1,Avg_RMovies,on='movie id').sort_values([('rating', 'mean')], ascending=False)[:25]

print(Movies_Top25)


#Option 2
Movies_Rated=Merge_UR_Movie_Data.groupby(['movie title','movie id']).agg({'rating':[np.size , np.mean]}).sort_values([('rating', 'mean')], ascending=False)
#print (Movies_Rated)
List_MovieId_100R_2=Movies_Rated['rating']['size'] >= 100
Movies_Rated[List_MovieId_100R_2].sort_values([('rating', 'mean')], ascending=False)[:25]


# ### 10. See gender distribution across different genres check for the validity of the below statements - 10 marks
# 
# * Men watch more drama than women
# * Women watch more Sci-Fi than men
# * Men watch more Romance than women
# 

# 1. There is no need to conduct statistical tests around this. Just compare the percentages and comment on the validity of the above statements.
# 
# 2. you might want ot use the .sum(), .div() function here.
# 3. Use number of ratings to validate the numbers. For example, if out of 4000 ratings received by women, 3000 are for drama, we will assume that 75% of the women watch drama.

# In[1195]:


#Merge_UR_Movie_Data.info()
Merge_All_Data=pd.merge(User_Data,Merge_UR_Movie_Data, on='user id') #merge with DataSet created in Q9 
Gender_Info=Merge_All_Data.groupby(['gender']).sum()
print((Gender_Info['Drama'] / Gender_Info['Drama'].sum())*100) #Men watch more drama than women
print((Gender_Info['Sci-Fi'] / Gender_Info['Sci-Fi'].sum())*100) #Women watch more Sci-Fi than men (Opposite)
print((Gender_Info['Romance'] / Gender_Info['Romance'].sum())*100) #Men watch more Romance than women
Gender_Info1= Gender_Info.iloc[:,5:] / Gender_Info.iloc[:,5:].sum()*100
Gender_Info1


# #### Conclusion:
# 
# 

# # All three data set link to each other using user id and data for more number Male gender then Female. The genre popularity not changes drastically over the years however a greater number of movies release over the years. Also, a greater number of Student category watch movies.
# 
