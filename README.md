# Internet Ads

### Project Overview
This purpose for this repository is for a machine learning project to predict whether an image is an advertisment ("ad") or not ("non ad"). 

## Data Pre-processing
Two data files had to be combined and read:

```python
# read and apply column names from file
feature_names = pd.read_csv(data_path + 'column.names.txt', usecols =[0], delimiter =':', skip_blank_lines=True)
feature_names.columns = ['col_name']
feature_names=  feature_names.col_name.values.tolist()
feature_names.append('target')
```

The features included the shape of the image and phrases in the URL, the image's URL and text such as alt text, anchor text, and words occurring near the anchor text.

Some facts about the data:
* There are 3,279 rows (2,821 non ads, 458 ads)
* There are 1,558 columns (3 continous; others binary)
* 28% of instances are missing some of the continuous attributes.
* Class Distribution - number of instances per class: 2,821 non ads, 458 ads.
* height, width and aratio are the only continuous variables

## Data Cleaning
```python
vars_cont = ['height','width','aratio']
'''
    Clean up 3 continuous variables:
        - Replace ? with NA 
        - strip balnks
        - convert to float
'''
dat[vars_cont]= dat[vars_cont].apply(lambda x: x.str.strip()).replace("?",np.nan).astype(np.float)
```

```python
'''
    Clean up Local - this should be 0,1
        - Replace ? with 2 (unknown can be its own category)
        - Convert instances were there are strings for 0 and 1 to integers
'''
dat['local']= dat['local'].replace("?",2)
dat.loc[dat.local == '1','local'] = dat[dat.local == '1'].local.astype(np.int64)
dat.loc[dat.local == '0','local'] = dat[dat.local == '0'].local.astype(np.int64)
```

## Data Exploration
In doing some data exploration, we can see that there are indeed more Non-Ads than Ads, and that some of the continuous variables need to be scaled.

```python

# Plot Ad vs Non Ad histogram
fig, ax = plt.subplots(ncols=1, nrows=1, figsize = (8,6));
data = dat.target.value_counts()
data= data.reset_index()

sns.barplot(data=data, y= 'target', x='index', palette='Set1', ax =ax);
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticklabels(labels, rotation = 0)
ax.set_title('Histogram of Ad vs. Non Ad')
ax.set_xlabel('')
ax.set_ylabel('')
plt.show(fig)
```
![hist](/images/hist.png)

```python

'''
    Box plots of continuous variables vs target
        - Need to scale height & width since these are not on the same scale
'''
var = ['height','width', 'aratio']
fig, axes = plt.subplots(figsize=(14, 6),ncols=3)
for i, v in enumerate(var):
    sns.boxplot(x = 'target', y =v, ax = axes[i], data=dat[[v, 'target']], palette='Set1')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].set_title(v)
plt.show(fig)
```
![boxplots](/images/boxplots.png)

Plotting the continuous variables on a log scale enables us to see that there are linear relationships between the variables and outliers that we can remove.

```python

'''        As Height increases, Aratio decreases - negative correlation expected
        As Width increases, Aratio increases - positive correlation expected
'''
fig, axes = plt.subplots(figsize=(6, 4), ncols=2)
dat.plot.scatter(x = var[0], y = var[2],ax=axes[0], title = 'Height vs Aratio - log', loglog = True)
dat.plot.scatter(x = var[1], y = var[2],ax=axes[1], title = 'Width vs Aratio - log', loglog = True)
```
![scatterplot](/images/scatterplot.png)

After scaling continuous features and removing outliers:
![boxplots_2](/images/boxplots_2.png)

## Correlation
After doing some exploratory analysis, it looks like some of the features are correlated with one another, and there is 
some correlation among the three continuous variables.

```python
corrmat = dat.corr()
# first 100 features - how correlated are they?
# There are quite a few highly correlated features in the data
corr_1 = corrmat.iloc[0:20,0:20]
corr_2 = corrmat.iloc[21:50,21:50]
corr_3 = corrmat.iloc[51:70,51:70]
corr_4 = corrmat.iloc[71:100,71:100]
# plot correlations of 1st 100 features to gain insight
# plot correlation matrix
fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
fig.subplots_adjust(hspace=.5, wspace=.5)
sns.heatmap(corr_1, vmax=.9, square=True, ax=axes[0][0]);
sns.heatmap(corr_2, vmax=.9, square=True, ax=axes[0][1]);
sns.heatmap(corr_3, vmax=.9, square=True, ax=axes[1][0]);
sns.heatmap(corr_4, vmax=.9, square=True, ax=axes[1][1]);
```

![corr](/images/corr.png)


## Exploratory Summary

After doing some exploratory analysis, it is clear that there is are more Non-Ads than Ads in the dataset (84% vs 16%). There are many features in the data, and it looks like a lot of them have majority value equal to 0, so I think that these may not add much information to the data. 

Also, there are many features that are highly correlated in the dataset. Since there are so many features in the data, next I will explore PCA to reduce dimensionality and penalized logistic regression. As a last step, I want to also explore random forests since they have embedded feature importance based on Gini impurity/ information gain.

## PCA and Logistic Regression
