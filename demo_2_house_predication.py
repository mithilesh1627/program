import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.neighbors
# df = pd.read_csv("C:\\Users\\mithilesh\\Downloads\\csv files\\housing.csv")
# #print(df.info())
#
#
# df.hist(bins=60, figsize=(15,15))
# plt.show()
#
#
# df["income_cat"] = np.ceil(df["median_income"]/ 1.5)
# df["income_cat"].where(df["income_cat"] < 5, other=5.0, inplace=True)
#
#
# from sklearn.model_selection import StratifiedShuffleSplit
#
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#
# for train_set, test_set in split.split(df, df['income_cat']):
#     strat_train_set = df.loc[train_set]
#     strat_test_set = df.loc[test_set]
#
#
# strat_train_set.drop(["income_cat"], axis=1, inplace=True)
# strat_test_set.drop(["income_cat"], axis=1, inplace=True)
#
#
# df = strat_train_set.copy()
# df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#             s=df["population"]/100, label="population",
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#             figsize=(15,7))
# plt.legend()
#
# from pandas.plotting import scatter_matrix
# attributes = ['median_house_value', 'median_income',
#              'total_rooms', 'df_median_age']
# scatter_matrix(df[attributes], figsize=(12,8))
#
# df.plot(kind='scatter', x='median_income', y='median_house_value',
#             alpha=0.1, figsize=(8,5))
#
#
# df.head(3)
#
#
# df['rooms_per_household'] = df['total_rooms']/df['households']
# df['bedrooms_per_room'] = df['total_bedrooms']/df['total_rooms']
# df['population_per_household'] = df['population']/df['households']
# df.head(3)
# corr_matrix = df.corr()
# corr_matrix['median_house_value'].sort_values(ascending=False)
#
# df= strat_train_set.drop("median_house_value", axis=1)
# df_labels = strat_train_set['median_house_value'].copy()
#
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
# df_num = df.drop('ocean_proximity', axis=1)
# df_num.head()
#
# from sklearn.base import BaseEstimator, TransformerMixin
# rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
#
# # this component gives us the flexibility to add extra attributes to our pipeline
# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#
#     def __init__(self, add_bedrooms_per_room=True):
#         self.add_bedrooms_per_room = add_bedrooms_per_room
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
#         population_per_household = X[:, population_ix] / X[:, household_ix]
#
#         if self.add_bedrooms_per_room:
#             bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
#             return np.c_[X, rooms_per_household, population_per_household,
#
#                          bedrooms_per_room]
#         else:
#             return np.c_[X, rooms_per_household, population_per_household]
#
#
# # this component allows us to select entire or partial dataframe
# # or in simpler words we can tell which attributes we want in our pipeline
# class DataFrameSelector(BaseEstimator, TransformerMixin):
#
#     def __init__(self, attribute_names):
#         self.attribute_names = attribute_names
#
#     def fit(self, X, y=None): return self
#
#     def transform(self, X): return X[self.attribute_names].values
#
#
# class MyLabelBinarizer(TransformerMixin):
#
#     def __init__(self, *args, **kwargs):
#         self.encoder = LabelBinarizer(*args, **kwargs)
#
#     def fit(self, x, y=0):
#         self.encoder.fit(x)
#         return self
#
#     def transform(self, x, y=0): return self.encoder.transform(x)
#
# num_attribs = list(df_num)
# cat_attribs = ["ocean_proximity"]
# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(df.values)
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import FeatureUnion
# from sklearn.preprocessing import LabelBinarizer
#
# # our numerical pipeline
# num_pipeline = Pipeline([
#                     ('selector', DataFrameSelector(num_attribs)),
#                     ('imputer', SimpleImputer(strategy="median")),
#                     ('attribs_adder', CombinedAttributesAdder()),
#                     ('std_scaler', StandardScaler()),
#                 ])
#
# # our categorical pipeline
# cat_pipeline = Pipeline([
#     ('selector', DataFrameSelector(cat_attribs)),
#     ('label_binarizer', MyLabelBinarizer()),
# ])
#
# # our full pipeline
# full_pipeline = FeatureUnion(transformer_list=[
#     ('num_pipeline', num_pipeline),
#     ('cat_pipeline', cat_pipeline),
# ])
#
# df_prepared = full_pipeline.fit_transform(df)
#
# print(df_prepared)
#
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import cross_val_score
# lin_reg = LinearRegression()
#
#
# scores = cross_val_score(lin_reg, df_prepared, df_labels,
#                         scoring="neg_mean_squared_error", cv=10)
#
#
# # find root mean squared error, scores is an array of negative numbers
# rmse_scores = np.sqrt(-scores)
#
# print("Mean:\t\t ", rmse_scores.mean(),
#       "\nStandard Deviation:", rmse_scores.std())
#
#
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import cross_val_score
# tree_reg = DecisionTreeRegressor()
# scores = cross_val_score(tree_reg, df_prepared, df_labels,
#                         scoring="neg_mean_squared_error", cv=10)
# # find root mean squared error
# rmse_scores = np.sqrt(-scores)
#
#
# print("Mean:\t\t ", rmse_scores.mean(),
#       "\nStandard Deviation:", rmse_scores.std())
# ######################
#
# from sklearn.ensemble import RandomForestRegressor
# forest_reg = RandomForestRegressor()
# forest_reg.fit(df_prepared, df_labels)
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)
# forest_scores = cross_val_score(forest_reg, df_prepared, df_labels,
#                                scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
#
# print("Mean:\t\t ", forest_rmse_scores.mean(),
#       "\nStandard Deviation:", forest_rmse_scores.std())
#
# from sklearn.model_selection import GridSearchCV
# param_grid = [
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#   ]
#
# forest_reg = RandomForestRegressor()
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                           scoring='neg_mean_squared_error')
# grid_search.fit(df_prepared, df_labels)
#
# GridSearchCV(cv=5, error_score='raise-deprecating',
#        estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
#            oob_score=False, random_state=None, verbose=0, warm_start=False),
#        fit_params=None, iid='warn', n_jobs=None,
#        param_grid=[{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}],
#        pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#        scoring='neg_mean_squared_error', verbose=0)
#
# grid_search.best_estimator_
#
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=30, n_jobs=None, oob_score=False,
#            random_state=None, verbose=0, warm_start=False)
#
# cvres = grid_search.cv_results_
# print("{}\t\t {}\n".format('Mean Score','Parameters'))
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     x = np.sqrt(-mean_score)
#     y = params
#     print("{:.2f}\t {}".format(x, y))
#
#
# final_model = grid_search.best_estimator_
# X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test = strat_test_set["median_house_value"].copy()
# # we prepare the data
# X_test_prepared = full_pipeline.transform(X_test)
#
# # do the prediction
# final_predictions = final_model.predict(X_test_prepared)
#
# # find mean square error
# final_mse = mean_squared_error(y_test, final_predictions)
# # find root mean square error
# final_rmse = np.sqrt(final_mse)
# final_rmse


df = pd.read_csv("C:\\Users\\mithilesh\\Downloads\\"
                 "csv files\\housing.csv")
print(df.head(5))

print(df["ocean_proximity"].value_counts())

print(df.describe())
df.hist(bins=50 , figsize=(20,15))
plt.show()

# def split_train_test(df, test_ratio):
#  shuffled_indices = np.random.permutation(len(df))
#  test_set_size = int(len(df) * test_ratio)
#  test_indices = shuffled_indices[:test_set_size]
#  train_indices = shuffled_indices[test_set_size:]
#  return df.iloc[train_indices], df.iloc[test_indices]
#
#
# train_set, test_set = split_train_test(df , 0.2)


from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(df , test_size = 0.2,
                                          random_state=42)
print(len(train_set))
print(len(test_set))
df["income_cat"] = pd.cut(df["median_income"],
                          bins=[0., 1.5, 3.0, 4.5, 6.,
                                    np.Infinity],labels=[1, 2, 3, 4, 5])
df["income_cat"].hist()
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index , test_index in split.split(df,df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set =df.loc[test_index]
strat_test_set["income_cat"].value_counts() / len(strat_test_set)


for set_ in (strat_train_set, strat_test_set):
 set_.drop("income_cat", axis=1, inplace=True)

df= strat_train_set.copy()

df.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
df.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)
plt.show()
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=df["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.show()

corr_matrix = df.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(df[attributes], figsize=(12, 8))

df.plot(kind="scatter", x="median_income", y="median_house_value",
 alpha=0.1)
plt.show()

df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"]=df["population"]/df["households"]

corr_matrix = df.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
df = strat_train_set.drop("median_house_value", axis=1)
df_labels = strat_train_set["median_house_value"].copy()

median = df["total_bedrooms"].median()
df["total_bedrooms"].fillna(median, inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

df_num = df.drop("ocean_proximity", axis=1)
imputer.fit(df_num)

df_cat = df[["ocean_proximity"]]
df_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
df_cat_encoded = ordinal_encoder.fit_transform(df_cat)
df_cat_encoded[:10]

ordinal_encoder.categories_

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
df_cat_1hot = cat_encoder.fit_transform(df_cat)

df_cat_1hot.toarray()

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        df_extra_attribs = attr_adder.transform(housing.values)





from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
df_num_tr = num_pipeline.fit_transform(df_num)


from sklearn.compose import ColumnTransformer
num_attribs = list(df_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),
                                   ("cat",OneHotEncoder(),cat_attribs),
                                   ])

df_prepared = full_pipeline.fit_transform(df)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(df_prepared,df_labels)


some_data = df.iloc[:5]
some_labels = df_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error
df_predictions = lin_reg.predict(df_prepared)
lin_mean = mean_squared_error(df_labels,df_predictions)
lin_rmse = np.sqrt(lin_mean)
print(lin_rmse)


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(df_prepared,df_labels)


df_predictions = tree_reg.predict(df_prepared)
tree_mse = mean_squared_error(df_labels, df_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, df_prepared, df_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, df_prepared, df_labels,
scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(df_prepared, df_labels)
print(forest_reg)
df_predictions = forest_reg.predict(df_prepared)
forest_mse = mean_squared_error(df_labels, df_predictions)
forest_rmse = np.sqrt(tree_mse)
print(forest_rmse)
forest_scores = cross_val_score(lin_reg, df_prepared, df_labels,
scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(df_prepared, df_labels)
grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
loc=squared_errors.mean(),
scale=stats.sem(squared_errors)))