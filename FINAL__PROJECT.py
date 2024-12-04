# %% [markdown]
# <div style="border-radius:10px; padding: 30px; background-color: #c0e5e9; font-size:90%; text-align:left">
#     <p style="font-family:Georgia; font-size:400%;font-weight:bold;text-align:center;color:navy;">
#         📱 Mobile Price Prediction 📱
#     </p>
#     
# <p style="font-family:newtimeroman; font-size:300%;font-weight:bold;text-align:center;color:navy;">
#         📉by Decision Tree, Random Forest & SVM📈
#     </p>
#     
# <p style="font-family:newtimeroman; font-size:200%;font-weight:bold;text-align:center;color:navy;">
#         Members : MOKSHA BHANDARI , HET AMIN
#     </p>
# </div>

# %% [markdown]
# <a id="lib"></a>
# # <p style="background-color:powderblue; font-family:roboto; color:navy; font-size:145%;font-weight:bold; text-align:center; border-radius:25px 10px; padding: 10px">1 - Libraries 📚</p>
# 
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# <a id="data"></a>
# # <p style="background-color:powderblue; font-family:roboto; color:navy; font-size:145%;font-weight:bold; text-align:center; border-radius:25px 10px; padding: 10px">2 - Importing Dataset🗄</p>
# 
# 

# %%
data = pd.read_csv('CellPhone_train.csv')
data

# %%
df=pd.DataFrame(data)
df

# %% [markdown]
# <a id="datades"></a>
# # <p style="background-color:powderblue; font-family:roboto; color:navy; font-size:145%;font-weight:bold; text-align:center; border-radius:25px 10px; padding: 10px">3 - Dataset Description📖</p>
# 
# 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; font-size:115%; text-align:left">
# 
# <h3 align="left"><font color=cyan>Dataset Description:</font></h3>
#     
# | __Index__      | __Variable__ | __Description__      |
# |:----|:----|:----|
# | 1  | __battery_power__ | _The maximum energy capacity of the mobile phone's battery, measured in milliamp hours (mAh)._ |
# | 2  | __blue__ | _Indicates the presence of Bluetooth functionality (1: Yes, 0: No)._ |
# | 3  | __clock_speed__ | _The processing speed of the mobile phone's microprocessor, measured in gigahertz (GHz)._ |
# | 4  | __dual_sim__ | _Indicates whether the mobile phone supports dual SIM cards (1: Yes, 0: No)._ |
# | 5  | __fc__ | _The resolution of the front camera, measured in megapixels (MP)._ |
# | 6  | __four_g__ | _Indicates whether the mobile phone supports 4G network connectivity (1: Yes, 0: No).._ |
# | 7  | __int_memory__ | _The internal storage capacity of the mobile phone, measured in gigabytes (GB)._ |
# | 8  | __m_dep__ | _The thickness of the mobile phone, measured in centimeters (cm)._ |
# | 9  | __mobile_wt__ | _The weight of the mobile phone, measured in grams (g)._ |
# | 10  | __n_cores__ | _The number of cores in the mobile phone's processor._ |
# | 11  | __pc__ | _The resolution of the primary (rear) camera, measured in megapixels (MP)._ |
# | 12  | __px_height__ | _The height of the screen's resolution, measured in pixels._ |
# | 13  | __px_width__ | _The width of the screen's resolution, measured in pixels._ |
# | 14  | __ram__ | _The amount of Random Access Memory (RAM) available in the mobile phone, measured in megabytes (MB)._ |
# | 15  | __sc_h__ | _The height of the mobile phone's screen, measured in centimeters (cm)._ |
# | 16  | __sc_w__ | _The width of the mobile phone's screen, measured in centimeters (cm)._ |
# | 17  | __talk_time__ | _The maximum duration the phone can be used for talking on a single battery charge, measured in hours._ |
# | 18  | __three_g__ | _Indicates whether the mobile phone supports 3G network connectivity (1: Yes, 0: No)._ |
# | 19  | __touch_screen__ | _Indicates whether the mobile phone has a touch-sensitive screen (1: Yes, 0: No)._ |
# | 20  | __wifi__ | _Indicates whether the mobile phone supports Wi-Fi connectivity (1: Yes, 0: No)._ |
# | 21  | __price_range__ | _The target variable indicating the price range of the mobile phone, with values: 0 (low cost), 1 (medium cost), 2 (high cost), and 3 (very high cost)._ |

# %%
df.info()

# %%
df.isna().sum()

# %%
df.duplicated()

# %%
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# %%
df.describe(include='all')

# %% [markdown]
# <a id="preprocessing"></a>
# # <p style="background-color:powderblue; font-family:roboto; color:navy; font-size:145%;font-weight:bold; text-align:center; border-radius:25px 10px; padding: 10px">4 - Preprocessing🎓</p>
# 
# 

# %%
df.drop(['px_height', 'px_width', 'sc_h', 'sc_w', 'm_dep'], axis=1, inplace=True)

# %%
# price_range_counts = df['price_range'].value_counts()
# plt.figure(figsize=(4, 4))
# plt.pie(price_range_counts, labels=price_range_counts.index, autopct='%1.1f%%', startangle=140, colors=['blue', 'red' , 'yellow','green'])
# plt.title('price_range Distribution')
# plt.axis('equal')  
# plt.show()

# categorical_features = [ 'blue', 'wifi']
# for feature in categorical_features:
#     plt.figure(figsize=(5, 2.5))
#     sns.countplot(x=feature, hue='price_range', data=df, palette='viridis')
#     plt.title(f'Distribution of {feature} by price_range')
#     plt.show()



# %%
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True), annot=True,linewidth=.5,cmap="coolwarm",mask=np.triu(df.corr(numeric_only=True)));

# %%
# categorical_features = ['blue','touch_screen', 'wifi' , 'four_g' , 'dual_sim']


# for feature in categorical_features:
    
#     feature_price_range = df.groupby([feature, 'price_range']).size().unstack().fillna(0)
    

#     for level in feature_price_range.index:
#         plt.figure(figsize=(4, 4))
#         plt.pie(feature_price_range.loc[level], labels=feature_price_range.columns, autopct='%1.1f%%', startangle=140, colors=['blue', 'orange' , 'red' , 'gray'])
#         plt.title(f'price_range Distribution for {feature} = {level}')
#         plt.axis('equal') 
#         plt.show()

# %%
# sns.set(style="whitegrid")

# features1 = ['ram', 'battery_power', 'int_memory' , 'clock_speed']
# df[features1].hist(bins=15, figsize=(15, 10), layout=(2, 3) ,  color='green')
# plt.suptitle('Histogram of Features')
# plt.show()


# plt.figure(figsize=(15, 10))
# for i, feature in enumerate(features1, 1):
#     plt.subplot(2, 3, i)
#     sns.boxplot(y=df[feature])
#     plt.title(f'Box plot of {feature}')
# plt.tight_layout()
# plt.show()


# %%
df

# %% [markdown]
# <a id="DT"></a>
# # <p style="background-color:powderblue; font-family:roboto; color:navy; font-size:145%;font-weight:bold; text-align:center; border-radius:25px 10px; padding: 10px">5 - Model Training without Hyperparameter Tunning 📊</p>
# 
# 

# %%
X = df.drop('price_range', axis=1)
y = df['price_range']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)



# Decision Tree
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

# Random Forest
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# SVM
svm_model = SVC(random_state=0)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

# %% [markdown]
# <a id="DT"></a>
# # <p style="background-color:powderblue; font-family:roboto; color:navy; font-size:145%;font-weight:bold; text-align:center; border-radius:25px 10px; padding: 10px">6 - Model Training with Hyperparameter Tuning 📊</p>
# 
# 

# %%
X = df.drop('price_range', axis=1)
y = df['price_range']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Decision Tree
dt_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), dt_params, cv=5)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
dt_pred = dt_best.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Best Parameters:", dt_grid.best_params_)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

# Random Forest
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=0), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Best Parameters:", rf_grid.best_params_)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# SVM
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(SVC(random_state=0), svm_params, cv=5)
svm_grid.fit(X_train, y_train)
svm_best = svm_grid.best_estimator_
svm_pred = svm_best.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVM Best Parameters:", svm_grid.best_params_)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

# Compare Models
accuracies = {'Decision Tree': dt_accuracy, 'Random Forest': rf_accuracy, 'SVM': svm_accuracy}
best_model = max(accuracies, key=accuracies.get)
print("Best Model:", best_model, "with accuracy:", accuracies[best_model])

# %%
data_test = pd.read_csv('CellPhone_test.csv')
data_test

# %%
dftest=pd.DataFrame(data_test)
dftest

# %%
dftest.dropna(inplace=True)
dftest.drop_duplicates(inplace=True)
dftest.drop(['px_height', 'px_width', 'sc_h', 'sc_w', 'm_dep' , 'id'], axis=1, inplace=True)

# %%
X_test1 =dftest.copy()

X_test1_scaled = scaler.transform(X_test1)

dftest['price_test_with_decision_tree'] = dt_best.predict(X_test1_scaled)
dftest['price_test_with_random_forest'] = rf_best.predict(X_test1_scaled)
dftest['price_test_with_svm'] = svm_best.predict(X_test1_scaled)


# %%
dftest

# %%
columns_to_extract = ['price_test_with_decision_tree', 'price_test_with_random_forest', 'price_test_with_svm']
extracted_columns = dftest[columns_to_extract]

# %%
extracted_columns


