# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso



warnings.filterwarnings('ignore')
output_dir = 'output_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
df = pd.read_csv('./data/train.csv')

# Splitting columns into Numerical and categorical columns
num_cols = [col for col in df.columns if df[col].dtype != 'O']
cat_cols = [col for col in df.columns if col not in num_cols]

columns_to_drop = ['MiscFeature', 'Alley', 'PoolQC','Fence']
df_cleaned = df.drop(columns=columns_to_drop)

df_cleaned['Electrical'].fillna(df_cleaned['Electrical'].mode()[0], inplace=True)
df_cleaned["MasVnrArea"].fillna(df_cleaned["MasVnrArea"].median(), inplace=True)
# print(df[['BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']])
df_cleaned['BsmtQual'].fillna(df_cleaned['BsmtQual'].mode()[0], inplace=True)
df_cleaned['BsmtCond'].fillna(df_cleaned['BsmtCond'].mode()[0], inplace=True)
df_cleaned['BsmtExposure'].fillna(df_cleaned['BsmtExposure'].mode()[0], inplace=True)
df_cleaned['BsmtFinType1'].fillna(df_cleaned['BsmtFinType1'].mode()[0], inplace=True)
df_cleaned['BsmtFinType2'].fillna(df_cleaned['BsmtFinType2'].mode()[0], inplace=True)
df_cleaned['GarageType'].fillna(df_cleaned['GarageType'].mode()[0], inplace=True)
df_cleaned['GarageFinish'].fillna(df_cleaned['GarageFinish'].mode()[0], inplace=True)
df_cleaned['GarageQual'].fillna(df_cleaned['GarageQual'].mode()[0], inplace=True)
df_cleaned['GarageCond'].fillna(df_cleaned['GarageCond'].mode()[0], inplace=True)
df_cleaned["GarageYrBlt"].fillna(df_cleaned["GarageYrBlt"].median(), inplace=True)

dffff=df_cleaned.copy()
categorical_columns = dffff.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    dffff[col] = label_encoder.fit_transform(dffff[col])


imputer = KNNImputer(n_neighbors=5)
columns_to_impute = ['FireplaceQu','LotFrontage','MasVnrType']
dffff[columns_to_impute] = imputer.fit_transform(dffff[columns_to_impute])

df_analysis = dffff.copy()
numerical_data = df_analysis.select_dtypes(include='number').columns.tolist()
categorical_data = df_analysis.select_dtypes(include='object').columns.tolist()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True)[['SalePrice']], annot=True, cmap='coolwarm')
plt.title('Correlation with SalePrice')
plt.savefig(os.path.join(output_dir, 'correlation_with_saleprice.png'))
plt.show()

# Determining if homes that haven't been modified since 1950 have a lower sale price than homes that have
df_analysis['ModifiedSince1950'] = df_analysis['YearRemodAdd'] > 1950

plt.figure(figsize=(10, 6))
sns.boxplot(x='ModifiedSince1950', y='SalePrice', data=df_analysis,color='skyblue')
plt.title('Impact of Modification Since 1950 on Sale Price')
plt.xlabel('Modified Since 1950')
plt.ylabel('Sale Price')
plt.xticks([0, 1], ['Not Modified Since 1950', 'Modified Since 1950'])
plt.savefig(os.path.join(output_dir, 'Modification_impact.png'))
plt.show()

# Testing if building age affects SalePrice
df_analysis['BuildingAge']=df_analysis['YrSold'].max()-df_analysis['YearBuilt']
sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))
sns.regplot(x=df_analysis['BuildingAge'], y=df_analysis['SalePrice'], scatter_kws={'s':30, 'alpha':0.5}, line_kws={'color':'red'})

plt.title('Building Age vs SalePrice', fontsize=16, fontweight='bold')
plt.xlabel('Age of Houses', fontsize=14)
plt.ylabel('Sale Price', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Age_vs_SalePrice.png'))
plt.show()


# Testing if overall quality is affected by modified vs non-modified houses
plt.figure(figsize=(8, 6))

# Overall Quality vs. Modification Status
sns.boxplot(data=df_analysis, x='ModifiedSince1950', y='OverallQual', palette='Set2')
plt.title('Overall Quality vs. Modification Status')
plt.xlabel('Modification Since 1950')
plt.ylabel('Overall Quality')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Quality_vs_Modification.png'))
plt.show()

# Multivarient Analysis
plt.figure(figsize=(25, 15))

corr = df.select_dtypes(exclude=['object']).corr()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, cbar_kws={"shrink": 1}, annot=True)
plt.savefig(os.path.join(output_dir, 'Correlation.png'))
plt.show()

df_sic=df_cleaned.copy()
df_sic=pd.get_dummies(df_sic, columns=['FireplaceQu','MasVnrType'],dtype=int)


categorical_columns = df_sic.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_sic, columns=categorical_columns, drop_first=True)
boolean_columns = df_encoded.select_dtypes(include=['bool']).columns
if len(boolean_columns) > 0:
    df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)

dfff=dffff.copy()

def detect_outliers_iqr(df):
    outlier_indices = []

    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_indices.extend(outliers.index)

        # print(f'Outliers in {column}:', outliers.shape[0])

    return outlier_indices

def outliers(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        # print(f"Cap and floor outliers in '{column}'")

outlier_indices = detect_outliers_iqr(dfff)

from sklearn.preprocessing import RobustScaler
numerical_columns = dfff.select_dtypes(include=['float64', 'int64']).columns

numerical_columns = [col for col in numerical_columns if col in dfff.columns]
scaler = RobustScaler()
dfff[numerical_columns] = scaler.fit_transform(dfff[numerical_columns])

df = dfff.copy()
unique_cols = df.loc[:, df.nunique() == 1].columns
df.drop(unique_cols, axis=1, inplace=True)
df.drop(columns = ['Id', 'MoSold', 'YrSold'], axis=1, inplace=True)
df_new=df.copy()


from datetime import datetime
current_year = datetime.now().year
df_new['house_age'] = current_year - df_new['YearBuilt']

room_columns = df_new[['TotRmsAbvGrd', 'BedroomAbvGr']]
df_new['TotalRooms'] = df_new['TotRmsAbvGrd'] + df_new['BedroomAbvGr']
area_columns = ['GrLivArea', 'TotalBsmtSF','1stFlrSF','2ndFlrSF']

df_new['TotalArea'] = df_new[area_columns].sum(axis=1)
df_new['AreaRatio'] = (df_new['TotalArea'] / df_new['LotArea'])*100
df_new['QualityScore'] = ( df_new['OverallQual'] +  df_new['BsmtQual'] + df_new['HeatingQC'] + df_new['KitchenQual'])/4

X = df_new.drop('SalePrice', axis=1)
y = df_new['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# predicting on train data
y_train_pred = lr_model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, y_train_pred) # Mean Squared Error for Train data
train_rmse = np.sqrt(train_mse) # Root Mean Squared Error for Train data
train_r2 = r2_score(y_train, y_train_pred) # R^2 Score on Train Data
print(f'Training MSE: {train_mse:.3f}')
print(f'Training RMSE: {train_rmse:.3f}')
print(f'Training R-squared: {round(train_r2,2)}%',"\n\n")


# predicting on test data
y_test_pred = lr_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test MSE: {test_mse:.3f}')
print(f'Test RMSE: {test_rmse:.3f}')
print(f'Test R-squared: {round(test_r2,2)}%')


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred, color='b', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='r')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Actual_vs_predicted.png'))
plt.show()



columns = df.columns
columns = columns.drop('SalePrice')
corr_matrix = df.corr()
corr_with_target = corr_matrix['SalePrice'][columns]

# Select the features with the highest correlation coefficients
top_features = corr_with_target.abs().nlargest(n=2)
ridge_model = Ridge(alpha=0.001)
ridge_model.fit(X_train_scaled, y_train)

# predicting on train data
y_train_pred = ridge_model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, y_train_pred) # Mean Squared Error for Train data
train_rmse = np.sqrt(train_mse) # Root Mean Squared Error for Train data
train_r2 = r2_score(y_train, y_train_pred) # R^2 Score on Train Data
print(f'Training RMSE: {train_rmse:.3f}')
print(f'Training R-squared: {round(train_r2,2)}%',"\n\n")


# predicting on test data
y_test_pred = ridge_model.predict(X_test_scaled)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test RMSE: {test_rmse:.3f}')
print(f'Test R-squared: {round(test_r2,2)}%')








