import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from skopt import plots

# 加载数据
province_data_path = r'IW_L2401-province-1.csv'
province_data = pd.read_csv(province_data_path)

# 数据预处理：去除逗号并转换数值列
for col in province_data.columns[1:]:  # 假设第一列是年份，跳过它
    province_data[col] = province_data[col].astype(str).str.replace(',', '')
    province_data[col] = pd.to_numeric(province_data[col], errors='coerce')

# 将数据转换为长格式
province_data_melted = province_data.melt(id_vars=['Year'], var_name='Region', value_name='Amount')

# 独热编码省份，这里不包括'Globe'列
encoder = OneHotEncoder(sparse=False)
province_encoded = encoder.fit_transform(province_data_melted[province_data_melted['Region'] != 'Globe'][['Region']])
province_encoded_df = pd.DataFrame(province_encoded, columns=encoder.get_feature_names_out())

# 准备模型输入数据，这里排除'Globe'数据
province_data_melted_filtered = province_data_melted[province_data_melted['Region'] != 'Globe']
X = pd.concat([province_encoded_df, province_data_melted_filtered[['Year']].reset_index(drop=True)], axis=1)
y = province_data_melted_filtered['Amount'].apply(pd.to_numeric, errors='coerce')

# 分离有缺失值和无缺失值的数据
X_train = X[y.notna()]
y_train = y[y.notna()]
X_missing = X[y.isna()]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# 定义贝叶斯优化的参数空间
space = [Integer(10, 1000, name="n_estimators"),
         Integer(1, 30, name="max_depth"),
         Integer(2, 10, name="min_samples_split"),
         Real(0.1, 0.999, name="max_features")]

# 定义目标函数，这里我们试图最小化负R2分数作为例子

# 定义目标函数
@use_named_args(space)
def objective(n_estimators, max_depth, min_samples_split, max_features):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_split=min_samples_split, max_features=max_features,
                                  random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# 运行贝叶斯优化
result = gp_minimize(objective, space, n_calls=50, random_state=0, verbose=True)

# 输出最优参数
print("Best parameters:", result.x)

_ = plots.plot_objective(result,
                   dimensions=["n_estimators", "max_depth", "min_samples_split", "max_features"])

plt.show()



# 使用最优参数重新训练模型
best_model = RandomForestRegressor(n_estimators=result.x[0], max_depth=result.x[1],
                                   min_samples_split=result.x[2], max_features=result.x[3],
                                   random_state=42)
best_model.fit(X_train, y_train)

# 预测并评估模型
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse}, R2: {r2}")
# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.subplot(1, 2, 2)
errors = predictions - y_test
plt.hist(errors, bins=25, alpha=0.5)
plt.title('Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
# 预测缺失的产生量
predicted_values = best_model.predict(X_missing)

# 填补缺失的产生量
province_data_melted_filtered.loc[y.isna(), 'Amount'] = predicted_values

# 调整每年的产生量以满足全国总量的限制
for year in province_data['Year']:
    national_total = province_data.loc[province_data['Year'] == year, 'Globe'].values[0]
    predicted_total = province_data_melted_filtered[province_data_melted_filtered['Year'] == year]['Amount'].sum()
    adjustment_factor = national_total / predicted_total if predicted_total else 1
    province_data_melted_filtered.loc[province_data_melted_filtered['Year'] == year, 'Amount'] *= adjustment_factor
# 定义要保存的CSV文件路径
output_csv_path = r'Processed_IW_L2401-province.csv'

# 将处理后的数据保存为CSV文件
province_data_melted_filtered.to_csv(output_csv_path, index=False)

# 打印保存成功的消息
print(f'Data saved to CSV at {output_csv_path}')

# 展示调整后的数据
print(province_data_melted_filtered.head())
