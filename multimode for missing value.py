import pandas as pd
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# 加载数据
province_data_path = r'IW_L2401-province-1.csv'
province_data = pd.read_csv(province_data_path)
from skopt import plots
import matplotlib.pyplot as plt

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



# 模型列表和贝叶斯优化的参数空间配置
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from skopt.space import Integer, Real, Categorical

models = {
    'RandomForest': (RandomForestRegressor(), [
        Integer(10, 1000, name="n_estimators"),
        Integer(1, 30, name="max_depth"),
        Integer(2, 10, name="min_samples_split"),
        Real(0.1, 0.999, name="max_features"),
        Integer(1, 10, name="min_samples_leaf"),
        Categorical([True, False], name="bootstrap"),
        Integer(0.0, 0.5, name="min_weight_fraction_leaf")
    ]),
    'SVR': (SVR(), [
        Real(1e-6, 1e+6, name="C", prior="log-uniform"),
        Real(1e-6, 1e+1, name="epsilon", prior="log-uniform"),
        Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name="kernel"),
        Integer(1, 5, name="degree"),
        Real(1e-5, 1e0, name="gamma", prior="log-uniform")
    ]),
    'KNeighbors': (KNeighborsRegressor(), [
        Integer(1, 30, name="n_neighbors"),
        Categorical(['uniform', 'distance'], name="weights"),
        Integer(1, 2, name="p"),
        Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name="algorithm"),
        Integer(10, 50, name="leaf_size")
    ]),
    'DecisionTree': (DecisionTreeRegressor(), [
        Integer(1, 30, name="max_depth"),
        Integer(2, 10, name="min_samples_split"),
        Integer(1, 10, name="min_samples_leaf"),
        Categorical([None, 'sqrt', 'log2'], name="max_features"),
        Integer(0, 10, name="max_leaf_nodes", transform="identity")
    ]),
    'XGBRegressor': (XGBRegressor(), [
        Integer(10, 1000, name="n_estimators"),
        Real(0.01, 0.5, name="learning_rate", prior="log-uniform"),
        Integer(1, 30, name="max_depth"),
        Real(0, 1, name="subsample", prior="uniform"),
        Real(0, 1, name="colsample_bytree", prior="uniform"),
        Real(0, 10, name="gamma", prior="uniform")
    ]),
    'LGBMRegressor': (LGBMRegressor(), [
        Integer(10, 1000, name="n_estimators"),
        Real(0.01, 0.5, name="learning_rate", prior="log-uniform"),
        Integer(1, 30, name="max_depth"),
        Real(0.1, 1.0, name="subsample", prior="uniform"),
        Real(0.1, 1.0, name="colsample_bytree", prior="uniform"),
        Integer(0, 100, name="num_leaves")
    ]),
    'MLPRegressor': (MLPRegressor(), [
        Integer(50, 500, name="hidden_layer_sizes"),
        Real(1e-5, 1e-1, name="alpha", prior="log-uniform"),
        Categorical(['identity', 'logistic', 'tanh', 'relu'], name="activation"),
        Real(0.001, 0.1, name="learning_rate_init", prior="log-uniform"),
        Categorical(['constant', 'invscaling', 'adaptive'], name="learning_rate"),
        Integer(200, 500, name="max_iter")
    ])
}



def optimize_and_evaluate(model_name, model, param_space):
    X_train = X[y.notna()]
    y_train = y[y.notna()]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  # 准备你的数据

    @use_named_args(param_space)
    def objective(**params):
        model.set_params(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return -r2_score(y_test, preds)  # 或其他性能指标

    # 执行贝叶斯优化
    res_gp = gp_minimize(objective, param_space, n_calls=50, random_state=0)

    # 使用最优参数
    best_params = {param.name: value for param, value in zip(param_space, res_gp.x)}
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 评估性能
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # 保存结果
    results_df = pd.DataFrame(data={'Actual': y_test, 'Predicted': predictions})
    results_df.to_csv(f"{model_name}_predictions.csv", index=False)

    # 保存R2和MSE
    performance_df = pd.DataFrame(data={'R2': [r2], 'MSE': [mse]})
    performance_df.to_csv(f"{model_name}_performance.csv", index=False)

    print(f"{model_name} - Best Parameters: {res_gp.x} R2: {r2}, MSE: {mse}")

    # 可视化贝叶斯优化结果（这里可能需要根据模型调整参数名称）
    _ = plots.plot_objective(res_gp, dimensions=[dim.name for dim in param_space])
    plt.savefig(f"{model_name}_optimization.png")  # 保存图表为PNG文件
    plt.close()  # 关闭图表，以便于生成下一个模型的图表

    predictions = model.predict(X_missing)
    # 计算不确定性（如果模型支持）
    if hasattr(model, "estimators_"):
        all_preds = np.stack([tree.predict(X_missing) for tree in model.estimators_], axis=0)
        uncertainties = np.std(all_preds, axis=0)
    else:
        uncertainties = np.full(predictions.shape, np.nan)

    # 加入年份和地区信息到结果DataFrame
    results_df = pd.concat([info_df.reset_index(drop=True),
                            pd.DataFrame({'Prediction': predictions, 'Uncertainty': uncertainties})], axis=1)
    results_csv_path = f"{model_name}_predictions_with_year_region.csv"
    results_df.to_csv(results_csv_path, index=False)

    print(f"Results with Year and Region saved to {results_csv_path}")

info_df = province_data_melted_filtered.loc[y.isna(), ['Year', 'Region']]
# 遍历模型列表并调用optimize_and_evaluate函数
for model_name, (model, param_space) in models.items():
    print(f"Optimizing and evaluating {model_name}...")
    optimize_and_evaluate(model_name, model, param_space)

