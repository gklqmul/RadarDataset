# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from scipy.stats import pearsonr

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor


# class TreeCalibrator:
#     def __init__(self):
#         self.models = []

#     def fit(self, X, y):
#         for i in range(y.shape[1]):
#             model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
#             model.fit(X[:, i].reshape(-1, 1), y[:, i])
#             self.models.append(model)

#     def predict(self, X):
#         adjusted = np.zeros_like(X, dtype=float)
#         for i, model in enumerate(self.models):
#             adjusted[:, i] = model.predict(X[:, i].reshape(-1, 1))
#         return adjusted

#     def evaluate(self, X, y_true):
#         y_pred = self.predict(X)
#         results = []
#         for i in range(y_true.shape[1]):
#             mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
#             rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
#             mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / np.clip(y_true[:, i], 1e-8, None))) * 100
#             corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
#             results.append((mae, rmse, mape, corr))
#         return y_pred, results
# class KNNCalibrator:
#     def __init__(self, n_neighbors=3):
#         self.n_neighbors = n_neighbors
#         self.models = []

#     def fit(self, X, y):
#         for i in range(y.shape[1]):
#             model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
#             model.fit(X[:, i].reshape(-1, 1), y[:, i])
#             self.models.append(model)

#     def predict(self, X):
#         adjusted = np.zeros_like(X, dtype=float)
#         for i, model in enumerate(self.models):
#             adjusted[:, i] = model.predict(X[:, i].reshape(-1, 1))
#         return adjusted

#     def evaluate(self, X, y_true):
#         y_pred = self.predict(X)
#         results = []
#         for i in range(y_true.shape[1]):
#             mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
#             rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
#             mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / np.clip(y_true[:, i], 1e-8, None))) * 100
#             corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
#             results.append((mae, rmse, mape, corr))
#         return y_pred, results
# class PolynomialCalibrator:
#     def __init__(self, degree=2):
#         self.degree = degree
#         self.poly = PolynomialFeatures(degree=degree)
#         self.models = []
    
#     def fit(self, X, y):
#         for i in range(y.shape[1]):
#             xi = X[:, i].reshape(-1, 1)
#             yi = y[:, i]
#             xi_poly = self.poly.fit_transform(xi)
#             model = LinearRegression()
#             model.fit(xi_poly, yi)
#             self.models.append(model)
    
#     def predict(self, X):
#         adjusted = np.zeros_like(X, dtype=float)
#         for i, model in enumerate(self.models):
#             xi = X[:, i].reshape(-1, 1)
#             xi_poly = self.poly.transform(xi)
#             adjusted[:, i] = model.predict(xi_poly)
#         return adjusted
    
#     def evaluate(self, X, y_true):
#         y_pred = self.predict(X)
#         results = []
#         for i in range(y_true.shape[1]):
#             mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
#             rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
#             mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / np.clip(y_true[:, i], 1e-8, None))) * 100
#             corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
#             results.append((mae, rmse, mape, corr))
#         return y_pred, results
       
# def excel_to_nparray(file_path, sheet_name=0, nrows=12, ncols=12):
#     """
#     从Excel文件读取12x12表格并转为NumPy数组
    
#     参数：
#         file_path: Excel文件路径
#         sheet_name: 工作表名或索引（默认第一个工作表）
#         nrows: 读取行数（默认12）
#         ncols: 读取列数（默认12）
    
#     返回：
#         12x12的NumPy数组
#     """
#     # 使用pandas读取Excel（从A1开始）
#     df = pd.read_excel(
#         file_path,
#         sheet_name=sheet_name,
#         header=None,       # 无列名
#         nrows=nrows,
#         usecols=range(ncols) # 读取前12列
#     )
    
#     # 转为NumPy数组
#     array = df.to_numpy()
    
#     # 验证尺寸
#     if array.shape != (nrows, ncols):
#         raise ValueError(f"读取的数组尺寸为{array.shape}，不是预期的{nrows}x{ncols}")
    
#     return array

# excel_path = "distance&energy.xlsx" 
# data_array = excel_to_nparray(excel_path)
        
# # 分列
# true_vals = data_array[:, :6]
# diff_arr = np.diff(true_vals, axis=1)  # 按列求差
# sum_col = np.sum(diff_arr, axis=1).reshape(-1, 1)  # 按行求和并调整形状
# true_res = np.hstack((diff_arr, sum_col))
# pred_vals = data_array[:, 6:]
# diff_arr = np.diff(pred_vals, axis=1)  # 按列求差
# sum_col = np.sum(diff_arr, axis=1).reshape(-1, 1)  # 按行求和并调整形状
# pred_res = np.hstack((diff_arr, sum_col))

# pred_TUG = pred_res[:,-1]
# true_TUG = true_res[:,-1]
# pred_TUGs = np.column_stack((pred_TUG[0::3], pred_TUG[1::3], pred_TUG[2::3]))
# true_TUGs = np.column_stack((true_TUG[0::3], true_TUG[1::3], true_TUG[2::3]))
# col1 = np.vstack((pred_TUG[0:3], pred_TUG[6:9]))  # 第1~3行 和 第7~9行
# col2 = np.vstack((pred_TUG[3:6], pred_TUG[9:12])) # 第4~6行 和 第10~12行
# pred_TUGLight = (np.hstack((col1, col2))).T
# col1 = np.vstack((true_TUG[0:3], true_TUG[6:9]))  # 第1~3行 和 第7~9行
# col2 = np.vstack((true_TUG[3:6], true_TUG[9:12])) # 第4~6行 和 第10~12行    
# true_TUGLight = (np.hstack((col1, col2))).T

# pred_person = np.column_stack((pred_TUG[:6],pred_TUG[6:12])) # 取前6行作为训练集
# true_person = np.column_stack((true_TUG[:6],true_TUG[6:12])) # 取前6行作为训练集

# print(true_TUGLight.shape)

# # 分组：前6行为训练，后6行为应用
# true_train = true_person[0::2]    # 取前6行作为训练集
# pred_train = pred_person[0::2]   # 取前6行作为训练集

# true_test = true_person      # 所有12行作为测试集
# pred_test = pred_person      # 所有12行作为测试集


# # === 训练 + 应用 ===
# calibrator = TreeCalibrator()
# calibrator.fit(pred_train, true_train)

# adjusted_test_preds, metrics = calibrator.evaluate(pred_test, true_test)

# # 输出结果
# print("✅ 应用阶段的校正后预测值（随机森林）：")
# # temp = np.hstack((true_vals, np.round(adjusted_test_preds)))
# # print(temp)

# print("\n📊 应用阶段每列的 MAE, RMSE, MAPE, Correlation：")
# for i, (mae, rmse, mape, corr) in enumerate(metrics):
#     print(f"列{i+1} → MAE: {mae/18.1:.2f}, RMSE: {rmse/18.1:.2f}, MAPE: {mape:.2f}%, Correlation: {corr:.4f}")
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

def add_noise_to_diff(data, real_values, sigma):
    """
    对原始数据和真实值进行差分，并添加噪声，生成新的增强数据和增强真实值。
    
    参数：
    data (numpy.ndarray): 原始数据，大小为 (12, 12)，其中后6列是预测值。
    real_values (numpy.ndarray): 真实值，大小为 (12, 12)，与数据对应。
    sigma (float): 高斯噪声的标准差。
    
    返回：
    augmented_data (numpy.ndarray): 增强后的数据，大小为 (12, 12)。
    augmented_real_values (numpy.ndarray): 增强后的真实值，大小为 (12, 12)。
    """
    # 计算数据的差分
    diff_data = np.diff(data, axis=0)
    
    # 计算真实值的差分
    diff_real_values = np.diff(real_values, axis=0)
    
    # 在差分值上添加噪声
    noise_data = np.random.normal(0, sigma, diff_data.shape)
    noisy_diff_data = diff_data + noise_data
    
    noise_real_values = np.random.normal(0, sigma, diff_real_values.shape)
    noisy_diff_real_values = diff_real_values + noise_real_values
    
    # 累加生成新数据
    augmented_data = np.zeros_like(data)
    augmented_data[0, :] = data[0, :]  # 初始化第一个数据点
    
    augmented_real_values = np.zeros_like(real_values)
    augmented_real_values[0, :] = real_values[0, :]  # 初始化第一个真实值
    
    for i in range(1, data.shape[0]):
        augmented_data[i, :] = augmented_data[i - 1, :] + noisy_diff_data[i - 1, :]
        augmented_real_values[i, :] = augmented_real_values[i - 1, :] + noisy_diff_real_values[i - 1, :]
    
    return augmented_data, augmented_real_values

def generate_augmented_data(original_data, original_real_values, sigma, num_augmented_samples=3):
    """
    生成增强后的数据集，通过多次应用数据增强过程来增加数据量。
    
    参数：
    original_data (numpy.ndarray): 原始数据，大小为 (12, 12)。
    original_real_values (numpy.ndarray): 真实值，大小为 (12, 12)。
    sigma (float): 高斯噪声的标准差。
    num_augmented_samples (int): 生成增强数据的数量（增加数据量的倍数）。
    
    返回：
    augmented_data (numpy.ndarray): 增强后的数据，大小为 (12 * num_augmented_samples, 12)。
    augmented_real_values (numpy.ndarray): 增强后的真实值，大小为 (12 * num_augmented_samples, 12)。
    """
    augmented_data_list = []
    augmented_real_values_list = []
    
    for _ in range(num_augmented_samples):
        augmented_data, augmented_real_values = add_noise_to_diff(original_data, original_real_values, sigma)
        augmented_data_list.append(augmented_data)
        augmented_real_values_list.append(augmented_real_values)
    
    # 将增强的数据合并
    augmented_data = np.vstack(augmented_data_list)
    augmented_real_values = np.vstack(augmented_real_values_list)
    
    return augmented_data, augmented_real_values


class TreeCalibrator:
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        for i in range(y.shape[1]):
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X[:, i].reshape(-1, 1), y[:, i])
            self.models.append(model)

    def predict(self, X):
        adjusted = np.zeros_like(X, dtype=float)
        for i, model in enumerate(self.models):
            adjusted[:, i] = model.predict(X[:, i].reshape(-1, 1))
        return adjusted

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        results = []
        for i in range(y_true.shape[1]):
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / np.clip(y_true[:, i], 1e-8, None))) * 100
            corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            results.append((mae, rmse, mape, corr))
        return y_pred, results


# 数据加载函数
def excel_to_nparray(file_path, sheet_name=0, nrows=12, ncols=12):
    """
    从Excel文件读取12x12表格并转为NumPy数组
    
    参数：
        file_path: Excel文件路径
        sheet_name: 工作表名或索引（默认第一个工作表）
        nrows: 读取行数（默认12）
        ncols: 读取列数（默认12）
    
    返回：
        12x12的NumPy数组
    """
    # 使用pandas读取Excel（从A1开始）
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,       # 无列名
        nrows=nrows,
        usecols=range(ncols) # 读取前12列
    )
    
    # 转为NumPy数组
    array = df.to_numpy()
    
    # 验证尺寸
    if array.shape != (nrows, ncols):
        raise ValueError(f"读取的数组尺寸为{array.shape}，不是预期的{nrows}x{ncols}")
    
    return array


# LOOCV实现
def loocv(X, y, calibrator_class):
    n_samples = X.shape[0]
    all_metrics = []

    for i in range(0, n_samples, 2):  # 每次跳过两个样本
        # 留出2个样本作为测试集，其余作为训练集
        X_train = np.delete(X, range(i, i+2), axis=0)
        y_train = np.delete(y, range(i, i+2), axis=0)
        X_test = X[i:i+2, :]
        y_test = y[i:i+2, :]

        # 初始化校准器并训练
        calibrator = calibrator_class()
        calibrator.fit(X_train, y_train)

        # 评估模型
        _, metrics = calibrator.evaluate(X_test, y_test)
        all_metrics.append(metrics)

    # 汇总结果
    avg_metrics = np.mean(all_metrics, axis=0)
    return avg_metrics

def kfold_cv(X, y, model_class, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        model = model_class()
        model.fit(X_train, y_train)
        
        # 预测并评估
        preds, metrics = model.evaluate(X_test, y_test)
        all_metrics.append(metrics)
    
    # 汇总所有的评估指标
    avg_metrics = np.mean(all_metrics, axis=0)
    
    return avg_metrics

# 主函数
if __name__ == "__main__":
    excel_path = "distance&energy.xlsx"
    data_array = excel_to_nparray(excel_path)
    sigma = 0.05  # 高斯噪声的标准差
    true_vals = data_array[:, :6]
    pred_vals = data_array[:, 6:]
    # 生成增强后的数据集，增加到 36x12
    augmented_data, augmented_real_values = generate_augmented_data(pred_vals, true_vals, sigma, num_augmented_samples=3)
    # 分列
    diff_arr = np.diff(augmented_real_values, axis=1)  # 按列求差
    sum_col = np.sum(diff_arr, axis=1).reshape(-1, 1)  # 按行求和并调整形状
    true_res = np.hstack((diff_arr, sum_col))

    diff_arr = np.diff(augmented_data, axis=1)  # 按列求差
    sum_col = np.sum(diff_arr, axis=1).reshape(-1, 1)  # 按行求和并调整形状
    pred_res = np.hstack((diff_arr, sum_col))
#    pred_TUGspeed = np.column_stack((pred_TUG[0::3], pred_TUG[1::3], pred_TUG[2::3]))
#     true_TUGspeed = np.column_stack((true_TUG[0::3], true_TUG[1::3], true_TUG[2::3]))
    pred_TUG_2d = pred_res[:,-1]
    true_TUG_2d = true_res[:,-1]
    pred_TUG = pred_TUG_2d.reshape(-1, 1)
    true_TUG = true_TUG_2d.reshape(-1, 1)
    col1 = np.vstack((pred_TUG[0:3],pred_TUG[6:9],pred_TUG[12:15], pred_TUG[18:21], pred_TUG[24:27], pred_TUG[30:33])) 

    col2 = np.vstack((pred_TUG[3:6], pred_TUG[9:12], pred_TUG[15:18], pred_TUG[21:24], pred_TUG[27:30], pred_TUG[33:36]))  
    pred_TUGLight = np.hstack((col1, col2))
    col11 = np.vstack((true_TUG[0:3],  true_TUG[6:9],   true_TUG[12:15], 
                  true_TUG[18:21], true_TUG[24:27], true_TUG[30:33])) 

    col21 = np.vstack((true_TUG[3:6],   true_TUG[9:12],  true_TUG[15:18], 
                  true_TUG[21:24], true_TUG[27:30], true_TUG[33:36])) 
    true_TUGLight = np.hstack((col11, col21))

    pred_person = np.column_stack((pred_TUG[:6],pred_TUG[6:12])) # 取前6行作为训练集
    true_person = np.column_stack((true_TUG[:6],true_TUG[6:12])) # 取前6行作为训练集

    
    metrics = kfold_cv(pred_TUGLight, true_TUGLight, TreeCalibrator, k=5)

    # 输出每列的平均性能指标
    print("\n📊 每列的平均 MAE, RMSE, MAPE, Correlation：")
    for i, (mae, rmse, mape, corr) in enumerate(metrics):
        print(f"列{i+1} → MAE: {mae/18.1:.2f}, RMSE: {rmse/18.1:.2f}, MAPE: {mape:.2f}%, Correlation: {corr:.4f}")
