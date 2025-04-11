import numpy as np

# data = [
#     [1, 24, 51, 28, 45, 52, 200, 32, 42, 26, 43, 72, 215],
#     [2, 16, 27, 14, 28, 45, 130, 15, 30, 19, 25, 39, 128],
#     [3, 32, 60, 27, 49, 65, 233, 56, 36, 25, 51, 73, 241],
#     [4, 21, 52, 24, 38, 51, 186, 26, 47, 19, 44, 48, 184],
#     [5, 16, 29, 16, 34, 47, 142, 17, 32, 12, 36, 47, 144],
#     [6, 29, 59, 26, 58, 65, 237, 27, 76, 67, 55, 38, 263],
#     [7, 28, 44, 45, 51, 69, 237, 26, 46, 42, 44, 73, 231],
#     [8, 16, 36, 20, 33, 47, 152, 23, 32, 19, 36, 48, 158],
#     [9, 40, 66, 52, 61, 68, 287, 40, 70, 52, 57, 66, 285],
#     [10, 30, 43, 29, 41, 62, 205, 46, 39, 30, 42, 62, 219],
#     [11, 21, 35, 21, 37, 35, 149, 19, 32, 25, 25, 52, 153],
#     [12, 40, 59, 51, 70, 75, 295, 69, 26, 58, 62, 84, 299]
# ]

# # 创建数组并归一化 (跳过第一列ID)
# data_array = np.array([row[1:] for row in data]) / 18.1

# data_array= np.array([
#     [34, 58, 109, 137, 182, 234, 26, 63, 118, 139, 182, 249],
#     [38, 54, 81, 95, 123, 168, 	38,	84,	125,111,136,170],
#     [26, 58, 118, 145, 194, 259, 32,	78,	141,	161,	212,	280],
#     [21, 42, 94, 118, 156, 207, 31,60,119,134,178,221],
#     [31, 47, 76, 92, 126, 173, 29,76,120,97,133,175],
#     [38, 67, 126, 152, 210, 275, 31,65,127,210,265,298],

# ])
# data_array = data_array /18.1
# # 前6列作为ground truth，后6列作为预测值
# ground_truth = data_array[:, :6]  # 形状应为(12,6)
# predicted = data_array[:, 6:]     # 形状应为(12,6)

# # 检查数组形状
# print(f"Ground truth shape: {ground_truth.shape}")
# print(f"Predicted shape: {predicted.shape}")

# def calculate_metrics(actual, predicted):
#     """计算评估指标"""
#     # RMSE
#     rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
#     # MAE
#     mae = np.mean(np.abs(actual - predicted))
    
#     # MAPE (处理除零情况)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mape = np.mean(np.where(actual != 0, np.abs((actual - predicted)/actual), 0)) * 100
    
#     # 相关系数
#     correlation = np.corrcoef(actual.flatten(), predicted.flatten())[0, 1]
    
#     return rmse, mae, mape, correlation

# # 计算每个动作的指标
# metrics = {}
# actions = ["Stand Up", "Walk", "Turn", "Walk back", "Turn and sit down", "End"]
# # actions = ['light','less light']
# for i, action in enumerate(actions):
#     try:
#         rmse, mae, mape, corr = calculate_metrics(
#             ground_truth[:, i], 
#             predicted[:, i]
#         )
#         metrics[action] = {
#             "RMSE": rmse,
#             "MAE": mae,
#             "MAPE": mape,
#             "Correlation": corr
#         }
#     except Exception as e:
#         print(f"Error calculating {action}: {str(e)}")
#         metrics[action] = {
#             "RMSE": np.nan,
#             "MAE": np.nan,
#             "MAPE": np.nan,
#             "Correlation": np.nan
#         }

# # 打印结果
# print("\nEvaluation Metrics:")
# for action, vals in metrics.items():
#     print(f"\n{action}:")
#     print(f"  RMSE: {vals['RMSE']:.4f}")
#     print(f"  MAE: {vals['MAE']:.4f}")
#     print(f"  MAPE: {vals['MAPE']:.2f}%")
#     print(f"  Correlation: {vals['Correlation']:.4f}")

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr

# 真实值和初步预测值
import numpy as np
import pandas as pd

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

def process_array(arr):
    # 计算每列与前一列的差值
    diff_arr = np.diff(arr, axis=1)  # 按列求差
    # 在最后一列加入前面所有差值的和
    sum_col = np.sum(diff_arr, axis=1).reshape(-1, 1)  # 按行求和并调整形状
    # 将差值和和列组合
    result = np.hstack((diff_arr, sum_col))
    return result

try:
        # 替换为你的Excel文件路径
        excel_path = "distance&energy.xlsx" 
        data_array = excel_to_nparray(excel_path)
        
        print("成功读取数组：")
        print(data_array)
        print(f"数组尺寸：{data_array.shape}")
        print(f"数据类型：{data_array.dtype}")
        # 前6列作为ground truth，后6列作为预测值
        true_vals = data_array[:, :6]  # 形状应为(12,6)
        pred_vals = data_array[:, 6:]     # 形状应为(12,6)
        
except FileNotFoundError:
        print("错误：文件未找到，请检查路径")
except Exception as e:
        print(f"发生错误：{str(e)}")

# 创建多项式回归模型
poly = PolynomialFeatures(degree=2)  # 使用二次多项式回归
model = LinearRegression()

# 拟合每一列并调整预测值
adjusted_preds = np.zeros_like(pred_vals)

for i in range(true_vals.shape[1]):
    true_col = true_vals[:, i]
    pred_col = pred_vals[:, i]
    
    # 将预测值转为多项式特征
    X_poly = poly.fit_transform(pred_col.reshape(-1, 1))
    
    # 拟合模型
    model.fit(X_poly, true_col)
    
    # 校正预测值
    adjusted_preds[:, i] = model.predict(X_poly)

# 计算校正后的 MAE, RMSE, MAPE, Correlation
metrics = []
# 处理两个数组
# true_vals = process_array(true_vals)
# adjusted_preds = process_array(adjusted_preds)
for i in range(true_vals.shape[1]):
    true_col = true_vals[:, i]
    pred_col = adjusted_preds[:, i]
    
    mae = mean_absolute_error(true_col, pred_col)
    rmse = root_mean_squared_error(true_col, pred_col)
    mape = np.mean(np.abs((true_col - pred_col) / np.clip(true_col, 1e-8, None))) * 100
    corr, _ = pearsonr(true_col, pred_col)
    
    metrics.append((mae, rmse, mape, corr))

# 输出校正后的预测值和误差指标
print("校正后的预测值：")
print(adjusted_preds)

print("\n每列的 MAE, RMSE, MAPE, Correlation：")
for i, (mae, rmse, mape, corr) in enumerate(metrics):
    print(f"列{i+1} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, Correlation: {corr:.4f}")
