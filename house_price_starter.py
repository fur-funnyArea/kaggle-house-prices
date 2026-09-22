"""
Kaggle房价预测入门项目
数据集: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
步骤: 下载数据 -> 读数据 -> 简单处理 -> 训练模型 -> 预测 -> 提交
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============ 第1步: 读取数据 ============
# 先去Kaggle页面下载train.csv和test.csv，放到同一目录下
print("读取数据...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")
print(f"列名示例: {train.columns[:10].tolist()}")

# ============ 第2步: 看看数据长什么样 ============
print("\n前5行数据:")
print(train.head())
print("\n房价统计信息:")
print(train['SalePrice'].describe())

# ============ 第3步: 简单处理数据 ============
print("\n处理数据...")

# 挑几个重要的特征列（不用全部，先用简单的）
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

X = train[features]
y = train['SalePrice']
X_test = test[features]

# 缺失值填中位数
X = X.fillna(X.median())
X_test = X_test.fillna(X.median())

# ============ 第4步: 划分验证集 ============
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

# ============ 第5步: 训练随机森林模型 ============
print("\n训练模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在验证集上看效果
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"验证集RMSE: {rmse:.2f} （越小越好）")

# ============ 第6步: 预测测试集 ============
print("\n预测测试集...")
predictions = model.predict(X_test)

# ============ 第7步: 生成提交文件 ============
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})
submission.to_csv("submission.csv", index=False)
print("提交文件已生成: submission.csv")
print("把这个文件上传到Kaggle就完成参赛了！")

# ============ 进阶优化（可选） ============
# 1. 用所有列而不是只挑6个
# 2. 处理类别变量（用get_dummies或LabelEncoder）
# 3. 试试XGBoost或LightGBM
# 4. 做特征工程（如总面积=地下室+一楼+二楼）
# 5. 用交叉验证调参
