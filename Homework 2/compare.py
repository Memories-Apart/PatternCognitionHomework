# model_comparison.py

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from generate_data import make_moons_3d

# 生成数据
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)  # 生成1000个训练数据点
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)    # 生成500个测试数据点

# 定义模型列表
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM Linear Kernel': SVC(kernel='linear'),
    'SVM RBF Kernel': SVC(kernel='rbf'),
    'SVM Polynomial Kernel': SVC(kernel='poly', degree=3),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 准备绘图
fig = plt.figure(figsize=(15, 8))

# 评估每个模型
results = []
for index, (name, model) in enumerate(models.items(), 1):
    start_time = time.time()
    # 对非线性模型进行特征标准化
    if 'SVM' in name or 'XGBoost' in name:
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time
    
    # 记录分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    result = {
        'Model': name,
        'Accuracy': report['accuracy'],
        'F1 Score': report['weighted avg']['f1-score'],
        'Time Taken': elapsed_time
    }
    results.append(result)
    
    # 绘制分类结果
    ax = fig.add_subplot(2, 3, index, projection='3d')
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', marker='o')
    ax.set_title(f"{name} (Accuracy: {report['accuracy']:.2f})")

# 保存性能数据到CSV
df_results = pd.DataFrame(results)
df_results.to_csv('model_performance.csv', index=False)

# 显示图表
plt.tight_layout()
plt.show()
