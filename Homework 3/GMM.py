import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# 定义男性和女性的参数
mu_M, sigma_M = 176, 8
mu_F, sigma_F = 164, 6

# 定义男女比例
male_prob = 0.6
female_prob = 1 - male_prob

# 生成一组数据点
x = np.linspace(120, 220, 1000)

# 计算男性和女性的概率密度函数
male_pdf = norm.pdf(x, mu_M, sigma_M)
female_pdf = norm.pdf(x, mu_F, sigma_F)

# 计算总体概率密度函数
total_pdf = male_prob * male_pdf + female_prob * female_pdf

# 绘制曲线
plt.plot(x, total_pdf, label='Total PDF')
plt.plot(x, male_prob * male_pdf, label='Male PDF')
plt.plot(x, female_prob * female_pdf, label='Female PDF')
plt.xlabel('Height')
plt.ylabel('Probability Density')
plt.title('Probability Density Function')

# 生成男性和女性身高数据
num_male = int(1000 * male_prob)
num_female = int(1000 * female_prob)

male_heights = np.random.normal(mu_M, sigma_M, num_male)
female_heights = np.random.normal(mu_F, sigma_F, num_female)

# 将男女身高数据合并
all_heights = np.concatenate([male_heights, female_heights])
np.random.shuffle(all_heights)
data = pd.DataFrame({'Height': all_heights})

# 将数据写入 Excel 文件
data.to_csv('height_data.csv', index=False)

# 绘制身高数据的直方图
plt.hist(data['Height'], bins=30, alpha=0.5, label='Generated Data', density=True)
plt.legend()
plt.grid(True)
plt.savefig('height_distribution.png')
plt.show()
