import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def expectation_step(data, n, theta, miu, sigma):
    data_len = len(data)
    gamma = np.zeros((data_len, n))
    for i in range(data_len):
        for j in range(n):
            gamma[i][j] = theta[j] * gaussian(data[i], miu[j], sigma[j])
        gamma[i] /= sum(gamma[i])
    return gamma

def gaussian(x, miu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - miu) ** 2 / (2 * sigma ** 2))

def maximization_step(data, gamma):
    n = gamma.shape[1]
    miu_new = [np.dot(data, gamma[:, j]) / np.sum(gamma[:, j]) for j in range(n)]
    sigma_new = [np.sqrt(np.dot((np.array(data) - miu_new[j]) ** 2, gamma[:, j]) / np.sum(gamma[:, j])) for j in range(n)]
    theta_new = [np.mean(gamma[:, j]) for j in range(n)]
    return theta_new, miu_new, sigma_new

def gmm_em(data, n, theta, miu, sigma, iter_max=10000):
    eps = 1e-10
    iter = 0
    while iter < iter_max:
        iter += 1
        gamma = expectation_step(data, n, theta, miu, sigma)
        theta_new, miu_new, sigma_new = maximization_step(data, gamma)

        if np.max(np.array(miu_new) - np.array(miu)) < eps and np.max(np.array(sigma_new) - np.array(sigma)) < eps \
                and np.max(np.array(theta_new) - np.array(theta)) < eps:
            miu = miu_new
            sigma = sigma_new
            theta = theta_new
            break
        else:
            miu = miu_new
            sigma = sigma_new
            theta = theta_new

    return theta, miu, sigma

data_read = pd.read_csv('height_data.csv')
data = data_read.values.flatten().tolist()

theta1, miu1, sigma1 = gmm_em(data, 2, theta=[0.5, 0.5], miu=[170, 170], sigma=[1, 1])
print("男生比例为{:.4f}，相对偏差为{:.2%}，男生身高均值为{:.4f}，相对偏差为{:.2%}，男生身高方差为{:.4f}，相对偏差为{:.2%}".format(
    theta1[0], (theta1[0] - 0.6) / 0.6, miu1[0], (miu1[0] - 176) / 176, sigma1[0], (sigma1[0] - 8) / 8))
print("女生比例为{:.4f}，相对偏差为{:.2%}，女生身高均值为{:.4f}，相对偏差为{:.2%}，女生身高方差为{:.4f}，相对偏差为{:.2%}".format(
    theta1[1], (theta1[1] - 0.4) / 0.4, miu1[1], (miu1[1] - 164) / 164, sigma1[1], (sigma1[1] - 6) / 6))

theta2, miu2, sigma2 = gmm_em(data, 2, theta=[0.5, 0.5], miu=[170, 160], sigma=[1, 1])
print("男生比例为{:.4f}，相对偏差为{:.2%}，男生身高均值为{:.4f}，相对偏差为{:.2%}，男生身高方差为{:.4f}，相对偏差为{:.2%}".format(
    theta2[0], (theta2[0] - 0.6) / 0.6, miu2[0], (miu2[0] - 176) / 176, sigma2[0], (sigma2[0] - 8) / 8))
print("女生比例为{:.4f}，相对偏差为{:.2%}，女生身高均值为{:.4f}，相对偏差为{:.2%}，女生身高方差为{:.4f}，相对偏差为{:.2%}".format(
    theta2[1], (theta2[1] - 0.4) / 0.4, miu2[1], (miu2[1] - 164) / 164, sigma2[1], (sigma2[1] - 6) / 6))

# 定义原始参数
mu_M, sigma_M = 176, 8
mu_F, sigma_F = 164, 6
male_prob = 0.6
female_prob = 1 - male_prob

# 生成一组数据点
x = np.linspace(120, 220, 1000)

# 计算概率密度函数
male_pdf = norm.pdf(x, mu_M, sigma_M)
female_pdf = norm.pdf(x, mu_F, sigma_F)
total_pdf = male_prob * male_pdf + female_prob * female_pdf
answer1_pdf = theta1[0] * norm.pdf(x, miu1[0], sigma1[0]) + theta1[1] * norm.pdf(x, miu1[1], sigma1[1])
answer2_pdf = theta2[0] * norm.pdf(x, miu2[0], sigma2[0]) + theta2[1] * norm.pdf(x, miu2[1], sigma2[1])

# 生成男性和女性身高数据
num_male = int(1000 * male_prob)
num_female = int(1000 * female_prob)
male_heights = np.random.normal(mu_M, sigma_M, num_male)
female_heights = np.random.normal(mu_F, sigma_F, num_female)

# 将男女身高数据合并
all_heights = np.concatenate([male_heights, female_heights])
np.random.shuffle(all_heights)
data = pd.DataFrame({'Height': all_heights})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax1.plot(x, total_pdf, label='Total PDF')
ax1.plot(x, answer1_pdf, label='Answer 1 PDF')
ax1.hist(data['Height'], bins=30, alpha=0.5, label='Generated Data', density=True)
ax1.set_xlabel('Height')
ax1.set_ylabel('Probability Density')
ax1.set_title('Probability Density Function - Answer 1')
ax1.legend()
ax1.grid(True)

ax2.plot(x, total_pdf, label='Total PDF')
ax2.plot(x, answer2_pdf, label='Answer 2 PDF')
ax2.hist(data['Height'], bins=30, alpha=0.5, label='Generated Data', density=True)
ax2.set_xlabel('Height')
ax2.set_ylabel('Probability Density')
ax2.set_title('Probability Density Function - Answer 2')
ax2.legend()
ax2.grid(True)

# 调整布局
plt.tight_layout()
plt.savefig('EM_result.png')
# 显示图形
plt.show()
