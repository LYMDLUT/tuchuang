import numpy as np
from scipy.stats import gaussian_kde
import torch

def generate_data(mean, cov, num_samples=1000):
    return np.random.multivariate_normal(mean, cov, num_samples)

def compute_kde(samples):
    return gaussian_kde(samples.T)

def kl_divergence(eval_points, mean, cov):
    # 计算KDE估计的对数概率密度
    kde = compute_kde(eval_points)
    log_p_eval = kde.logpdf(eval_points.T)
    # 计算高斯分布的对数概率密度
    log_q_eval = -0.5 * np.sum((eval_points - mean) ** 2 / cov.diagonal(), axis=1) - 0.5 * np.log(2 * np.pi * cov.diagonal()).sum()
    # 计算KL散度
    kl_div = np.mean(log_p_eval - log_q_eval)
    return kl_div

def main():
    dim = 3072
    # 生成两组数据
    mean = np.zeros(dim)
    covariance = np.eye(dim)  # 标准高斯分布
    
    data1 = torch.load('gsr003_merge.pt',map_location='cpu')[:20].reshape(20*1024,-1).detach().numpy()
    data2 = torch.load('linf_merge.pt',map_location='cpu')[:20].reshape(20*1024,-1).detach().numpy()
    # 计算KL散度
    kl1 = kl_divergence(data1, mean, covariance)
    kl2 = kl_divergence(data2, mean, covariance)

    print(f"KL Divergence for Group 1: {kl1}")
    print(f"KL Divergence for Group 2: {kl2}")

if __name__ == "__main__":
    main()
