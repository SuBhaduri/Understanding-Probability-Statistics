#!/usr/bin/env python
# coding: utf-8

# ### Understanding Probability & Statistics…

# In[1]:


#Import Common Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

# ### Bernoulli Distribution

# In[2]:


#Bernoulli Distribution
from scipy.stats import bernoulli
p = 0.7
x = np.arange(bernoulli.ppf(0.01, p), bernoulli.ppf(0.99, p)) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", bernoulli.stats(p, moments='m'))
print("Variance          : ", bernoulli.stats(p, moments='v'))
print("Prob. Mass Func.  : ", bernoulli.pmf(x, p).item())
print("Cum. Density Func.: ", bernoulli.cdf(x, p).item())

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, bernoulli.pmf(x, p), 'ro', ms=8, label='PMF=(1-p)')
plt.plot(1 - x, 1 - bernoulli.pmf(x, p), 'go', ms=8, label='PMF=p')
plt.vlines(x, 0, bernoulli.pmf(x, p), colors='r', lw=5, alpha=0.5)
plt.vlines(1 - x, 0, 1 - bernoulli.pmf(x, p), colors='g', lw=5, alpha=0.5)
plt.xlabel("Sample Space of Bernoulli Distribution", fontsize=14)  
plt.ylabel("PMF", fontsize=14)
plt.title("Probability Distribution of Bernoulli(p=0.7) Distribution", fontsize=16)
plt.xticks(np.arange(0, 2, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.hlines(0, -2, 0, colors='b', lw=5, alpha=0.5, linestyle='dashed')
plt.plot(0, 0, 'bo', ms=8)
plt.plot(x, bernoulli.cdf(x, p), 'ro', ms=8, label='CDF')
plt.hlines(0.3, 0, 1, colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.plot(1, 0.3, 'bo', ms=8)
plt.plot(1 - x, 1, 'ro', ms=8)
plt.hlines(1, 1, 2, colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.vlines(0, 0, 0.3, colors='b', lw=5, alpha=0.5)
plt.vlines(1, 0.3, 1, colors='b', lw=5, alpha=0.5)
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Bernoulli(p=0.7) Distribution", fontsize=16)
plt.xticks(np.arange(-2, 3, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Binomial Distribution

# In[3]:


#Binomial Distribution
from scipy.stats import binom
n, p = 10, 0.4
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p)) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", binom.stats(n, p, moments='m'))
print("Variance          : ", binom.stats(n, p, moments='v'))
print("Prob. Mass Func.  : ", binom.pmf(x, n, p))
print("Cum. Density Func.: ", binom.cdf(x, n, p))

CDF = binom.cdf(x, n, p)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, binom.pmf(x, n, p), 'go', ms=8, label='PMF')
plt.vlines(x, 0, binom.pmf(x, n, p), colors='g', lw=5, alpha=0.5)
plt.xlabel("Sample Space of Binomial Distribution", fontsize=14)  
plt.ylabel("PMF", fontsize=14)
plt.title("Probability Distribution of Binomial(n=10,p=0.4) Distribution", fontsize=16)
plt.xticks(np.arange(0, 8, 1))
plt.yticks(np.arange(0, 0.5, 0.1))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.hlines(0, -2, 1, colors='b', lw=5, alpha=0.5, linestyle='dashed')
plt.vlines(1, 0, CDF[0], colors='b', lw=5, alpha=0.5)
plt.plot(1, 0, 'bo', ms=8)
for i in range(len(CDF) - 1):
    plt.plot(i + 2, CDF[i], 'bo', ms=8)
    plt.vlines(i + 2, CDF[i], CDF[i + 1], colors='b', lw=5, alpha=0.5)
    plt.plot(i + 1, CDF[i], 'ro', ms=8)
    plt.hlines(CDF[i], i + 1, i + 2, colors='r', lw=5, alpha=0.5, linestyle='dashed')

plt.plot(len(CDF), CDF[len(CDF) - 1], 'ro', ms=8, label='CDF')
plt.hlines(CDF[len(CDF) - 1], len(CDF), len(CDF) + 1, colors='r', lw=5, alpha=0.5, linestyle='dashed')

plt.hlines(1, 7, 8, colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Binomial(n=10,p=0.4) Distribution", fontsize=16)
plt.xticks(np.arange(-2, 9, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Geometric Distribution

# In[4]:


#Geometric Distribution
from scipy.stats import geom
p = 0.6
x = np.arange(geom.ppf(0.01, p), geom.ppf(0.99, p)) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", geom.stats(p, moments='m'))
print("Variance          : ", geom.stats(p, moments='v'))
print("Prob. Mass Func.  : ", geom.pmf(x, p))
print("Cum. Density Func.: ", geom.cdf(x, p))

CDF = geom.cdf(x, p)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, geom.pmf(x, p), 'go', ms=8, label='PMF')
plt.vlines(x, 0, geom.pmf(x, p), colors='g', lw=5, alpha=0.5)
plt.xlabel("Sample Space of Geometric Distribution", fontsize=14)  
plt.ylabel("PMF", fontsize=14)
plt.title("Probability Distribution of Geometric(p=0.6) Distribution", fontsize=16)
plt.xticks(np.arange(0, 6, 1))
plt.yticks(np.arange(0, 0.8, 0.1))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.hlines(0, -2, 1, colors='b', lw=5, alpha=0.5, linestyle='dashed')
plt.vlines(1, 0, CDF[0], colors='b', lw=5, alpha=0.5)
plt.plot(1, 0, 'bo', ms=8)
for i in range(len(CDF) - 1):
    plt.plot(i + 2, CDF[i], 'bo', ms=8)
    plt.vlines(i + 2, CDF[i], CDF[i + 1], colors='b', lw=5, alpha=0.5)
    plt.plot(i + 1, CDF[i], 'ro', ms=8)
    plt.hlines(CDF[i], i + 1, i + 2, colors='r', lw=5, alpha=0.5, linestyle='dashed')
    
plt.plot(len(CDF), CDF[len(CDF) - 1], 'ro', ms=8, label='CDF')
plt.hlines(CDF[len(CDF) - 1], len(CDF), len(CDF) + 1, colors='r', lw=5, alpha=0.5, linestyle='dashed')

plt.hlines(1, 5, 6, colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Geometric(p=0.6) Distribution", fontsize=16)
plt.xticks(np.arange(-2, 7, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Poisson Distribution

# In[5]:


#Poisson Distribution
from scipy.stats import poisson

loc, mu = 0, 10 # Mu is basically Lambda
x = np.arange(poisson.ppf(0.01, mu, loc), poisson.ppf(0.99, mu, loc)) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", poisson.stats(mu, loc, moments='m'))
print("Variance          : ", poisson.stats(mu, loc, moments='v'))
print("Prob. Dens. Func. : ", poisson.pmf(x, mu, loc))
print("Cum. Density Func.: ", poisson.cdf(x, mu, loc))

CDF = poisson.cdf(x, mu, loc)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, poisson.pmf(x, mu, loc), 'go', ms=8, label='PMF')
plt.vlines(x, 0, poisson.pmf(x, mu, loc), colors='g', lw=5, alpha=0.5)
plt.xlabel("Sample Space of Poisson Distribution", fontsize=14)  
plt.ylabel("PMF", fontsize=14)
plt.title("Probability Distribution of Poisson(λ=10) Distribution", fontsize=16)
plt.xticks(np.arange(0, 20, 1))
plt.yticks(np.arange(0, 0.25, 0.05))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.hlines(0, -1, 3, colors='b', lw=5, alpha=0.5, linestyle='dashed')
plt.vlines(3, 0, CDF[0], colors='b', lw=5, alpha=0.5)
plt.plot(3, 0, 'bo', ms=8)
for i in range(len(CDF) - 1):
    plt.plot(i + 4, CDF[i], 'bo', ms=8)
    plt.vlines(i + 4, CDF[i], CDF[i + 1], colors='b', lw=5, alpha=0.5)
    plt.plot(i + 3, CDF[i], 'ro', ms=8)
    plt.hlines(CDF[i], i + 3, i + 4, colors='r', lw=5, alpha=0.5, linestyle='dashed')
    
plt.plot(len(CDF) + 2, CDF[len(CDF) - 1], 'ro', ms=8, label='CDF')
plt.hlines(CDF[len(CDF) - 1], len(CDF) + 2, len(CDF) + 3, colors='r', lw=5, alpha=0.5, linestyle='dashed')
    
plt.hlines(1, 17, 18, colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Poisson(λ=10) Distribution", fontsize=16)
plt.xticks(np.arange(-1, 20, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Uniform (Discrete) Distribution

# In[6]:


#Uniform (Discrete) Distribution
from scipy.stats import randint

low, high = 1, 10
x = np.arange(randint.ppf(0.01, low, high), randint.ppf(0.99, low, high)) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", randint.stats(low, high, moments='m'))
print("Variance          : ", randint.stats(low, high, moments='v'))
print("Prob. Mass Func.  : ", randint.pmf(x, low, high))
print("Cum. Density Func.: ", randint.cdf(x, low, high))

CDF = randint.cdf(x, low, high)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, randint.pmf(x, low, high), 'go', ms=8, label='PMF')
plt.vlines(x, 0, randint.pmf(x, low, high), colors='g', lw=5, alpha=0.5)
plt.xlabel("Sample Space of Discrete Uniform Distribution", fontsize=14)  
plt.ylabel("PMF", fontsize=14)
plt.title("Probability Distribution of Discrete Uniform(a=1,b=10) Distribution", fontsize=16)
plt.xticks(np.arange(0, 10, 1))
plt.yticks(np.arange(0, 0.3, 0.05))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.hlines(0, -1, 1, colors='b', lw=5, alpha=0.5, linestyle='dashed')
plt.vlines(1, 0, CDF[0], colors='b', lw=5, alpha=0.5)
plt.plot(1, 0, 'bo', ms=8)
for i in range(len(CDF) - 1):
    plt.plot(i + 2, CDF[i], 'bo', ms=8)
    plt.vlines(i + 2, CDF[i], CDF[i + 1], colors='b', lw=5, alpha=0.5)
    plt.plot(i + 1, CDF[i], 'ro', ms=8)
    plt.hlines(CDF[i], i + 1, i + 2, colors='r', lw=5, alpha=0.5, linestyle='dashed')
    
plt.plot(len(CDF), CDF[len(CDF) - 1], 'ro', ms=8, label='CDF')
plt.hlines(CDF[len(CDF) - 1], len(CDF), len(CDF) + 1, colors='r', lw=5, alpha=0.5, linestyle='dashed')

plt.plot(9, CDF[-1], 'bo', ms=8)
plt.vlines(9, CDF[-1], 1, colors='b', lw=5, alpha=0.5)

plt.hlines(1, 9, 10, colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Discrete Uniform(a=1,b=10) Distribution", fontsize=16)
plt.xticks(np.arange(-1, 12, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Exponential Distribution

# In[7]:


#Exponential Distribution
from scipy.stats import expon

loc, scale = 0, 0.67 # Scale is 1/Lambda
x = np.linspace(expon.ppf(0.01, loc, scale), expon.ppf(0.99, loc, scale), 25) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", expon.stats(loc, scale, moments='m'))
print("Variance          : ", expon.stats(loc, scale, moments='v'))
print("Prob. Dens. Func. : ", expon.pdf(x, loc, scale))
print("Cum. Density Func.: ", expon.cdf(x, loc, scale))

CDF = expon.cdf(x, loc, scale)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, expon.pdf(x, loc, scale), 'g', ms=8, label='PDF')
plt.xlabel("Sample Space of Exponential Distribution", fontsize=14)  
plt.ylabel("PDF", fontsize=14)
plt.title("Probability Distribution of Exponential(λ=1.5) Distribution", fontsize=16)
plt.xticks(np.arange(0, 5, 1))
plt.yticks(np.arange(0, 1.7, 0.1))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.plot(x, expon.cdf(x, loc, scale), 'r', ms=8, label='CDF')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Exponential(λ=1.5) Distribution", fontsize=16)
plt.xticks(np.arange(-1, 5, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Uniform (Continuous) Distribution

# In[8]:


#Uniform (Continuous) Distribution
from scipy.stats import uniform

loc, scale = 1, 10
x = np.linspace(uniform.ppf(0.01, loc, scale), uniform.ppf(0.99, loc, scale), 100) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", uniform.stats(loc, scale, moments='m'))
print("Variance          : ", uniform.stats(loc, scale, moments='v'))
print("Prob. Dens. Func. : ", uniform.pdf(x, loc, scale))
print("Cum. Density Func.: ", uniform.cdf(x, loc, scale))

CDF = randint.cdf(x, loc, scale)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, uniform.pdf(x, loc, scale), 'g', ms=8, label='PDF')
plt.vlines(loc, 0, 0.1, colors='g', lw=5, alpha=0.5, linestyle='dashed')
plt.vlines(scale + 1, 0, 0.1, colors='g', lw=5, alpha=0.5, linestyle='dashed')
plt.xlabel("Sample Space of Continuous Uniform Distribution", fontsize=14)  
plt.ylabel("PDF", fontsize=14)
plt.title("Probability Distribution of Continuous Uniform(a=1,b=10) Distribution", fontsize=16)
plt.xticks(np.arange(0, 12, 1))
plt.yticks(np.arange(0, 0.3, 0.05))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.hlines(0, -1, 1, colors='b', lw=5, alpha=0.5, linestyle='dashed')
plt.plot(loc, 0, 'bo', ms=8)
plt.plot(scale + 1, CDF[-1], 'bo', ms=8)
x_lab = [loc, scale + 1]
y_lab = [0, CDF[-1]]
plt.plot(x_lab, y_lab, color='red', label='CDF')
plt.hlines(1, 11, 12, colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.vlines(11, 0, CDF[-1], colors='r', lw=5, alpha=0.5, linestyle='dashed')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Continuous Uniform(a=1,b=10) Distribution", fontsize=16)
plt.xticks(np.arange(-1, 14, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Gaussian (Normal) Distribution

# In[9]:


#Gaussian (Normal) Distribution
from scipy.stats import norm

loc, scale = 1, 2 # Mean and Variance
x = np.linspace(norm.ppf(0.01, loc, scale), norm.ppf(0.99, loc, scale), 25) #Percent Point Function (inverse of cdf — percentiles)

print("Mean              : ", norm.stats(loc, scale, moments='m'))
print("Variance          : ", norm.stats(loc, scale, moments='v'))
print("Prob. Dens. Func. : ", norm.pdf(x, loc, scale))
print("Cum. Density Func.: ", norm.cdf(x, loc, scale))

CDF = norm.cdf(x, loc, scale)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(x, norm.pdf(x, loc, scale), 'g', ms=8, label='PDF')
plt.xlabel("Sample Space of Gaussian Distribution", fontsize=14)  
plt.ylabel("PDF", fontsize=14)
plt.title("Probability Distribution of Gaussian(µ=1,σ=2) Distribution", fontsize=16)
plt.xticks(np.arange(-5, 7, 1))
plt.yticks(np.arange(0, 0.30, 0.05))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.plot(x, norm.cdf(x, loc, scale), 'r', ms=8, label='CDF')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Gaussian(µ=1,σ=2) Distribution", fontsize=16)
plt.xticks(np.arange(-5, 7, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Covariance and Correlation

# In[10]:


#Covariance and Correlation
from scipy.stats import pearsonr

np.random.seed(1)
data1 = 20 * np.random.randn(20000) + 100
data2 = data1 + 10 * np.random.randn(20000) - 50

print("data1: Mean=%.3f; Variance=%.3f" % (np.mean(data1), np.var(data1)))
print("data2: Mean=%.3f; Variance=%.3f" % (np.mean(data2), np.var(data2)))
print("Covariance between data1 and data2: \n", np.cov(data1, data2))

corr, p = pearsonr(data1, data2)
print("Pearson Correlation Coefficient is %.3f with p-value %.3f" % (corr, p))

plt.scatter(data1, data2)
plt.title("Correlation", fontsize=16)
plt.xlabel("data1", fontsize=14)  
plt.ylabel("data2", fontsize=14)

plt.show()


# ### Law of Large Numbers

# In[11]:


#Generate Samples from Standard Normal Distribution and Multiply Them by 5 and Add 10;
#So the Resulting Distribution of 'data' is Also Gaussian

np.random.seed(1)
data = 5 * np.random.randn(10) + 10
print("Average when n=10 is    :", np.mean(data))

data = 5 * np.random.randn(1000) + 10
print("Average when n=1000 is  :", np.mean(data))

data = 5 * np.random.randn(10000) + 10
print("Average when n=10000 is :", np.mean(data))

data = 5 * np.random.randn(100000) + 10
print("Average when n=100000 is:", np.mean(data))


# ### Central Limit Theorem

# In[12]:


#Generate a Sample of Dice Rolls
np.random.seed(1)
T_n = []
mu = (1 + 2 + 3 + 4 + 5 + 6) / 6
n = 1000
for i in range(n):
    data = np.random.randint(low=1, high=7, size=1000) #Returns discrete uniform integers
    X_n  = np.sum(data)/1000 #Sample average
    X_c  = X_n - mu #Subtract true mean mu
    X_f  = X_c * np.sqrt(n) #Multiply by sqrt(n)
    T_n.append(X_f)

fig = plt.figure(figsize=(20,10))
plt.hist(T_n, bins=100)
plt.title("Histogram of Sample Means from Dice Roll Simulations", fontsize=16)
plt.xticks(np.arange(-5, 6, 1))

plt.show()


# ### Confidence Interval

# In[13]:


#Confidence Interval
from scipy.stats import norm

np.random.seed(1)
data = 5 * np.random.randn(100) + 50
q_a_2 = norm.ppf(0.90)

low  = np.mean(data) - (q_a_2 * np.std(data)) / np.sqrt(len(data))
high = np.mean(data) + (q_a_2 * np.std(data)) / np.sqrt(len(data))

print("90 percent Confidence Interval is: %.3f, %.3f" % (low, high))


# ### Student's t-Distribution

# In[14]:


#Student's t-Distribution
from scipy.stats import t

sample_space = np.arange(-5, 5, 0.001)
dof = len(sample_space) - 1 #Number of independent variables
pdf = t.pdf(sample_space, dof)
cdf = t.cdf(sample_space, dof)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(sample_space, pdf, 'g', ms=8, label='PDF')
plt.xlabel("Sample Space of t-Distribution", fontsize=14)  
plt.ylabel("PDF", fontsize=14)
plt.title("Probability Distribution of t-Distribution", fontsize=16)
plt.xticks(np.arange(-5, 7, 1))
plt.yticks(np.arange(0, 0.50, 0.1))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.plot(sample_space, cdf, 'r', ms=8, label='CDF')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of t-Distribution", fontsize=16)
plt.xticks(np.arange(-5, 7, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Chi-Squared Distribution

# In[15]:


#Chi-Squared Distribution
from scipy.stats import chi2

sample_space = np.arange(0, 50, 0.01)
dof = 20 #Number of independent variables
pdf = chi2.pdf(sample_space, dof)
cdf = chi2.cdf(sample_space, dof)

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(sample_space, pdf, 'g', ms=8, label='PDF')
plt.xlabel("Sample Space of Chi Square Distribution", fontsize=14)  
plt.ylabel("PDF", fontsize=14)
plt.title("Probability Distribution of Chi Square Distribution", fontsize=16)
plt.xticks(np.arange(0, 60, 10))
plt.yticks(np.arange(0, 0.07, 0.01))
plt.legend(loc='best', shadow=True)

plt.subplot(222)
plt.plot(sample_space, cdf, 'r', ms=8, label='CDF')
plt.xlabel("Observation", fontsize=14)  
plt.ylabel("CDF", fontsize=14)
plt.title("Cumulative Density Function of Chi Square Distribution", fontsize=16)
plt.xticks(np.arange(0, 60, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='upper left', shadow=True)

plt.show()


# ### Student's t-Test

# In[16]:


#Student's t-Test
from scipy.stats import ttest_ind

np.random.seed(1)
#np.random.randn generates Standard Normal data
data1 = 20 * np.random.randn(200) + 50 # mean=50, standard-deviation=20
data2 = 10 * np.random.randn(200) + 51 # mean=51, standard-deviation=10

stat, p = ttest_ind(data1, data2)
print('Test Statistic=%.3f, p=%.3f' % (stat, p))

alpha = 0.05 #Our desired confidence interval is 0.95

if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')


# ### Paired Student's t-Test

# In[17]:


#Paired Student's t-Test
from scipy.stats import ttest_rel

np.random.seed(1)
#np.random.randn generates Standard Normal data
data1 = 20 * np.random.randn(200) + 50 # mean=50, standard-deviation=20
data2 = 10 * np.random.randn(200) + 51 # mean=51, standard-deviation=10

stat, p = ttest_rel(data1, data2)
print('Test Statistic=%.3f, p=%.3f' % (stat, p))

alpha = 0.05 #Our desired confidence interval is 0.95

if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')


# ### Analysis of Variance (ANOVA)

# In[18]:


#ANOVA
from scipy.stats import f_oneway

np.random.seed(1)

data1 = np.array([6, 8, 4, 5, 3, 4])
data2 = np.array([8, 12, 9, 11, 6, 8])
data3 = np.array([13, 9, 11, 8, 7, 12])

stat, p = f_oneway(data1, data2, data3)
print('Test Statistic=%.3f, p=%.3f' % (stat, p))

alpha = 0.05 #Our desired confidence interval is 0.95

if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
    
# Native Method
print("==> Native Method To Show the Calculation Method <==")
# Mean within each group
mean_1 = np.mean(data1)
mean_2 = np.mean(data2)
mean_3 = np.mean(data3)
n = len(data1)

# Overall mean
mean_o = (mean_1 + mean_2 + mean_3) / 3

# Calculate the "between-group" sum of squared differences
S_B = n * (mean_1 - mean_o) ** 2 + n * (mean_2 - mean_o) ** 2 + n * (mean_3 - mean_o) ** 2
dof_B = 3 - 1

# Between-group mean square value
MS_B = S_B / dof_B

# "within-group" sum of squares
# Centering the data in each group
data1_C = data1 - mean_1
data2_C = data2 - mean_2
data3_C = data3 - mean_3

# Within-group sum of squares
S_W = (np.sum(data1_C ** 2) + np.sum(data2_C ** 2) + np.sum(data3_C ** 2)).item()

# Within-group degrees of freedom
dof_W = 3 * (n - 1)

# Within-group mean square value
MS_W = S_W / dof_W

# F-Ratio
F = MS_B / MS_W

print("F Statistic:", F)


# ### Chi-Squared Test

# In[19]:


#Chi-Squared Test

#We will use this table from Wikipedia
#-------------------------------------
#           	A	B	C	D	Total
#White collar	90	60	104	95	349
#Blue collar	30	50	51	20	151
#No collar  	30	40	45	35	150
#Total      	150	150	200	150	650

from scipy.stats import chi2, chi2_contingency
np.random.seed(1)
# contingency table
observed = np.array([[90, 60, 104, 95],[30, 50, 51, 20], [30, 40, 45, 35]], dtype=np.float64)
print("Observed Frequencies: \n", observed)
stat, p, dof, expected = chi2_contingency(observed)

print('dof=%d' % dof)
print("Expected Frequencies: \n", expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
    
# Native Method
print("==> Native Method <==")
for i in range(observed.shape[0]):
    for j in range(observed.shape[1]):
        expected[i, j] = (np.sum(observed, axis=1))[i] * (np.sum(observed, axis=0))[j] / np.sum(observed)

print("Observed: \n", observed)
print("Expected: \n", expected)

stat = ((observed - expected) ** 2) / expected
        
dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        
print("Statistic, dof: ", (np.sum(stat)).item(), ",", dof)


# ### KDE Plot

# In[20]:


#KDE Plot
np.random.seed(1)
data = np.random.random(5000) + 50

fig = plt.figure(figsize=(20,10))
sns.kdeplot(data, shade=True, color="r", legend=True)
plt.title("Kernel Density Plot of the Data", fontsize=16)
plt.xticks(np.arange(49.5, 51.5, 0.2))
plt.yticks(np.arange(0, 1.2, 0.1))

plt.show()


# ### Q-Q Plot

# In[21]:


#Q-Q Plot
from statsmodels.graphics.gofplots import qqplot
np.random.seed(1)
data = np.random.random(5000) + 50

qqplot(data, line='s')

plt.show()


# #### Shapiro-Wilk Test|D'Agostino's K2 Test|Anderson-Darling Test

# In[22]:


#Shapiro-Wilk Test
from scipy.stats import shapiro, normaltest, anderson
np.random.seed(1)
data = np.random.random(5000) + 50
alpha = 0.05

print("\n ==> Shapiro-Wilk Test <==")
stat, p = shapiro(data)
print('Statistic=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#D'Agostino's K2 Test
print("\n ==> D'Agostino's K2 Test <==")
stat, p = normaltest(data)
print('Statistic=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
    
#Anderson-Darling Test
print("\n ==> Anderson-Darling Test <==")
result = anderson(data, dist='norm')

print('Statistic, Critical Values: %.3f' % result.statistic, result.critical_values)
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

