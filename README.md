---
title: streamlit快速搭建芝加哥雨型应用
author: 陈文茂
date: 2024-06-16 21:25
updated: 2024-06-16 21:25
tags:
  - streamlit
  - python
  - 水文分析
categories: 经验分享
---
<meta name="referrer" content="no-referrer" />

## 目的
通过streamlit快速搭建芝加哥雨型应用，该应用实现以下功能：
- 内置重庆各区县的暴雨强度计算公式，
- 支持自定义城市的暴雨公式参数，
- 支持雨型生成，输出图片和文字
- 提供降雨历时、汇水面积、径流系数输入，支持径流量计算
- 在线部署，网页使用。
## 原型
用 obsidian excalidraw 简单绘制原型。

![image.png](https://gitee.com/chenwenmao/picbed/raw/master/img/202406162135564.png)

## 主要原理

- 《关于发布重庆市暴雨强度修订公式及设计暴雨雨型的通知》（渝建〔2017〕443 号）
-  [芝加哥合成暴雨过程线的公式推导 (vivifree.com)](https://www.vivifree.com/rain-model-chicago-formula.html)
## 主要功能实现
### 雨强计算
```python
def intensity(A, B, C, N, t, P):
    """
    雨强计算。
    参数:
    - a,b,c,n: 参数。
    - p (float): 设计重现期（单位：年）。
    - t (np.ndarray): 分钟数组
    返回:
    - its: 雨强（单位：mm/min）。
    """
    a = A * 0.4 * (1 + C * math.log10(P))
    its = a * ((1 - N) * t + B) / np.power(t + B, N + 1)
    return its
    
q = (A * (1 + C * math.log10(P))) / ((duration_minutes + B) ** N)
```
### 雨强分布计算
```python
def rainCalc_single_period(A, B, C, N, T: int, p: float, peak_ratio: float):
    """
    计算单一时段内的降雨强度分布。
    参数:
    - T (int): 降雨持续时间（单位：分钟）。
    - p (float): 设计重现期（单位：年）。
    - peak_ratio (float): 雨强峰值所在时间占总降雨历时的比例。
    返回:
    - np.ndarray: 随时间变化的降雨强度数组（单位：mm/min）。
    内部参数:
    - t (np.ndarray): 分钟数组
    - peak_time (float): 峰值时间
    """
    # ...函数实现代码...
    t = np.arange(0, T)
    peak_time = T * peak_ratio
    itAr = np.zeros(len(t))
    # 计算雨强
    for i in range(len(t)):
        if t[i] < peak_time:
            itAr[i] = intensity(A, B, C, N, (peak_time - t[i]) / peak_ratio, p) / 60
        else:
            itAr[i] = intensity(A, B, C, N, (t[i] - peak_time) / peak_ratio, p) / 60
    return itAr
```

其余功能实现均较为简单，项目所有代码均开源在 GitHub，有兴趣的可以去参观，点个 star 最好了，仓库如下：
[maoyu92/Chicago_rain_pattern (github.com)](https://github.com/maoyu92/Chicago_rain_pattern)
如果发现代码有不对的地方，也留言提醒我一下。
## 应用部署
Streamlit 是非常好用的一个应用框架，支持免费部署一个应用，非常方便。
### 部署 GitHub
应用部署在 GitHub。通过 git 将本地仓库部署在 GitHub 上，具体方法可参考：
[在vscode中使用git-新手向_新手vscode git-CSDN博客](https://blog.csdn.net/weixin_42984235/article/details/136906942)
这里跳过。
### Streamlit 应用发布
登录 Streamlit，登录后点击右上角 create app。
选择 I have an app。
![image.png](https://gitee.com/chenwenmao/picbed/raw/master/img/202406162151546.png)

在连接上自己的 GitHub 后，依次选择仓库地址，分支，主文件，定义 url 名称，点击 depoly。
![image.png](https://gitee.com/chenwenmao/picbed/raw/master/img/202406162152131.png)

稍等片刻，如果一切顺利就成功了，就非常丝滑。

### 注意事项
1. 仓库文件需要包括 requirement.txt，为项目运行需要的环境。
2. 注意代码引用文件需要用相对应用的方式，不要绝对引用。

### 成果预览

网址：[Streamlit (chicagorainpattern-meuiz8zrvbesh32nfm9iv7.streamlit.app)](https://chicagorainpattern-meuiz8zrvbesh32nfm9iv7.streamlit.app/)

![image.png](https://gitee.com/chenwenmao/picbed/raw/master/img/202406162156332.png)








