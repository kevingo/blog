## Classification

### Support Vector Machine (SVM)

* 簡介

SVM是一個二元分類模型，最主要的概念就是想要讓 training data 在特徵空間中，找到一個超平面（高維度中的平面）將這些資料分開來。

SVM 主要是 binary classifier，如果要解決多類別的問題是，可以用以下幾個方法：

1. One-against-One
2. One-against-All

* 參數

1. kernal
2. cost : The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example.

* 目標

找到一個超平面，這個超平面是擁有「最大邊距」的超平面。

* Reference
	- [理解 SVM 背後的數學原理（1）](http://blog.jobbole.com/102082/)
	- [支持向量機器 (Support Vector Machine)](https://cg2010studio.com/2012/05/20/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%A9%9F%E5%99%A8-support-vector-machine/)

### Decision Tree

### Logistic Regression

### Naive Bayes 

## Clustering

### K-Means

* 簡介

對已知的資料 (x1, x2, ... , xn)，劃分成 k 個集合

* 參數

1. k
2. maxIterations: 指定一個最多要跑的次數

* 目標

* References

### Reinforcement Learning

* 不容易告訴機器你想要的是什麼
* 常見應用：廣告系統