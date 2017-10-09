## Deep learning

### Hello World of deep learning
影片連結：[https://www.youtube.com/watch?v=Lx3l4lOrquw](https://www.youtube.com/watch?v=Lx3l4lOrquw)

### Batch and Epoch
- 在做 gradient descent 的時候，並不會真的去 minimize total loss
- 我們會把 training data 分成一個一個的 batch
	- 假設全部的 training data 有 1000 張圖片，我們希望一個 batch 有 100 張，那我們就有 10 個 batch。
	- 我們去計算第一個 batch 中每個 sample 的 loss 之後，來計算第一個 batch 的 total loss (L')
	- 根據 L'，使用 gradient descent 來更新整個 network 的參數(計算參數對 L' 的偏微分)
	- 重複這個流程，做完 10 個 batch 我們叫做一個 epoch
- 我們在做 DL，就是重複上面的流程，所以在訓練 model 時，會需要多個 epoch
- 在每個 batch 裡面都會 update 一次模型的參數，所以有 10 個 batch，做了 20 次 epoch，我們就會更新 10*20=200 次參數

![batch_and_epoch](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/batch_and_epoch.png)

#### 梯度下降法 stochastic gradient descent (SGD)
- SGD 演算法將一筆一筆的資料放入模型中推算參數，又稱為 online learning
- 如果 batch_size = 1，等於使用 stochastic gradient descent

#### mini-batch
- 要用 mini-batch 的原因
	- batch_size 越小，代表在一個 epoch 裡面更新參數的次數越多，速度越慢
	- 當 batch_size 不同時，一個 epoch 需要的時間是不一樣的
	- 以下圖為例，其實 batch_size=1 和 batch_size=10 更新參數的次數差不多 (batch_size=1，跑一個 epoch; 相同時間內，batch_size=10 的時候，已經跑了 10 個 epoch(5000*10)，同樣更新了 50000 次參數)
	- 如果不同的 batch_size 更新參數的次數差不多，那我們會選擇 batch_size 比較大的，因為在做 gradient descent 時會比較穩定(參考的sample 比較多，會比較容易往正確的方向前進)
- 為什麼 batch_size 較大時，速度會較快？
	- 因為使用了平行運算
	- 但是 batch_size 太大，GPU 也無法負荷

![mini_batch_speed](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/mini_batch_speed.png)

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