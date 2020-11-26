# 打造自己的 pipeline

在 [使用 Scikit-learn Pipeline 建構你的機器學習流水線](https://github.com/kevingo/blog/blob/master/ML/scikit-learn-pipeline.md) 文章中，我們學習到如何透過 `scikit-learn` 的 `pipeline` 來打造機器學習的流水線，讓我們可以透過 `pipeline` 一步步的將資料進行轉換及預測。

`scikit-learn` 固然內建了許多好用的 `transformer`，但畢竟資料千百萬種，許多時候我們必須要自己撰寫符合自己使用的 `transformer`，本篇文章會會告訴你如何實作這個部分。

## 撰寫自己的 transfomer

實作一個自己的 transfomer 可以透過以下幾個步驟來完成。

1. 實作繼承 `BaseEstimator` 和 `TransformerMixin` 的 class。

一個 transformer 通常會有 `fit`、`transform` 和 `fit_transform` 等方法。