# 使用 Scikit-learn Pipeline 建構你的機器學習流水線

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/pipeline.png)

在進行機器學習專案時，你一定經常需要透過 step by step 的方式，把原始資料轉換成機器學習可以使用的「訓練資料」。這些步驟可能包含了：將數值資料做正規化，以避免有某些過大的數值影響到預測結果、對類別型資料進行 one-hot-encoding，轉換成數值表示法、將某些包含缺失值欄位的資料進行補值或是捨棄、抑或是其他客製化的處理...等等。

不管要做什麼處理，我們都需要將資料透過一站站，像是「流水線」一般的處理步驟，把資料轉換成可以讓模型訓練的「訓練資料」。在這篇文章裡，讓我們來看看怎麼透過 `scikit-learn` 的 `pipeline` 來進行流水線處理。

透過 `scikit-learn` 的 `pipeline` 來處理資料有以下幾個好處：

1. 讓工作流程更為清楚明瞭
2. 確保工作流程的順序

要學習 `pipeline` 之前，我們需要了解在 `scikit-learn` 當中包含以下幾種類型的物件：

## estimator

estimator 是 `scikit-learn` 中基礎的物件，只要有實作 `fit` 方法類別，都可以稱之為一個 estimator。很多時候，你可以把 estimator 想像成分類器。通常 `estimator` 都會包含以下方法：

- `fit`：透過訓練資料來訓練分類器演算法中的參數。
- `predict`：將訓練好的分類器用在預測測試資料集上。

## transformer

具有 `fit` 和 `transform` 或是 `fit_transform` 方法的類別。這種類別的用途顧名思義就是要對於資料進行轉換，你看到許多在 `preprocessing` 中的類別，或是 `PCA` 等降維的方法都是屬於此類型。主要會包含以下幾個方法：

- `fit`：透過資料來訓練演算法的參數
- `transform`：實際進行資料轉換
- `fit_transform`：同時進行 `fit` 和 `transform` 兩個方法

## predictor

顧名思義就是要有 `fit` 和 `predict`，或是 `fit_predict` 等方法的類別。你看到許多 supervised 的分類器或是 regressor，甚至是一些 unsupervised 的方法都是屬於此類型。

上面這三種類別的物件，幾乎就是我們在使用 `scikit-learn` 會用到的方法，而 `pipeline` 就是由這這些類型所組成的流水線。舉例來說，一個典型的流水線如下：

```python
model = Pipeline([
    ('std', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', RandomForestClassifier())
])
```

我們來看看流水線的方法定義：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/sklearn-pipeline.png)

可以看到，一個流水線會接收一個 list of tuples，這個 tuple 其實就是數個 transformer 和一個 estimator 所組成，規定是流水線的最後一個方法一定要是一個 estimator，前面數個方法必須要是 transformer。從上述定義中可以得知，基本上 transformer 或 predictor 都是屬於一個 estimator。

當我們建構好一個 `pipeline` 的流水線後，我們可以這樣使用它：

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

很方便吧！而如果我們沒有一個好的流水線時，我們會怎麼做呢？

```python
std = StandardScaler()
pca = PCA(n_components=2)
clf = RandomForestClassifier()

x_train_trans = std.fit_transform(X_train)
x_train_pca = pca.fit_transform(x_train_trans)
clf.fit(x_train_pca, y_train)

x_test_trans = std.fit_transform(X_test)
x_test_pca = pca.fit_transform(x_test_trans)
y_pred = clf.predict(x_test_trans)
```

可以看到我們需要把同樣的處理步驟重複套用在 train 和 test 的資料上，整個處理過程會變得很冗長而且不清楚。

而透過 `pipeline` 就可以很方便的幫助我們搭建一個清楚明瞭且順序明確的機器學習流水線。

此外，你還可以透過 `make_pipeline` 方法，來幫助我們更快的建立流水線：

```python
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    RandomForestClassifier()
)
```

`make_pipeline` 的官方文件是這樣說的：

```
Construct a Pipeline from the given estimators.

This is a shorthand for the Pipeline constructor; it does not require, and does not permit, naming the estimators. Instead, their names will be set to the lowercase of their types automatically.
```

所以你在宣告每一個 edtimator 的時候，不需要也不允許給定它名稱，它會自動被命名為對應型別的小寫名稱。

甚至，你可以自訂屬於自己的 estimator 和 transformer，來打造自己的流水線，並透過 `FeatureUnion` 來串接不同的 `pipeline`，我們在下一篇會繼續探討這部分的內容。
