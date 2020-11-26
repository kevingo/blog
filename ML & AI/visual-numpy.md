本文翻譯自 [Jay Alammar](https://twitter.com/jalammar) 的 [A Visual Intro to NumPy and Data Representation
](https://jalammar.github.io/visual-numpy/) 這篇部落格文章。對於從事資料分析或機器學習的朋友來說，numpy 一定是不陌生的 Python 套件。不管是資料處理所使用的 Pandas 、機器學習用到的 scikit-learn 或是 deep learning 所使用的 tensorflow 或 pytorch，底層在資料的操作或儲存上，大多會用到 numpy 來做科學的操作。而本篇文章以圖文並茂的方式詳細說明了在 numpy 中必學的操作，並且告訴你 numpy 的資料結構如何用來儲存文字、圖片、聲音等重要的資料，如果你有心學習資料科學或機器學習，一定要把 numpy 學好。希望透過此文的分享，讓大家在學習 numpy 的過程中能夠更加清楚其操作與用途。

---

# 圖解 numpy 與資料表示法

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array.png)

[NumPy](https://www.numpy.org/) 套件是 python 生態系中針對資料分析、機器學習和科學計算的重要角色。它大量簡化了向量和矩陣的操作運算，某些 python 的主要套件大量依賴 numpy 作為其架構的基礎 (例如：scikit-learn、SciPy、Pandas 和 tensorflow)。除了可以針對資料進行切片 (slice) 和 切塊 (dice) 之外，熟悉 numpy 還可以對使用上述套件帶來極大的好處。

在本文中，我們會學習 numpy 主要的使用方式，並且看到它如何用來表示不同類型的資料 (表格、影像、文字 ... 等) 作為機器學習模型的輸入。

```python
import numpy as np
```

## 建立陣列

我們可以透過 `np.array()` 並傳入一個 python list 來建立一個 numpy 的陣列 (又叫 [ndarray]((https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)))，如下所示：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/create-numpy-array-1.png)

通常我們希望 numpy 能夠初始化陣列的值，因此它提供了像 ones()、zeros() 和 random.random() 等方法。我們只要傳入希望產生的值即可：

![image](https://github.com/kevingo/blog/blob/master/screenshot/create-numpy-array-ones-zeros-random.png)

一但我們建立了陣列後，就可以透過有趣的方式來操作它們。

## 矩陣運算

讓我們建立兩個 numpy 陣列來展示如何進行運算，分別是 `data` 和 `ones`：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-arrays-example-1.png)

將這兩個 numpy 陣列依照位置相加 (即每一行相加)，只要使用 `data + ones` 即可：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-arrays-adding-1.png)

當我開始學習 numpy 後，我發現這樣抽象的思考讓我不需要使用類似迴圈的方式來進行計算，如此一來，我可以透過更高層次的角度來思考問題。

而除了相加之外，我們還可以進行以下操作：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-subtract-multiply-divide.png)

通常的情況下，我們會希望一個陣列可以和單一數字進行運算 (即向量和純量之間進行運算)。比如說，陣列中的數字是以英里為單位的距離，而我們希望將其轉換為公里，只需要透過 `data * 1.6` 即可：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-broadcast.png)

看到 numpy 是如何處理這樣的操作了嗎？這個概念稱為 *廣播 (broadcasting)*，它非常有用。

## 索引

我們可以像 python 的 list 進行切片一樣，對 numpy 的陣列進行索引和切片：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-slice.png)

## 聚合 (aggregation)

Numpy 另外一個好處是提供了聚合函式：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-aggregation.png)

除了 `min`、`max` 和 `sum` 之外，你還可以使用像是 `mean` 來得到平均值，`prod` 來得到所有元素的乘積，`std` 來得到標準差，以及[其他更多的功能](https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html)。

## 更多維度

上述我們所看到的範例都是在單一維度的向量上進行，而 numpy 之美在於這些操作可以擴展到任意維度的資料上。

### 建立矩陣

我們可以透過傳遞 python 的 list 讓 numpy 建立一個矩陣：

```python
np.array([[1,2],[3,4]])
```

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-create-2d.png)

我們也可以使用上面提到的方法 (`ones()`、`zeros()` 和 `random.random()`)，只要傳入一個 tuple 來描述我們建立矩陣的維度即可：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-ones-zeros-random.png)

### 矩陣運算

當兩個矩陣的大小相同時，我們可以透過運算元 (`+-*/`) 來對其進行相加或相乘。Numpy 是透過 position-wise 的方式進行運算：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-arithmetic.png)

我們也可以針對不同大小的矩陣進行運算，前提是其中一個矩陣的的某一維度為 1 (比如說其中一個矩陣只有一行或一列)，如此一來，numpy 就可以透過廣播的機制來進行運算：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-broadcast.png)

### 內積

算術運算和[矩陣乘法](https://www.mathsisfun.com/algebra/matrix-multiplying.html)一個最主要的區別在於內積。在 Numpy，每一個矩陣都有一個 `dot()` 方法，我們可以透過它讓矩陣之間進行內積運算：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-dot-product-1.png)

我在上圖的右下角顯示了矩陣的維度來強調相臨的兩個維度必須相同，你可以把上述的運算看作：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-dot-product-2.png)

### 矩陣索引

當我們在矩陣的運算時，索引和切片變得相當有用：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-indexing.png)

### 矩陣聚合 (aggregation)

我們可以針對矩陣進行聚合操作，就和向量一樣：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-aggregation-1.png)

我們不僅可以針對整了矩陣的值進行聚合操作，也可以透過 `axis` 參數來對行或列進行操作：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-aggregation-4.png)

## 轉置和 reshape

矩陣經常會進行的操作是轉置，當我們要對兩個矩陣進行內積操作時，經常會需要將其共享的維度對齊。在 Numpy 中，有一個方便的屬性 `T` 可以得到一個轉置矩陣：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-transpose.png)

在更進階的使用情境中，你可能需要變換某個特定矩陣的維度。這是因為在機器學習的應用中，特定的模型會需要特定的輸入維度，而這個維度可能跟你原本的資料集不同。在 numpy 中，`reshape()` 方法可以很方便地讓你變更資料的維度。你只要將所需的維度傳入此方法即可，也可以傳入 -1，numpy 會自動判斷出正確的維度：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-reshape.png)

## 更多維度

上述所提到的任何操作，都可以套用在任意的維度上，其核心的資料結構叫做 ndarray (N 維陣列)。

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-3d-array.png)

在很多情況下，處理一個新的維度只需要在 numpy 的函數中多增加一個逗號：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-3d-array-creation.png)

注意：當你要顯示一個 3 維的 numpy 陣列時，其顯示方式和在此文中所見不同，numpy 會從最後一維開始呈現，意思就是 `np.ones((4,3,2))` 會顯示如下：

```
array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])
```

## 實務用法

作為學習到目前的回報，底下是一些透過 numpy 陣列來完成特定任務的範例。

### 公式

實作需要透過陣列或向量來完成的數學公式是 numpy 主力的戰場之一，這也是為什麼 numpy 在 python 的社群中會被用在科學運算的原因。舉例來說，均方差是監督式學習來處理回歸問題的核心：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/mean-square-error-formula.png)

實作此公式在 numpy 中很容易：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mean-square-error-formula.png)

這美妙的地方在於，numpy 不在乎 `predictions` 和 `labels` 裡面是一個還是一千個值 (只要它們的大小相同)。我們接下來會一步步拆解這個範例：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-1.png)

predictions 和 labels 向量都有三個值，也就是 n = 3，在我們進行相減後，結果如下：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-2.png)

接著對向量進行平方，得到：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-3.png)

接著進行加總：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-4.png)

此結果即是 prediction 的誤差值，以及此模型的品質。

## 資料表示

想想看所有你需要用來處理和建立模型的資料 (例如：表格、影像、聲音...等等)，它們有許多都非常適合使用 n 維陣列來表示：

### 表格和電子試算表

- 電子試算表或是表格是一個二維陣列。每一個電子試算表中的工作表都可以有他自己的變數。在 python 中處理這類型資料最熱門的方法是使用 [pandas dataframe](https://jalammar.github.io/gentle-visual-intro-to-data-analysis-python-pandas/)，它正是建構在 numpy 之上的套件。

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/excel-to-pandas.png)

### 聲音和時序資料

- 聲音是一維陣列的檔案格式。陣列中的每一個值代表聲音訊號的一小部分。CD 品質的聲音每一秒會有 44,100 筆資料，每一筆資料是 -32767 到 32768 的整數。換句話說，如果你有一個長度十秒的 CD 聲音檔案，你可以透過 10 * 44,100 = 441,000 的 numpy 陣列來讀取資料。如果想要讀取聲音檔案的第一秒，只需要將資料讀入 numpy 陣列中，然後透過 `audio[:44100]` 就可以讀取了。

底下是一段聲音檔案：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-audio.png)

而時間序列的資料也是相同的處理方法 (比如說，股價隨著時間波動的資料)。

### 圖片

- 一張圖片是一個像素所形成的矩陣 (長 * 寬)
    - 如果圖片是黑白的 (也就是灰階圖片)，每一個像素可以透過單一數字表示 (通常會介於 0 (黑色) 到 255 (白色) 之間)。當你想要擷取一張圖片左上角 10 x 10 像素的圖片時，只要透過 numpy `image[:10, :10]` 即可：

底下是一張灰階圖片的範例：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-grayscale-image.png)

- 如果圖片是彩色的，每一個像素會用三個數字來表示 - 紅色、綠色和藍色。這種情況下，我們需要一個三維陣列 (因為每個位置只能包含一個數字)。所以一張彩色圖片會透過 ndarray 的資料結構來表示：(長 * 寬 * 3)

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-color-image.png)

### 語言

如果我們想要處理文字，狀況會有一點點不一樣。當你想要透過數值來表示文字的時候，你需要建立一個詞庫 (這個詞庫指的是模型需要用到的所有單字的列表)，還有一個 [嵌入的步驟](https://jalammar.github.io/illustrated-word2vec/)。讓我們一步步來看如何處理底下這個詩句：

“Have the bards who preceded me left any theme unsung?”

在模型想要用數值來表示上面該詩句之前，需要先看過大量的文字。我們可以處理一個[小的資料集](http://mattmahoney.net/dc/textdata.html)來看看要怎麼建立一個詞庫(共有 71,290 個字)：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-vocabulary.png)

上面的詩句可以被分割成一個 token 的陣列 (基於某些規則所分割出來的字或部分字)：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-tokenization.png)

接著，我們用詞庫中的 id 來取代每個字：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-ids.png)

這一些 id 對於模型來說沒有提供有用的資訊，所以在交給模型訓練之前，我們需要使用 [word2vec embedding](https://jalammar.github.io/illustrated-word2vec/) 來取代原本的 id 表示法 (在這個範例中是一個 50 維的 embedding)：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-embeddings.png)

你可以看到這個 numpy 陣列的維度是 [embedding_dimension x sequence_length]，在實務上，呈現的樣子可能不太一樣，但在這裡為了視覺的一致性，我透過下圖來表示其結果。由於效能的考量，深度學習模型會保留等同於 batch 大小第一維 (因為當多筆訓練資料時，模型就可以透過平行化的方式來訓練)。在這種情況下，`reshape()` 就變得很有用，比如說像 [Bert](https://jalammar.github.io/illustrated-bert/) 模型的輸入維度就會是 [batch_size, sequence_length, embedding_size]。

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-bert-shape.png)

現在上述的詩句被表示成數值形式，模型就可以對其進行訓練。其他行雖然目前是空白的，但它將會被更多的訓練資料給填滿。
