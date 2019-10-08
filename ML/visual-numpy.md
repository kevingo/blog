本文翻譯自 [A Visual Intro to NumPy and Data Representation
](https://jalammar.github.io/visual-numpy/)，對於從事資料分析或機器學習的朋友來說，numpy 一定是不陌生的 Python 套件。不管是資料處理所使用的 Pandas 、機器學習用到的 scikit-learn 或是 deep learning 所使用的 tensorflow 或 pytorch，底層在資料的操作或儲存上，大多會用到 numpy 來做科學的操作。而本篇文章以圖文並茂的方式詳細說明了 numpy 重要的操作，並且告訴你 numpy 的資料結構如何用來儲存文字、圖片、聲音等重要的資料。同時，numpy 底層使用 C/C++ 和 fortran 來實作，也大幅的提升了多維度資料運算上的效能。希望透過此文的分享，讓大家在學習 numpy 的過程中能夠更加清楚其操作與用途。

---

# 圖解 numpy 與資料表示法

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array.png)

[NumPy](https://www.numpy.org/) 套件是 python 生態系中針對資料分析、機器學習和科學計算的重要角色。它極大的簡化了向量和矩陣的操作運算，某些 python 的主要套件大量的依賴 numpy 作為其架構的基礎 (例如：scikit-learn、SciPy、Pandas 和 tensorflow)。除了可以針對資料進行切片 (slice) 和 切塊 (dice) 之外，熟悉 numpy 還可以對使用上述套件帶來極大的好處。

在本文中，我們會學習 numpy 主要的使用方式，並且看到它如何用來表示不同類型的資料 (表格、影像、文字 ... 等) 作為機器學習模型的輸入。

```python
import numpy as np
```

## 建立陣列

We can create a NumPy array (a.k.a. the mighty [ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)) by passing a python list to it and using ` np.array()`. In this case, python creates the array we can see on the right here:

我們可以透過 `np.array()` 並傳入一個 python list 來建立一個 numpy 的陣列 (又叫 [ndarray]((https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)))，如下所示：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/create-numpy-array-1.png)

通常我們希望 numpy 能夠初始化陣列的值，因此它提供了像 ones()、zeros() 和 random.random() 等方法。我們只要傳入希望產生的值即可：

![image](https://github.com/kevingo/blog/blob/master/screenshot/create-numpy-array-ones-zeros-random.png)

一但我們建立了陣列後，就可以透過有趣的方式盡情的操作他們。

## 矩陣運算

讓我們建立兩個 numpy 陣列來展示如何進行運算，分別是 `data` 和 `ones`：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-arrays-example-1.png)

將這兩個 numpy 陣列依照位置相加 (即每一行相加)，只要輸入 `data + ones` 即可：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-arrays-adding-1.png)

當我開始學習這個工具後，我發現這樣抽象的思考讓我不用透過類似迴圈的方式來進行計算，而這樣的方式可以讓我透過更高層次的角度來思考問題。

而除了相加之外，我們還可以進行以下的操作：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-subtract-multiply-divide.png)

通常的情況下，我們會希望一個陣列可以和單一數字進行運算 (即向量和純量之間進行運算)。比如說，陣列中的數字是以英里為單位的距離，而我們希望將其轉換為公里，只需要透過 `data * 1.6` 即可：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-broadcast.png)

看到 numpy 是如何處理這樣的操作了嗎？這個概念稱為 *廣播 (broadcasting)*，它非常有用。

## 索引

我們可以像 python 的 list 進行切片一樣，對 numpy 的陣列進行索引和切片：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-slice.png)

## 聚合

Numpy 另外一個好處是提供了聚合函式：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-aggregation.png)

除了 `min`、`max` 和 `sum` 之外，你還可以使用像是 `mean` 來得到平均值，`prod` 來得到所有元素的乘積，`std` 來得到標準差，以及[其他更多的功能](https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html)。

## 更多維度

上述我們所看到的範例都是在單一為度的向量上進行，而 numpy 之美在於這些操作可以擴展到任意維度的資料上。

### 建立矩陣

我們可以透過傳遞 python 的 list 型態來讓 numpy 建立一個矩陣：

```python
np.array([[1,2],[3,4]])
```

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-create-2d.png)

我們也可以使用上面提到的方法 (`ones()`、`zeros()` 和 `random.random()`)，只要傳入一個 tuple 來描述我們建立矩陣的維度即可：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-ones-zeros-random.png)

### 矩陣運算

當兩個矩陣的大小一樣時，我們可以透過運算元 (`+-*/`) 來對其進行相加或相乘。Numpy 是透過 position-wise 的方式進行運算：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-arithmetic.png)

我們也可以針對不同大小的矩陣進行運算，前提是其中一個矩陣的的某一維度為 1 (比如說其中一個矩陣只有一行或一列)，如此一來，numpy 就可以透過廣播的機制來進行運算：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-broadcast.png)

### 內積

算術運算和 [矩陣乘法](https://www.mathsisfun.com/algebra/matrix-multiplying.html) 一個最主要的區別在於內積。在 Numpy，每一個矩陣都有一個 `dot()` 方法，我們可以透過它讓矩陣之間進行內積運算：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-dot-product-1.png)

我在上圖的右下角顯示了矩陣的維度來強調相臨的兩個維度必須要有相同的維度，你可以把上述的運算看作：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-dot-product-2.png)

### 矩陣索引

當我們在矩陣的運算時，索引和切片變得相當有用：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-indexing.png)

### 矩陣聚合 (aggregation)

我們可以針對矩陣進行聚合操作，就和向量一樣：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-aggregation-1.png)

我們不僅可以針對整了矩陣的值進行聚合操作，也可以透過 `axis` 參數來對行或列進行操作：

![image](https:;plopol//raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-aggregation-4.png)

## 轉置和 reshape

矩陣經常會進行的必要操作是轉置，當我們要對兩個矩陣進行內積操作時，經常會需要將其共享的維度對齊。在 Numpy 中，有一個方便的屬性 `T` 可以得到一個轉置矩陣：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-transpose.png)

在更進階的使用情境中，你可能需要變換某個特定矩陣的維度。這是因為在機器學習的應用中，特定的模型會需要特定的輸入維度，而這個維度可能跟你原本的資料集不同。在 numpy 中，`reshape()` 方法可以很方便地讓你變更資料的維度。你只要將所需的維度傳入此方法即可，也可以傳入 -1，numpy 會自動判斷出正確的維度：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-reshape.png)

## 更多維度

上述所提到的任何操作，都可以套用在任意的維度上，其核心的資料結構叫做 ndarray (N 維陣列)。

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-3d-array.png)

在很多情況下，處理一個新的維度只需要在 numpy 的函數中多增加一個逗號：

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-3d-array-creation.png)

注意：當你要顯示一個 3 維的 numpy 陣列時，其顯示方式和在此文中所見不同，numpy 會從最後一為開始呈現，意思就是 `np.ones((4,3,2))` 會顯示如下：

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

現在作為回報，底下是一些透過 numpy 陣列來完成特定任務的範例。

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

- 電子試算表或是表格是一個二維陣列。每一個電子試算表中的工作表都可以有他自己的變數。在 python 中處理這類型資料最熱門的方法是使用 [pandas dataframe](https://jalammar.github.io/gentle-visual-intro-to-data-analysis-python-pandas/)，它正是建構在 numpy 之上。

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

### Language

If we’re dealing with text, the story is a little different. The numeric representation of text requires a step of building a vocabulary (an inventory of all the unique words the model knows) and an [embedding step](https://jalammar.github.io/illustrated-word2vec/). Let us see the steps of numerically representing this (translated) quote by an ancient spirit:

“Have the bards who preceded me left any theme unsung?”

A model needs to look at a large amount of text before it can numerically represent the anxious words of this warrior poet. We can proceed to have it process a [small dataset](http://mattmahoney.net/dc/textdata.html) and use it to build a vocabulary (of 71,290 words):

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-vocabulary.png)

The sentence can then be broken into an array of tokens (words or parts of words based on common rules):

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-tokenization.png)

We then replace each word by its id in the vocabulary table:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-ids.png)

These ids still don’t provide much information value to a model. So before feeding a sequence of words to a model, the tokens/words need to be replaced with their embeddings (50 dimension [word2vec embedding](https://jalammar.github.io/illustrated-word2vec/) in this case):

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-embeddings.png)

You can see that this NumPy array has the dimensions [embedding_dimension x sequence_length]. In practice these would be the other way around, but I’m presenting it this way for visual consistency. For performance reasons, deep learning models tend to preserve the first dimension for batch size (because the model can be trained faster if multiple examples are trained in parallel). This is a clear case where `reshape()` becomes super useful. A model like [BERT](https://jalammar.github.io/illustrated-bert/), for example, would expect its inputs in the shape: [batch_size, sequence_length, embedding_size].

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-nlp-bert-shape.png)

This is now a numeric volume that a model can crunch and do useful things with. I left the other rows empty, but they’d be filled with other examples for the model to train on (or predict).

(It turned out the [poet’s words](https://en.wikisource.org/wiki/The_Poem_of_Antara) in our example were immortalized more so than those of the other poets which trigger his anxieties. Born a slave owned by his father, [Antarah’s](https://en.wikipedia.org/wiki/Antarah_ibn_Shaddad) valor and command of language gained him his freedom and the mythical status of having his poem as one of [seven poems suspended in the kaaba](https://en.wikipedia.org/wiki/Mu%27allaqat) in pre-Islamic Arabia).
