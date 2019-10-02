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

### Creating Matrices

We can pass python lists of lists in the following shape to have NumPy create a matrix to represent them:

```python
np.array([[1,2],[3,4]])
```

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-array-create-2d.png)

We can also use the same methods we mentioned above (`ones()`, `zeros()`, and `random.random()`) as long as we give them a tuple describing the dimensions of the matrix we are creating:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-ones-zeros-random.png)

### Matrix Arithmetic

We can add and multiply matrices using arithmetic operators (`+-*/`) if the two matrices are the same size. NumPy handles those as position-wise operations:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-arithmetic.png)

We can get away with doing these arithmetic operations on matrices of different size only if the different dimension is one (e.g. the matrix has only one column or one row), in which case NumPy uses its broadcast rules for that operation:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-broadcast.png)

### Dot Product

A key distinction to make with arithmetic is the case of [matrix multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html) using the dot product. NumPy gives every matrix a `dot()` method we can use to carry-out dot product operations with other matrices:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-dot-product-1.png)

I’ve added matrix dimensions at the bottom of this figure to stress that the two matrices have to have the same dimension on the side they face each other with. You can visualize this operation as looking like this:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-dot-product-2.png)

### Matrix Indexing

Indexing and slicing operations become even more useful when we’re manipulating matrices:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-indexing.png)

### Matrix Aggregation

We can aggregate matrices the same way we aggregated vectors:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-aggregation-1.png)

Not only can we aggregate all the values in a matrix, but we can also aggregate across the rows or columns by using the `axis` parameter:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-matrix-aggregation-4.png)

## Transposing and Reshaping

A common need when dealing with matrices is the need to rotate them. This is often the case when we need to take the dot product of two matrices and need to align the dimension they share. NumPy arrays have a convenient property called `T` to get the transpose of a matrix:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-transpose.png)

In more advanced use case, you may find yourself needing to switch the dimensions of a certain matrix. This is often the case in machine learning applications where a certain model expects a certain shape for the inputs that is different from your dataset. NumPy’s `reshape()` method is useful in these cases. You just pass it the new dimensions you want for the matrix. You can pass -1 for a dimension and NumPy can infer the correct dimension based on your matrix:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-reshape.png)

## Yet More Dimensions

NumPy can do everything we’ve mentioned in any number of dimensions. Its central data structure is called ndarray (N-Dimensional Array) for a reason.

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-3d-array.png)

In a lot of ways, dealing with a new dimension is just adding a comma to the parameters of a NumPy function:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-3d-array-creation.png)

Note: Keep in mind that when you print a 3-dimensional NumPy array, the text output visualizes the array differently than shown here. NumPy’s order for printing n-dimensional arrays is that the last axis is looped over the fastest, while the first is the slowest. Which means that `np.ones((4,3,2))` would be printed as:

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

## Practical Usage

And now for the payoff. Here are some examples of the useful things NumPy will help you through.

### Formulas

Implementing mathematical formulas that work on matrices and vectors is a key use case to consider NumPy for. It’s why NumPy is the darling of the scientific python community. For example, consider the mean square error formula that is central to supervised machine learning models tackling regression problems:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/mean-square-error-formula.png)

Implementing this is a breeze in NumPy:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mean-square-error-formula.png)

The beauty of this is that numpy does not care if `predictions` and `labels` contain one or a thousand values (as long as they’re both the same size). We can walk through an example stepping sequentially through the four operations in that line of code:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-1.png)

Both the predictions and labels vectors contain three values. Which means n has a value of three. After we carry out the subtraction, we end up with the values looking like this:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-2.png)

Then we can square the values in the vector:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-3.png)

Now we sum these values:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-mse-4.png)

Which results in the error value for that prediction and a score for the quality of the model.

## Data Representation

Think of all the data types you’ll need to crunch and build models around (spreadsheets, images, audio…etc). So many of them are perfectly suited for representation in an n-dimensional array:

### Tables and Spreadsheets

- A spreadsheet or a table of values is a two dimensional matrix. Each sheet in a spreadsheet can be its own variable. The most popular abstraction in python for those is the [pandas dataframe](https://jalammar.github.io/gentle-visual-intro-to-data-analysis-python-pandas/), which actually uses NumPy and builds on top of it.

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/excel-to-pandas.png)

### Audio and Timeseries

- An audio file is a one-dimensional array of samples. Each sample is a number representing a tiny chunk of the audio signal. CD-quality audio may have 44,100 samples per second and each sample is an integer between -32767 and 32768. Meaning if you have a ten-seconds WAVE file of CD-quality, you can load it in a NumPy array with length 10 * 44,100 = 441,000 samples. Want to extract the first second of audio? simply load the file into a NumPy array that we’ll call `audio`, and get `audio[:44100]`.

Here’s a look at a slice of an audio file:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-audio.png)

The same goes for time-series data (for example, the price of a stock over time).

### Images

- An image is a matrix of pixels of size (height x width).
    - If the image is black and white (a.k.a. grayscale), each pixel can be represented by a single number (commonly between 0 (black) and 255 (white)). Want to crop the top left 10 x 10 pixel part of the image? Just tell NumPy to get you `image[:10,:10]`.

Here’s a look at a slice of an image file:

![image](https://raw.githubusercontent.com/kevingo/blog/master/screenshot/numpy-grayscale-image.png)

- If the image is colored, then each pixel is represented by three numbers - a value for each of red, green, and blue. In that case we need a 3rd dimension (because each cell can only contain one number). So a colored image is represented by an ndarray of dimensions: (height x width x 3).

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
