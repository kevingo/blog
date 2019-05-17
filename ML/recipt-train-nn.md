# 一個訓練神經網路的菜單

幾週之前，我發了一則 [tweet](https://twitter.com/karpathy/status/1013244313327681536?lang=en) - "最常見的神經網路錯誤"，列出了一些和訓練神經網路常見的問題。這則 tweet 比我預期得到還要多的回饋 (也包含了某次的[網路研討會 :)](https://www.bigmarker.com/missinglink-ai/PyTorch-Code-to-Unpack-Andrej-Karpathy-s-6-Most-Common-NN-Mistakes))。顯然，許多人在 "卷積神經網路是這樣運作的" 和 "我們的卷積神經網路已經達到了最佳結果" 這兩者之前還存在巨大的差距。

因此，我想如果在我蓋滿塵土的部落格上，將 tweet 上所述擴展成一個主題，應該會是一件有趣的事情。然而，與其列出一些常見的錯誤一些常見的錯誤或深入分析他們，不如我想更深入探討如何避免發生這些錯誤 (或是快速的修正它們)。而其中的關鍵是遵循某個特定的流程，就我所知，目前並沒有人將這樣的資訊完整的記錄下來。讓我們開始吧。

### 1) 神經網路的訓練一種抽象漏洞

有一種說法是開始訓練神經網路是很容易的。許多書籍和框架都自豪的顯示出可以用 30 行左右的神奇的程式碼來解決你的問題，這帶給大眾一個錯誤的印象，以為訓練神經網路是隨插即用，我們經常可以看到下面的程式碼片段：


```python
>>> your_data = # 帶入你神奇的資料
>>> model = SuperCrossValidator(SuperDuper.fit, your_data, ResNet50, SGDOptimizer)
# 接著你就戰勝世界了
```

這些函式庫和範例讓我們激起了對於標準軟體設計的部分 - 一個簡潔的 API 設定與乾淨的抽象層是很容易達到的。比如 Requests 函式庫展示了這樣的能力：

```python
>>> r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
>>> r.status_code
200
```

這很酷！一個勇敢的開發者已經理解了查詢字串、url、GET/POST 請求、HTTP 連線等資訊，並在相當大程度上隱藏了那幾行程式碼背後的複雜度。這剛好是我們熟悉且期待的。但不幸的是，神經網路並不是這樣的。他並不是一個 "現成" 的技術，與你訓練一個 ImageNet 的分類器不同。我嘗試在我的文章 "是的，你應該理解反向傳遞" 中，透過反向傳播的範例說明這一部分，並稱其為 "抽象漏洞"，但不幸的，這問題更加嚴重。反向傳遞加上隨機梯度下降並不會神奇的讓你的網路訓練正常，Batch Normalization 也不會讓網路收斂的更快。RNN 不會讓你的文字資料神奇的 "嵌入" 網路中，而僅僅因為你的問題可以透過增強式學習來建模，但不表示你應該如此做。如果你堅持使用該技術，但不去了解其運作原理，你非常有可能會失敗。這讓我想到 ...

### 2) 訓練神經網路會悄悄的失敗

當你的程式碼寫錯或是設定有錯誤時，通常會得到一些例外的錯誤訊息。本來應該是字串，你輸入了一個整數、某個函式只預期接收三個參數、引用失敗、某個鍵值不存在、兩個 list 中的元素數量不相等。而且，我們通常可以針對特定的功能建立單元測試。

當我們在訓練神經網路時，解決這些問題只是訓練的開始。在語法上一切正確，但很有可能在整個訓練還是出了問題，而且很難解釋清楚為什麼。"可能錯誤的面向" 非常大，邏輯 (跟語法無關) 面的問題，同時很難進行單元測試。舉例來說，也許你在做資料增量時將圖片進行左右翻轉，但資料的標籤忘了進行同步的處理。你的神經網路依舊可以 (令人震驚的) 進行訓練，因為網路可以在內部學習翻轉的圖片，然後在預測的結果上進行了左右翻轉。或是你的自迴歸模型因為不小心將預測的對象當成輸入。或是你在訓練時想要調整梯度，但不小心改到損失函數，導致異常的樣本在訓練時期被忽略。或是你使用預訓練的模型來初始化參數，但沒有使用原始的平均值來對資料進行標準化。或是你只是用錯了正規化的權重、學習率、衰減率、模型的大小等。因此，錯誤的設定只會在你運氣好的時候得到例外的錯誤訊息，大多時候模型依舊會進行訓練，但默默的輸出看起來不太好的結果。

因此，(這很難不強調其重要性) "快速且激烈" 訓練神經網路並沒有辦法起到作用並達到失敗。失敗的過程是在訓練一個良好的神經網路相當自然的部份，但這樣的過程可以透過謹慎、某種程度的視覺化來降低其失敗的風險。在我的經驗中，耐心和對於細節的注意是訓練神經網路最重要的關鍵。

## 秘訣

基於以上兩個狀況，我建立了一個開發神經網路的流程，讓我能夠在新問題上運用神經網路來解決，底下我會試著介紹。你會發現，我特別重視上述兩個狀況。特別來說，這個流程遵循從簡單到複雜的規則，而且在每一個步驟上，我們建立具體的假設，並且透過實驗來驗證它，直到我們發現問題為止。我們要極力避免的狀況是一次引入太多 "未經驗證" 的複雜狀況，這會導致我們要花許多精力在尋找臭蟲/錯誤的設定上。如果撰寫你的神經網路程式就像在進行模型訓練一樣，我們會使用非常小的學速率進行猜測，並且在每個訓練回合中，在完整的測試資料集上進行驗證。

### 1. 徹底了解你的資料

訓練神經網路的第一步不是開始撰寫任何程式碼，而是徹底了解你的資料。這個步驟相當關鍵，我喜歡花費大量的時間 (以小時計) 瀏覽數千筆資料，瞭解資料的分佈並尋找規則。幸運的是，你的大腦在這方面是很擅長的。有一次我發現重複的資料、另一次我則找到錯誤的圖片/標籤。我會尋找資料的不平衡或偏誤。我通常也會觀察自己如何針對資料進行分類，這個過程會提示最後我們要使用的架構為何。舉例來說，我們需要局部的特徵還是全局的上下文？資料有多大的變化？這些變化透過什麼形式呈現？哪些變化是假的，可以處理掉？空間位置是否重要，或是我們想要將其平均掉？細節有多重要，而我們可以接受多大程度的取樣？標籤有多少雜訊？

此外，由於神經網路實際上是資料的壓縮/編譯過後的版本，你必須要看看那些預測錯誤的資料，瞭解預測不一致的原因為何。如果你的神經網路給你的預測結果和你在資料中觀察到的不一致時，代表一定漏掉一些東西了。

一旦你得到一些定性的感覺後，開始撰寫一些簡單的程式碼來搜尋/過濾/排序任何你想得到的資料是一個很好的主意 (例如：標籤的種類、數量或大小等)，同時透過視覺化來檢視其分佈，並且找出沿著任何座標上的異常值。異常值特別能展現出資料的品質或前處理上的一些錯誤。

### 2. 建立完整的訓練/評估框架 + 取得基準

當我們了解資料之後，就可以開始設計超級強大的 Multi-scale ASPP FPN ResNet 模型了嗎？當然不是，這一條道路上是充滿崎嶇的。我們的下一步是建立一個完整的訓練 + 評估的框架，並且透過一系列的實驗來驗證其正確性。在這個階段最好挑選一些簡單的模型，例如：線性分類器或非常小的 ConvNet。我們希望透過這樣的方法進行訓練、始覺化損失或其他的指標 (例如：準確率)、模型預測的結果，並在這樣的過程透過實驗來驗證我們一系列的假設。

在這個階段的一些秘訣 & 技巧：

- 設定相同的隨機種子來讓你每次執行相同程式碼時，可以得到相同的結果。這消除了不確定的因素，並且讓你保持清醒。
- simplify. Make sure to disable any unnecessary fanciness. As an example, definitely turn off any data augmentation at this stage. Data augmentation is a regularization strategy that we may incorporate later, but for now it is just another opportunity to introduce some dumb bug.
- add significant digits to your eval. When plotting the test loss run the evaluation over the entire (large) test set. Do not just plot test losses over batches and then rely on smoothing them in Tensorboard. We are in pursuit of correctness and are very willing to give up time for staying sane.
- verify loss @ init. Verify that your loss starts at the correct loss value. E.g. if you initialize your final layer correctly you should measure -log(1/n_classes) on a softmax at initialization. The same default values can be derived for L2 regression, Huber losses, etc.
- init well. Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.
- human baseline. Monitor metrics other than loss that are human interpretable and checkable (e.g. accuracy). Whenever possible evaluate your own (human) accuracy and compare to it. Alternatively, annotate the test data twice and for each example treat one annotation as prediction and the second as ground truth.
- input-indepent baseline. Train an input-independent baseline, (e.g. easiest is to just set all your inputs to zero). This should perform worse than when you actually plug in your data without zeroing it out. Does it? i.e. does your model learn to extract any information out of the input at all?
- overfit one batch. Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model (e.g. add layers or filters) and verify that we can reach the lowest achievable loss (e.g. zero). I also like to visualize in the same plot both the label and the prediction and ensure that they end up aligning perfectly once we reach the minimum loss. If they do not, there is a bug somewhere and we cannot continue to the next stage.
- verify decreasing training loss. At this stage you will hopefully be underfitting on your dataset because you’re working with a toy model. Try to increase its capacity just a bit. Did your training loss go down as it should?
- visualize just before the net. The unambiguously correct place to visualize your data is immediately before your y_hat = model(x) (or sess.run in tf). That is - you want to visualize exactly what goes into your network, decoding that raw tensor of data and labels into visualizations. This is the only “source of truth”. I can’t count the number of times this has saved me and revealed problems in data preprocessing and augmentation.
- visualize prediction dynamics. I like to visualize model predictions on a fixed test batch during the course of training. The “dynamics” of how these predictions move will give you incredibly good intuition for how the training progresses. Many times it is possible to feel the network “struggle” to fit your data if it wiggles too much in some way, revealing instabilities. Very low or very high learning rates are also easily noticeable in the amount of jitter.
- use backprop to chart dependencies. Your deep learning code will often contain complicated, vectorized, and broadcasted operations. A relatively common bug I’ve come across a few times is that people get this wrong (e.g. they use view instead of transpose/permute somewhere) and inadvertently mix information across the batch dimension. It is a depressing fact that your network will typically still train okay because it will learn to ignore data from the other examples. One way to debug this (and other related problems) is to set the loss to be something trivial like the sum of all outputs of example i, run the backward pass all the way to the input, and ensure that you get a non-zero gradient only on the i-th input. The same strategy can be used to e.g. ensure that your autoregressive model at time t only depends on 1..t-1. More generally, gradients give you information about what depends on what in your network, which can be useful for debugging.
- generalize a special case. This is a bit more of a general coding tip but I’ve often seen people create bugs when they bite off more than they can chew, writing a relatively general functionality from scratch. I like to write a very specific function to what I’m doing right now, get that to work, and then generalize it later making sure that I get the same result. Often this applies to vectorizing code, where I almost always write out the fully loopy version first and only then transform it to vectorized code one loop at a time.

### 3. Overfit
At this stage we should have a good understanding of the dataset and we have the full training + evaluation pipeline working. For any given model we can (reproducibly) compute a metric that we trust. We are also armed with our performance for an input-independent baseline, the performance of a few dumb baselines (we better beat these), and we have a rough sense of the performance of a human (we hope to reach this). The stage is now set for iterating on a good model.

The approach I like to take to finding a good model has two stages: first get a model large enough that it can overfit (i.e. focus on training loss) and then regularize it appropriately (give up some training loss to improve the validation loss). The reason I like these two stages is that if we are not able to reach a low error rate with any model at all that may again indicate some issues, bugs, or misconfiguration.

A few tips & tricks for this stage:

- picking the model. To reach a good training loss you’ll want to choose an appropriate architecture for the data. When it comes to choosing this my #1 advice is: Don’t be a hero. I’ve seen a lot of people who are eager to get crazy and creative in stacking up the lego blocks of the neural net toolbox in various exotic architectures that make sense to them. Resist this temptation strongly in the early stages of your project. I always advise people to simply find the most related paper and copy paste their simplest architecture that achieves good performance. E.g. if you are classifying images don’t be a hero and just copy paste a ResNet-50 for your first run. You’re allowed to do something more custom later and beat this.
- adam is safe. In the early stages of setting baselines I like to use Adam with a learning rate of 3e-4. In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate. For ConvNets a well-tuned SGD will almost always slightly outperform Adam, but the optimal learning rate region is much more narrow and problem-specific. (Note: If you are using RNNs and related sequence models it is more common to use Adam. At the initial stage of your project, again, don’t be a hero and follow whatever the most related papers do.)
- complexify only one at a time. If you have multiple signals to plug into your classifier I would advise that you plug them in one by one and every time ensure that you get a performance boost you’d expect. Don’t throw the kitchen sink at your model at the start. There are other ways of building up complexity - e.g. you can try to plug in smaller images first and make them bigger later, etc.
- do not trust learning rate decay defaults. If you are re-purposing code from some other domain always be very careful with learning rate decay. Not only would you want to use different decay schedules for different problems, but - even worse - in a typical implementation the schedule will be based current epoch number, which can vary widely simply depending on the size of your dataset. E.g. ImageNet would decay by 10 on epoch 30. If you’re not training ImageNet then you almost certainly do not want this. If you’re not careful your code could secretely be driving your learning rate to zero too early, not allowing your model to converge. In my own work I always disable learning rate decays entirely (I use a constant LR) and tune this all the way at the very end.

### 4. Regularize
Ideally, we are now at a place where we have a large model that is fitting at least the training set. Now it is time to regularize it and gain some validation accuracy by giving up some of the training accuracy. Some tips & tricks:

- get more data. First, the by far best and preferred way to regularize a model in any practical setting is to add more real training data. It is a very common mistake to spend a lot engineering cycles trying to squeeze juice out of a small dataset when you could instead be collecting more data. As far as I’m aware adding more data is pretty much the only guaranteed way to monotonically improve the performance of a well-configured neural network almost indefinitely. The other would be ensembles (if you can afford them), but that tops out after ~5 models.
- data augment. The next best thing to real data is half-fake data - try out more aggressive data augmentation.
- creative augmentation. If half-fake data doesn’t do it, fake data may also do something. People are finding creative ways of expanding datasets; For example, domain randomization, use of simulation, clever hybrids such as inserting (potentially simulated) data into scenes, or even GANs.
- pretrain. It rarely ever hurts to use a pretrained network if you can, even if you have enough data.
- stick with supervised learning. Do not get over-excited about unsupervised pretraining. Unlike what that blog post from 2008 tells you, as far as I know, no version of it has reported strong results in modern computer vision (though NLP seems to be doing pretty well with BERT and friends these days, quite likely owing to the more deliberate nature of text, and a higher signal to noise ratio).
- smaller input dimensionality. Remove features that may contain spurious signal. Any added spurious input is just another opportunity to overfit if your dataset is small. Similarly, if low-level details don’t matter much try to input a smaller image.
- smaller model size. In many cases you can use domain knowledge constraints on the network to decrease its size. As an example, it used to be trendy to use Fully Connected layers at the top of backbones for ImageNet but these have since been replaced with simple average pooling, eliminating a ton of parameters in the process.
- decrease the batch size. Due to the normalization inside batch norm smaller batch sizes somewhat correspond to stronger regularization. This is because the batch empirical mean/std are more approximate versions of the full mean/std so the scale & offset “wiggles” your batch around more.
- drop. Add dropout. Use dropout2d (spatial dropout) for ConvNets. Use this sparingly/carefully because dropout does not seem to play nice with batch normalization.
- weight decay. Increase the weight decay penalty.
- early stopping. Stop training based on your measured validation loss to catch your model just as it’s about to overfit.
- try a larger model. I mention this last and only after early stopping but I’ve found a few times in the past that larger models will of course overfit much more eventually, but their “early stopped” performance can often be much better than that of smaller models.

Finally, to gain additional confidence that your network is a reasonable classifier, I like to visualize the network’s first-layer weights and ensure you get nice edges that make sense. If your first layer filters look like noise then something could be off. Similarly, activations inside the net can sometimes display odd artifacts and hint at problems.

### 5. Tune
You should now be “in the loop” with your dataset exploring a wide model space for architectures that achieve low validation loss. A few tips and tricks for this step:

- random over grid search. For simultaneously tuning multiple hyperparameters it may sound tempting to use grid search to ensure coverage of all settings, but keep in mind that it is best to use random search instead. Intuitively, this is because neural nets are often much more sensitive to some parameters than others. In the limit, if a parameter a matters but changing b has no effect then you’d rather sample a more throughly than at a few fixed points multiple times.

- hyper-parameter optimization. There is a large number of fancy bayesian hyper-parameter optimization toolboxes around and a few of my friends have also reported success with them, but my personal experience is that the state of the art approach to exploring a nice and wide space of models and hyperparameters is to use an intern :). Just kidding.

### 6. Squeeze out the juice

Once you find the best types of architectures and hyper-parameters you can still use a few more tricks to squeeze out the last pieces of juice out of the system:

- ensembles. Model ensembles are a pretty much guaranteed way to gain 2% of accuracy on anything. If you can’t afford the computation at test time look into distilling your ensemble into a network using dark knowledge.
- leave it training. I’ve often seen people tempted to stop the model training when the validation loss seems to be leveling off. In my experience networks keep training for unintuitively long time. One time I accidentally left a model training during the winter break and when I got back in January it was SOTA (“state of the art”).

### Conclusion
Once you make it here you’ll have all the ingredients for success: You have a deep understanding of the technology, the dataset and the problem, you’ve set up the entire training/evaluation infrastructure and achieved high confidence in its accuracy, and you’ve explored increasingly more complex models, gaining performance improvements in ways you’ve predicted each step of the way. You’re now ready to read a lot of papers, try a large number of experiments, and get your SOTA results. Good luck!