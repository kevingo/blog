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
- 簡化。不要做任何不必要的嘗試。舉例來說，在這個階段不需要做任何的資料擴展 (data augmentation)，資料擴展是一種正規化的策略，我們會在後面使用。但在這個階段引入可能會導致一些愚蠢的錯誤。
- 在你的評估階段盡量加入有意義的數字。在繪製測試的損失數值時，對整個測試資料集來進行，而不要只在批量的資料進行，然後在 Tensorboard 上去平滑此損失。
- 驗證初期的損失。驗證你的損失是不是從正確的損失值開始。舉例來說，如果你正確的初始化神經網路的最後一層，則應該在初始化實察看 softmax 上的 -log(1/n_classes)，這初始值可以使用 L2 迴歸、Huber losses 等。
- 正確的初始化。正確初始化每一層的權重。例如：如果你正在針對某些資料進行回歸分析，該資料的平均值為 50，則將偏差值設定為 50。而如果資料集的正負樣本比例是不平衡的 1:10，請在你的資料上將這樣的偏差放入，讓你的網路可以在一開始就可以處理這樣的狀況。正確的設定初始化參數可以加速收斂的過程，並消除 "曲棍棒球" 般的損失函數，因為在最初的幾次迭代中，網路基本上只是學習偏差。
- 人工驗證。除了損失之外，監控任何可以透過人工解釋的指標 (例如：準確性)。盡可能針對模型預測出的準確性和自我評估的準確性進行比較。或是針對每筆測試資料進行二次標注，將第一個標注當作預測值，第二個標注當作標準答案。
- 設定與輸入無關的基準。訓練一個與輸入無關的基準值 (例如：最簡單的方法就是將輸入設定為零)，這樣模型的表現應該要比起原本的資料差很多，對嗎？也就是說你的模型是否有學會從輸入中學到任何訊息？
- 在小批次的資料上過擬合。嘗試在模型上增加一兩層網路或幾個 filter，看看模型能否在少量的資料上過擬合 (例如兩筆資料)，確認我們可以達到的最小損失值 (比如說零)。我還會在一張圖上同時顯示預測值和實際值，並確保一但達到最小損失時，他們會完美的重合。如果沒有，那就代表某些地方出現錯誤，就無法進行下一階段的任務。
- 驗證降低訓練損失。在這個階段你希望模型在資料上是欠擬合的，因為你使用像玩具一樣的模型，試著增加模型的能力，觀察看看訓練損失是否正在下降？
- 在訓練之前進行資料視覺化。在 y_hat = model(x) (或 tf 中的 sess.run) 之前，明確的進行資料的視覺化，目的在於清楚的知道什麼樣的資料會輸入到神經網路中，將原始的 tensor 資料和類別標籤呈現出來，這是唯一的 "真實資料"。這節省了我大量的時間，並且清楚的揭露資料在預處理和增量中所遇到的問題。
- 視覺化預測變動的部分。我喜歡在模型訓練過程中，針對固定批次的資料進行可視覺化。這些「動態」的部分給你很好的直覺來觀察訓練的過程。當你看到網路在訓練時不斷跳動，可能代表你選的模型不適合你的資料。而過高或過低的學習率也會造成模型學習時跳動很明顯。
- 使用反向傳遞來繪製相依性。你的深度學習程式碼經常會包含複雜的、向量化和廣播的操作。我遇到一個經常發生的錯誤是人們自己弄錯了 (例如他們在某處使用 view 而不是 transpose/permute)，並且無意間混合了批量的維度。令人失望的是，你的神經網路通常還是可以正常的訓練，因為它會學著忽略該筆錯誤的資料。檢查這種錯誤 (和其他相關的問題) 的其中一種方法是，將你的損失值 (loss) 設定為特定值，例如樣本 i 的所有輸出的總和，執行反向傳遞，確保你僅僅會在第 i 個輸入時會得到非零的梯度值。類似的策略也可以用在，例如說驗證你的自迴歸模型在時間 t 時，僅依賴於 1 ... t-1。更一般來說來說，梯度提供你在網路中什麼東西互相依賴的資訊，這對於找出臭蟲來說相當有用。
- 一般化特例。這一點更像是一個撰寫程式的技巧，我經常看到有許多人為了想要寫出一個非常通用的函式而產生 bug。而我的做法是會先寫一個專門用來解決目前工作或問題的函式，驗證它是正確無誤，之後再擴展到更通用的版本。這在撰寫以向量為主的程式碼時相當適用，一開始我會寫出一個都是用 for 迴圈來計算的版本，之後再修改為以向量計算為主的版本。

### 3. 過擬合

到了這個階段，我們應該對資料有很深入的理解，並且有一個完整的訓練 + 評估的工作流程。對於任何給定的模型，我們可以 (重複) 計算出一個可以信任的指標。我們同樣會擁有一個獨立於輸入的表現作為效能的基準，也會有一些基本的基準 (我們的模型最好能夠打敗這些基準)，而我們對於人類的性能有一個粗略的感覺 (希望能夠達到這一點)，現在我們已經為迭代模型做好準備了。

尋找一個好的模型分為兩個階段：首先，訓練一個足夠大的模型使其容易過擬合 (即關注在降低訓練的損失上)，接著嘗試正規化 (放棄訓練損失，改善驗證資料集的損失值)。我喜歡這兩階段的原因在於，如果我不能降低錯誤率時，那代表一定有一些問題、bug 或錯誤的配置。

在這一階段的一些提示和技巧：

- 挑選模型。為了達到良好的訓練損失，你需要為資料挑選一個合適的架構。而我第一個建議是：不要當一個英雄。我看過許多人渴望變的瘋狂與富有創意，把神經網路的架構像樂高積木一樣堆疊成各式各樣的架構。在專案的早期，盡量避免這樣做。我總是建議人們去找尋最相關的論文，複製貼上他們最簡單的架構，就可以得到很好的效能。舉例來說，如果你打算進行圖片分類，僅僅複製 ResNet-50 來作為你第一次食做的模型，你可以在之後針對模型進行客製化，然後超越其原本的效能。
- adam 是很安全的選擇。在設定基準的早期階段，我喜歡使用 adam，學習率為 3e-4。根據我的經驗，adam 對於超參數的的選擇來說更為寬容，包括錯誤的學習率。對於卷積神經網路來說，一個調整良好的 SGD 幾乎可以表現得比 Adam 來的好，不過其最佳學習率的區間要窄得多，而且是針對特定問題而言 (注意：如果你使用 RNN 或相關序列模型時，那使用 Adam 是更為常見的，在專案初期，再次提醒，不要逞英雄，按照原始的論文實作即可)。
- 每次只增加一個複雜度。如果你有多個想要在分類器上嘗試的想法，建議你一個一個的嘗試，確保每次都有達到效能增加，不要在一開始把所有的東西都放進去。還有其他會增加複雜度的方法，例如：一開始先使用比較小的圖片，之後再將其放大等。
- 不要相信預設的學習率衰減方式。如果你將某個程式碼運用在新的領域時，要隨時注意學習率的衰減方式。不僅僅是針對不同的問題應該使用不同學習速率衰減的方式，更要注意的是，在特定的問題上，衰減的方式應該基於目前學習的 epoch 數，而這會基於你目前訓練資料的大小。舉例來說，ImageNet 的學習速率衰減的方式會在第 30 個 epoch 時衰減 10 倍。如果你不是訓練 ImageNet，那最好不要這樣設定。如果你不小心讓學習速率衰減的太快，模型可能沒辦法收斂。以我自己的經驗來說，我在一開始不會使用衰減的學習速率，而是讓學習速率維持一個常數值，在最後才去調整學習速率的衰減程度。

### 4. 正規化

理論上，我們現在會有一個足夠大的模型來擬合訓練資料集。現在，是該放棄一些訓練的準確率，對其做正規化來讓驗證資料集的準確率上升了。底下是一些秘訣：

- 取得更多資料。首先，要正規化一個模型，在實際環境中最好的方法就是搜集更多真實的訓練資料。一個很常見的錯誤是你嘗試透過許多工程的技巧，嘗嘗試從小量的資料中擠出更多的效能，而不是把時間花費在嘗試搜集更多真實的訓練資料上。就我所知，增加更多的資料幾乎是唯一能夠保證提高一個良好配置的神經網路效能的方法。另一種方式是透過 ensemble 的方法 (如果你可以負擔得起使用該方法的話)，但這只有在 ensemble 五個以上的模型時會比較有顯著的效果。
- 資料增量。除了使用真實資料之外，你還可以嘗試混合真假資料 - 試著使用更積極的資料增量的方法。
- 有創意的資料增量方法。如果混合真假資料沒用，你還可以嚐試使用假資料。試著找出一些有創意的方法來建立資料及，比如說，領域隨機、使用模擬資料、甚至是 GAN。
- 預訓練。如果可以的話，即使你有足夠的資料，可以使用預訓練網路。
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