## [Advice For New and Junior Data Scientists](https://medium.com/@rchang/advice-for-new-and-junior-data-scientists-2ab02396cf5b) 閱讀筆記

Robert Chang 是目前在 Airbnb 工作的資深資料科學家(senior data scientist)，以下是我閱讀他所撰寫的 [Advice For New and Junior Data Scientists](https://medium.com/@rchang/advice-for-new-and-junior-data-scientists-2ab02396cf5b) 的筆記

這篇文章是寫給那些已經進入資料科學相關工作，但還只是剛剛起步的人

### Whose Critical Path Are You On?

- 將個人目標與公司目標綁在一起，利用公司資源成長
- 作者在 Airbnb 的目標明確，要加入一個機器學習會起重大影響的專案或團隊中
- 最有效的學習方式是投身解決具體商業問題

### Picking the Right Tools For The Problem

- 與其困擾在選擇哪種語言，不如想想哪種語言可以提供最好的 Domain Language，進而解決你的問題
- 作者原本寫 R，對於後來是否要轉到 Python，給了以下的範例
    - 如果你是要使用最新的統計方法，R 可能是比較好的選擇，因為 R 是統計學家設計出來的，許多研究者在發表論文時，也會提供 R library 來實驗他們的想法
    - 如果你是要做 production 的 data pipeline 或系統，Python 可能就是比較通用的選擇

### Building A Learning Project

- 進行刻意訓練
- 確定學習大綱，並且由相關領域專家來檢驗這份綱要
    - 作者也提供了學習他在學習 Python 的[大綱](https://github.com/robert8138/python-deliberate-practice)
- 建立大綱後，依照以下步驟進行實作
    - Practice Repeatedly：反覆練習，熟悉相關語法及套件
    - Create Feedback Loop：找機會看看別人的程式碼，嘗試重構或修改小的 bug
    - Learn By Chunking and Recalling：寫下每週紀錄，用來回憶與消化

### Partnering With Experienced Data Scientists

- 成功的人不會羞愧於自己在某方面的無知，而是經常地尋求他人的回饋
- 作者經由揭露自己在「如何將機器學習運用於正式上線環境中」了解甚少，得到了相當寶貴的回饋

### Teaching And Evangelizing

- 教學相長，透過傳授知識的過程，確保自己了解知識的核心概念
- 教學是檢驗自己對知識的理解程度，學到有價值的東西後，記得與他人分享，不必只想著要創造新的軟體，解釋目前工具如何運作也是很有價值

### At Step K, Think About Your Step K+1

- 對於現狀永遠保持不滿足，覺得事情可以更好，可以得到改善，並思考對應的方法

### Parting Thoughts: You And Your Work

- 找到什麼問題對你是最重要的，你就會想盡辦法去解決

### 參考連結

文章中有很多具有價值的參考連結，條列如下：

- [Time series cross validation](https://robjhyndman.com/hyndsight/tscv/)
- [Scikit-Learn Pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [DataEngConf 2017 - Machine Learning Models in Production](https://www.slideshare.net/SharathRao6/lessons-from-integrating-machine-learning-models-into-data-products)
- [7 Important Model Evaluation Error Metrics Everyone should know](https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/)
- [作者練習 Python 的大綱](https://github.com/robert8138/python-deliberate-practice)