# How to solve 90% of NLP problems: a step-by-step guide 讀後筆記

## 蒐集並觀察資料

文字資料的來源可能包含以下幾種：

產品的評論(例如：Amazon 或其他產品網站的評論)
使用者產生的文字(例如：Tweet、Facebook 的文字)
其他問題處理的內文(例如：客戶請求單、聊天的 log)

這篇文章用了一個 CrowdFlower 提供的 「Disasters on Social Media」資料集。

## 資料清理

- 移除已知不相關的字元(例如：非英文字元)
- 進行 Tokenizer
- 進行大小寫轉換(例如：全部轉換為小寫)
- 處理可能的錯字或將同義字轉換為同一個字(例如：cool/kewl/cooool 可以都轉換為cool)
- 處理 stemming(把字尾去掉)和 lemmatization(把字轉換為原型，例如：，例如：am/are/is轉換為 be)

## 建立簡單的模型

## 觀察模型的結果

