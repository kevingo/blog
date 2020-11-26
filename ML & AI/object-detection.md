## RCNN
- selective search + CNN + SVM

Selective Search
- 目的：建立 region proposal
- 方法：透過不同的 windows size 擷取 region proposals。輸出的時候要把 region proposals 轉成相同的長寬

CNN + SVM
- 目的：CNN 是拿 pre-trained 好的 network 來用 (e.g. AlexNet)。CNN 的最後一層會透果 SVM 來判斷 output 是不是一個 object

缺點
- 慢：每一個 image 的每一個 region proposal 都要經過一次 CNN 的 forward pass
- 訓練複雜且困難：要分別 train 三個 model。CNN 用來提取 image features、SVM 的 classifer 用來預測類別、regression model 用來還原 bounding boxes 的 size

## YOLO (https://read01.com/zh-tw/NLoEyN.html#.WuqxzNOFONY)

- YOLO 在訓練與使用階段皆是針對整張影像，所以對於背景的偵測效果有較佳的結果，其背景錯誤偵測率僅有Fast R-CNN 的一半

流程
1. 將 image 切成 SxS 個 grids
2. 如果一個 object 落在某個 grid 中，就由此 grid 。

## SSD