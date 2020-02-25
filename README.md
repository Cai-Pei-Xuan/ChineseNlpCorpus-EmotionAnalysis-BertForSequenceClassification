# ChineseNlpCorpus-EmotionAnalysis-BertForSequenceClassification
使用[外賣評論資料集](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k)去做判斷正負面句子(fine-turing BertForSequenceClassification)

## 檔案說明
### Data
- data_split.py : 將waimai_10k_zh_tw.csv切割成train(6000筆)、test(2000筆)資料的程式(正負面句子都各占一半)
- core.py : 處理dataset和label的轉換
- preprocess_data.py : BertForSequenceClassification的前處理
- train.py : BertForSequenceClassification的訓練
- predict.py : BertForSequenceClassification的預測
- requestment.txt : 紀錄需要安裝的環境
## 使用說明
### train的順序
```
python data_split.py    # 如果已經存在train、test資料，就可以跳過這步驟
python train.py         # 如果想用訓練好的model可以去release下載，並將資料放入trained_model內，就可以跳過這步驟
```
### Demo
```
python predict.py
```
## 環境需求
- python 3.6+
- pytorch 1.3+
- transformers 2.2+
- CUDA Version: 10.0
