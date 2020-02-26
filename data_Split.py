# -*-coding:UTF-8 -*-

# 將要訓練的句子存起來
def SaveSentence(filepath, sent_list):
    f = open(filepath, 'w', encoding='UTF-8')
    for sent in sent_list:
        f.write(sent + '\n')
    f.close()

# 分割train_data和test_data
def data_Split(FileName):

    fp = open(FileName, 'r', encoding='utf-8')
    line = fp.readline()        # 第一行是label,review
    line = fp.readline()

    train_sent_num = 3000
    test_sent_num = 1000
    train_positive_num = 0
    train_negative_num = 0
    test_positive_num = 0
    test_negative_num = 0
    train_data = []
    test_data = []

    # 用 while 逐行讀取檔案內容，直至檔案結尾
    while line:
        sent = ''
        sent = line.replace('\n', '')

        if line[:2] == '1,':
            if train_positive_num < train_sent_num:
                train_data.append(sent)
                train_positive_num += 1
            elif test_positive_num < test_sent_num:
                test_data.append(sent)
                test_positive_num += 1
        else:
            if train_negative_num < train_sent_num:
                train_data.append(sent)
                train_negative_num += 1
            elif test_negative_num < test_sent_num:
                test_data.append(sent)
                test_negative_num += 1
        
        line = fp.readline()
    
    fp.close()

    SaveSentence('train_data.txt', train_data)
    SaveSentence('test_data.txt', test_data)

if __name__ == "__main__":
    data_Split('waimai_10k_zh_tw.csv')
