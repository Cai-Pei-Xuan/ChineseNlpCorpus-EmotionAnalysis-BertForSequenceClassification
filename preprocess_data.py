# -*-coding:UTF-8 -*-
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset

def convert_data_to_feature(FileName):
    # 載入字典
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')

    # 載入資料
    Labels = []
    Sentences = []
    with open(FileName,'r',encoding='utf-8') as f:
        data = f.read()
    LS_pairs = data.split("\n")

    for LS_pair in LS_pairs:
        if LS_pair != "":
            try:
                L = LS_pair[:1]
                S = LS_pair[2:]
                Labels.append(int(L))
                Sentences.append(S)
            except:
                continue
    
    assert len(Labels) == len(Sentences)

    # BERT input embedding
    max_seq_len = 0     # 紀錄最大長度
    input_ids = []
    original_length = []    # 紀錄原本長度
    for S in Sentences:
        # 將句子切割成一個個token
        word_piece_list = tokenizer.tokenize(S)
        # 將token轉成字典中的id
        input_id = tokenizer.convert_tokens_to_ids(word_piece_list)
        # 補上[CLS]和[SEP]
        input_id = tokenizer.build_inputs_with_special_tokens(input_id)

        if(len(input_id)>max_seq_len):
            max_seq_len = len(input_id)
        input_ids.append(input_id)

    print("最長句子長度:",max_seq_len)
    assert max_seq_len <= 512 # 小於BERT-base長度限制

    # 補齊長度
    for c in input_ids:
        # 紀錄原本長度
        length = len(c)
        original_length.append(length)
        while len(c)<max_seq_len:
            c.append(0)
    
    segment_ids = [[0]*max_seq_len for i in range(len(Sentences))]         # token_type_ids # segment_ids存儲的是句子的id，id為0就是第一句，id為1就是第二句
    position_ids = []                                                      # attention_mask # position_ids:1代表是真實的單詞id，0代表補全位
    for i in range(len(Sentences)):
        position_id = []
        for j in range(original_length[i]):
            position_id.append(1)
        while len(position_id)<max_seq_len:
            position_id.append(0)
        position_ids.append(position_id)

    assert len(input_ids) == len(segment_ids) and len(input_ids) == len(position_ids) and len(input_ids) == len(Labels)

    data_features = {'input_ids':input_ids,
                    'segment_ids':segment_ids,
                    'position_ids':position_ids,
                    'labels':Labels}

    return data_features

def makeDataset(data_feature):
    input_ids = data_feature['input_ids']
    segment_ids = data_feature['segment_ids']
    position_ids = data_feature['position_ids']
    labels = data_feature['labels']

    all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
    all_segment_ids = torch.tensor([segment_id for segment_id in segment_ids], dtype=torch.long)
    all_position_ids = torch.tensor([position_id for position_id in position_ids], dtype=torch.long)
    all_labels = torch.tensor([label for label in labels], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_segment_ids, all_position_ids, all_labels)

    return dataset

if __name__ == "__main__":
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    data_features = convert_data_to_feature("train_data.txt")
    print(data_features['input_ids'][5999])
    print(tokenizer.convert_ids_to_tokens(data_features['input_ids'][5999]))
    print(data_features['segment_ids'][5999])
    print(data_features['position_ids'][5999])
    print(data_features['labels'][5999])
