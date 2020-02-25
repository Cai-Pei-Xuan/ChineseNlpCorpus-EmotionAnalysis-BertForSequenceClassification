# -*-coding:UTF-8 -*-
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F     # 激励函数都在这

def to_input_id(sentence_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_input)))

if __name__ == "__main__":

    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    config = BertConfig.from_pretrained('trained_model/config.json')
    model = BertForSequenceClassification.from_pretrained('trained_model/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.eval()

    print("請輸入句子")
    sentence = input()

    input_id = to_input_id(sentence)
    assert len(input_id) <= 512
    input_ids = torch.LongTensor(input_id).unsqueeze(0)

    # predict時，因為沒有label所以沒有loss
    outputs = model(input_ids)

    prediction = torch.max(F.softmax(outputs[0]), dim = 1)[1] # 在第1維度取最大值並返回索引值 
    predict_label = prediction.data.cpu().numpy().squeeze()   # 降維

    if str(predict_label) == "1":
        print("正面")
    else:
        print("負面")