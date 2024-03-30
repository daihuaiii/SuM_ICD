# -*- coding:utf-8 -*-

from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformers import logging
logging.set_verbosity_error()


def single_data_encode(data, max_len):
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='model name or file path',
        cache_dir=None,
        force_download=False,
    )

    if max_len == 0:
        token_out = tokenizer.encode_plus(
            text=data,
            add_special_tokens=True,
            return_tensors='pt',
        )
    else:
        token_out = tokenizer.encode_plus(
            text=data,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            add_special_tokens=True,
            return_tensors='pt',
        )
    return token_out


def batch_data_encode(file, batch_size, train):
    df = pd.read_csv(file)
    data = df['data']
    key = df['key']
    name = df['name']
    tag = df['tag']

    empty = []
    input1 = torch.Tensor(empty).long()
    att1 = torch.Tensor(empty).long()
    type1 = torch.Tensor(empty).long()
    input2 = torch.Tensor(empty).long()
    att2 = torch.Tensor(empty).long()
    type2 = torch.Tensor(empty).long()

    input3 = torch.Tensor(empty).long()
    att3 = torch.Tensor(empty).long()
    type3 = torch.Tensor(empty).long()
    target_len = []

    if train:
        true_name = df['true_name']
        input4 = torch.Tensor(empty).long()

    tag_batch = []

    merged1 = torch.Tensor(empty).long()
    merged2 = torch.Tensor(empty).long()
    merged_index = []


    for i in range(len(data)):
        d0 = data[i]
        m0 = d0.replace(' ', '')
        token_out1 = single_data_encode(m0, max_len=512)

        input1 = torch.cat((input1, token_out1['input_ids']), 0)
        att1 = torch.cat((att1, token_out1['attention_mask']), 0)
        type1 = torch.cat((type1, token_out1['token_type_ids']), 0)

        if train:
            tn = true_name[i]
            token_out4 = single_data_encode(tn, max_len=10)
            input4 = torch.cat((input4, token_out4['input_ids']), 0)


        k0 = key[i]
        token_out2 = single_data_encode(k0, max_len=10)

        input2 = torch.cat((input2, token_out2['input_ids']), 0)
        att2 = torch.cat((att2, token_out2['attention_mask']), 0)
        type2 = torch.cat((type2, token_out2['token_type_ids']), 0)

        t0 = name[i]
        token_out3 = single_data_encode(t0, max_len=10)
        input3 = torch.cat((input3, token_out3['input_ids']), 0)
        att3 = torch.cat((att3, token_out3['attention_mask']), 0)
        type3 = torch.cat((type3, token_out3['token_type_ids']), 0)
        t_len = len(t0)
        target_len.append(t_len)
        tag_batch.append(tag[i])

        merged_input = torch.cat((token_out1['input_ids'], token_out2['input_ids']), dim=-1)
        merged_input = torch.unique(merged_input, return_inverse=True)
        index = merged_input[0]

        merged_input1 = merged_input[1][:, :512]
        merged_input2 = merged_input[1][:, 512:]

        merged1 = torch.cat((merged1, merged_input1), 0)
        merged2 = torch.cat((merged2, merged_input2), 0)
        merged_index.append(index)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if (i + 1) % batch_size == 0:
            data_dict = {}
            data_dict['source_l_input_ids'] = input1.to(device)
            data_dict['source_l_attention_mask'] = att1.to(device)
            data_dict['source_l_token_type_ids'] = type1.to(device)

            data_dict['source_s_input_ids'] = input2.to(device)
            data_dict['source_s_attention_mask'] = att2.to(device)
            data_dict['source_s_token_type_ids'] = type2.to(device)

            data_dict['target_input_ids'] = input3.to(device)
            data_dict['target_attention_mask'] = att3.to(device)
            data_dict['target_token_type_ids'] = type3.to(device)
            data_dict['target_length'] = target_len

            tag_batch = torch.Tensor(tag_batch).long()
            data_dict['tag'] = tag_batch

            data_dict['merged_input1'] = merged1.to(device)
            data_dict['merged_input2'] = merged2.to(device)

            merged_index = pad_sequence(merged_index, batch_first=True, padding_value=0)
            data_dict['merged_index'] = merged_index.to(device)

            if train:
                data_dict['true_name_input_ids'] = input4

            empty = []
            target_len = []
            tag_batch = []
            merged_index = []
            input1 = torch.Tensor(empty).long()
            att1 = torch.Tensor(empty).long()
            type1 = torch.Tensor(empty).long()
            input2 = torch.Tensor(empty).long()
            att2 = torch.Tensor(empty).long()
            type2 = torch.Tensor(empty).long()
            input3 = torch.Tensor(empty).long()
            att3 = torch.Tensor(empty).long()
            type3 = torch.Tensor(empty).long()
            merged1 = torch.Tensor(empty).long()
            merged2 = torch.Tensor(empty).long()

            if train:
                input4 = torch.Tensor(empty).long()

            yield data_dict

def get_single_token(main_diag, full_text, label):
    full_text = full_text.replace(' ', '')
    token_out1 = single_data_encode(full_text, max_len=512)
    token_out2 = single_data_encode(main_diag, max_len=10)
    token_out3 = single_data_encode(label, max_len=10)

    merged_input = torch.cat((token_out1['input_ids'], token_out2['input_ids']), dim=-1)
    merged_input = torch.unique(merged_input, return_inverse=True)

    index = merged_input[0].unsqueeze(0)

    merged_input1 = merged_input[1][:, :512]
    merged_input2 = merged_input[1][:, 512:]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_dict = {}
    data_dict['source_l_input_ids'] = token_out1['input_ids'].to(device)
    data_dict['source_l_attention_mask'] = token_out1['attention_mask'].to(device)
    data_dict['source_l_token_type_ids'] = token_out1['token_type_ids'].to(device)

    data_dict['source_s_input_ids'] = token_out2['input_ids'].to(device)
    data_dict['source_s_attention_mask'] = token_out2['attention_mask'].to(device)
    data_dict['source_s_token_type_ids'] = token_out2['token_type_ids'].to(device)

    data_dict['target_input_ids'] = token_out3['input_ids'].to(device)
    data_dict['target_attention_mask'] = token_out3['attention_mask'].to(device)
    data_dict['target_token_type_ids'] = token_out3['token_type_ids'].to(device)

    data_dict['merged_input1'] = merged_input1.to(device)
    data_dict['merged_input2'] = merged_input2.to(device)

    data_dict['merged_index'] = index.to(device)

    return data_dict


def get_single_token_seperated(main_diag, full_text, label):
    token_out1 = single_data_encode(full_text)
    token_out2 = single_data_encode(main_diag)
    token_out3 = single_data_encode(label)

    merged_input = torch.cat((token_out1['input_ids'], token_out2['input_ids']), dim=-1)

    merged_input = torch.unique(merged_input, return_inverse=True)

    index = merged_input[0].unsqueeze(0)

    merged_input1 = merged_input[1][:, :512]
    merged_input2 = merged_input[1][:, 512:]

    data_dict = {}

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_dict['source_l_input_ids'] = token_out1['input_ids'].to(device)
    data_dict['source_l_attention_mask'] = token_out1['attention_mask'].to(device)
    data_dict['source_l_token_type_ids'] = token_out1['token_type_ids'].to(device)

    data_dict['source_s_input_ids'] = token_out2['input_ids'].to(device)
    data_dict['source_s_attention_mask'] = token_out2['attention_mask'].to(device)
    data_dict['source_s_token_type_ids'] = token_out2['token_type_ids'].to(device)

    data_dict['target_input_ids'] = token_out3['input_ids'].to(device)
    data_dict['target_attention_mask'] = token_out3['attention_mask'].to(device)
    data_dict['target_token_type_ids'] = token_out3['token_type_ids'].to(device)

    data_dict['merged_input1'] = merged_input1.to(device)
    data_dict['merged_input2'] = merged_input2.to(device)

    data_dict['merged_index'] = index.to(device)

    return data_dict


if __name__ == "__main__":
    print(torch.cuda.is_available())