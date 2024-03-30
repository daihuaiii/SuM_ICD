# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import BertModel
from utils import weighted_sum, logsumexp
from search import BeamSearch


class AdditiveAttention(nn.Module):
    def __init__(self, vector_dim: int, matrix_dim: int):
        super().__init__()
        self._w_matrix = Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = Parameter(torch.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._w_matrix)
        nn.init.xavier_uniform_(self._u_matrix)
        nn.init.xavier_uniform_(self._v_vector)

    def forward(self, vector, matrix):
        if vector.dim() == 2:
            # (b, 1, v) = (b, q, v)
            w_q = vector.matmul(self._w_matrix).unsqueeze(1)
        # elif vector.dim() == 3:
        else:
            # (b, q, v)
            w_q = vector.matmul(self._w_matrix)
        # (b, k, v)
        w_k = matrix.matmul(self._u_matrix)
        # (b, q, k, v)
        intermediate = w_q.unsqueeze(2) + w_k.unsqueeze(1)
        intermediate = torch.tanh(intermediate)
        # (b, q, k)
        return intermediate.matmul(self._v_vector).squeeze(-1)


        # if vector.dim() == 2:
        #     intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        # elif vector.dim() == 3:
        #     intermediate = vector.matmul(self._w_matrix) + matrix.matmul(self._u_matrix)
        # intermediate = torch.tanh(intermediate)
        # return intermediate.matmul(self._v_vector).squeeze(2)


class Classification_MLP(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()

        self.hidden1 = nn.Linear(input_size, 200)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(200, 100)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(100, 10)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(10, 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.hidden4.weight)

    def forward(self, matrix):
        matrix = self.hidden1(matrix)
        matrix = self.act1(matrix)
        matrix = self.hidden2(matrix)
        matrix = self.act2(matrix)
        matrix = self.hidden3(matrix)
        matrix = self.act3(matrix)
        matrix = self.hidden4(matrix)

        return matrix


class SuM_Model(nn.Module):

    def __init__(self, args):
        super(SuM_Model, self).__init__()
        self.args = args
        self.lr = args.lr
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.embedding_dim = args.emb_dim
        self.target_embedding_dim = self.embedding_dim
        self.dropout_rate = 1.0 - args.dropout_keep_prob
        self.source1_max_length = args.source1_max_length
        self.source2_max_length = args.source2_max_length
        self.target_max_length = args.target_max_length
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        self.flag_use_layernorm = args.flag_use_layernorm
        self.flag_use_dropout = args.flag_use_dropout
        self.encoder_output_dim = args.encoder_output_dim
        self.cated_encoder_out_dim = self.encoder_output_dim + self.encoder_output_dim
        self.decoder_output_dim = args.decoder_output_dim
        self.decoder_input_dim = self.cated_encoder_out_dim + self.target_embedding_dim
        self.flag_pool = args.flag_pool
        self.flag_inter_attention = args.flag_inter_attention

        self.pretrained_embedding = BertModel.from_pretrained('the model name or file path')
        for param in self.pretrained_embedding.parameters():
            param.requires_grad_(False)

        self.source1_encoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                       num_layers=1, batch_first=True, bidirectional=False)
        self.source2_encoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                       num_layers=1, batch_first=True, bidirectional=False)
        self.source1_attention_layer = AdditiveAttention(self.hidden_dim, self.encoder_output_dim)
        self.source2_attention_layer = AdditiveAttention(self.hidden_dim, self.encoder_output_dim)
        self.source1_dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.source2_dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.encoder_out_projection_layer = nn.Linear(in_features=self.cated_encoder_out_dim,
                                                      out_features=self.decoder_output_dim)
        self.gate_projection_layer = torch.nn.Linear(in_features=self.decoder_output_dim + self.decoder_input_dim,
                                                     out_features=1, bias=False)
        self.decoder_cell = nn.modules.LSTMCell(input_size=self.decoder_input_dim,
                                                hidden_size=self.decoder_output_dim, bias=True)
        self.beam_search = BeamSearch(end_index=2, max_steps=self.target_max_length, beam_size=self.args.beam_size)

        self.summarization_encoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                             num_layers=1, batch_first=True, bidirectional=True)
        self.label_encoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                     num_layers=1, batch_first=True, bidirectional=True)
        self.summarization_dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.label_dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.max_pooling_layer = nn.MaxPool1d(4, stride=2)
        self.mean_pooling_layer = nn.AvgPool1d(4, stride=2)
        self.inner_attention_layer_summa = AdditiveAttention(self.hidden_dim * 2, self.hidden_dim * 2)
        self.inner_attention_layer_label = AdditiveAttention(self.hidden_dim * 2, self.hidden_dim * 2)
        self.inner_attention_layer_pool_summa = AdditiveAttention(self.hidden_dim - 1, self.hidden_dim*2)
        self.inner_attention_layer_pool_label = AdditiveAttention(self.hidden_dim - 1, self.hidden_dim*2)
        self.inter_attention_layer = AdditiveAttention(self.hidden_dim * 2, self.hidden_dim * 2)
        self.inter_attention_layer2 = AdditiveAttention(self.hidden_dim * 2, self.hidden_dim * 2)
        self.aggregation_layer1 = nn.LSTM(self.hidden_dim * 4, self.hidden_dim,
                                          num_layers=1, batch_first=True, bidirectional=True)
        self.aggregation_layer2 = nn.LSTM(self.hidden_dim * 4, self.hidden_dim,
                                          num_layers=1, batch_first=True, bidirectional=True)
        self.aggregation_layer_no_inter1 = nn.LSTM(self.hidden_dim * 2, self.hidden_dim,
                                                   num_layers=1, batch_first=True, bidirectional=True)
        self.aggregation_layer_no_inter2 = nn.LSTM(self.hidden_dim * 2, self.hidden_dim,
                                                   num_layers=1, batch_first=True, bidirectional=True)
        self.summarization_dropout_layer2 = nn.Dropout(p=self.dropout_rate)
        self.label_dropout_layer2 = nn.Dropout(p=self.dropout_rate)
        self.flag_use_cosine = args.flag_use_cosine
        self.classification = Classification_MLP(self.target_max_length * 4 * self.hidden_dim)
        self.loss_function = nn.CrossEntropyLoss()
        self.loss_function_bi = nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        if self.flag_use_layernorm:
            self.source1_encoder_layernorm = nn.LayerNorm(normalized_shape=[self.max_seq_len, self.embedding_dim])
            self.source2_encoder_layernorm = nn.LayerNorm(normalized_shape=[self.max_seq_len, self.embedding_dim])
            self.decoder_hidden_layernorm = nn.LayerNorm(normalized_shape=self.decoder_output_dim)
            self.decoder_cell_layernorm = nn.LayerNorm(normalized_shape=self.decoder_output_dim)


    def embedding_roberta(self, input, att, types):
        out = self.pretrained_embedding(input_ids=input,
                                        attention_mask=att,
                                        token_type_ids=types)

        return out.last_hidden_state


    def encode(self, batch_input):

        input1 = batch_input['source_l_input_ids']
        att1 = batch_input['source_l_attention_mask']
        type1 = batch_input['source_l_token_type_ids']
        input2 = batch_input['source_s_input_ids']
        att2 = batch_input['source_s_attention_mask']
        type2 = batch_input['source_s_token_type_ids']

        source1_embedding = self.embedding_roberta(input1, att1, type1)
        source2_embedding = self.embedding_roberta(input2, att2, type2)
        
        source1_encoder_output, source1_last_state = self.source1_encoder(source1_embedding)
        source2_encoder_output, source2_last_state = self.source2_encoder(source2_embedding)

        source1_hidden_state = source1_last_state[0]
        source2_hidden_state = source2_last_state[0]

        if self.flag_use_layernorm:
            source1_encoder_output = self.source1_encoder_layernorm(source1_encoder_output)
            source2_encoder_output = self.source2_encoder_layernorm(source2_encoder_output)

        if self.flag_use_dropout:
            source1_encoder_output = self.source1_dropout_layer(source1_encoder_output)
            source2_encoder_output = self.source2_dropout_layer(source2_encoder_output)

        return source1_encoder_output, source2_encoder_output, source1_hidden_state, source2_hidden_state




    def get_decoder_initial_state(self, batch_input, train=True):
        model_state = {}  # dict

        source1_encoder_output, source2_encoder_output, source1_hidden_state, source2_hidden_state = self.encode(batch_input)

        initial_decoder_hidden = torch.tanh(self.encoder_out_projection_layer(
            torch.cat([source1_hidden_state, source2_hidden_state], dim=-1))).squeeze(0)
        batch_size = source1_encoder_output.shape[0]
        initial_decoder_cell = initial_decoder_hidden.new_zeros(batch_size, self.decoder_output_dim)

        model_state["source1_input_id"] = batch_input["source_l_input_ids"]
        model_state["source2_input_id"] = batch_input["source_s_input_ids"]
        model_state["merged_input1"] = batch_input["merged_input1"]
        model_state["merged_input2"] = batch_input["merged_input2"]
        model_state["source1_encoder_output"] = source1_encoder_output
        model_state["source2_encoder_output"] = source2_encoder_output
        model_state["decoder_hidden_state"] = initial_decoder_hidden
        model_state["decoder_cell_state"] = initial_decoder_cell

        initial_source1_decoder_attention = self.source1_attention_layer(initial_decoder_hidden,
                                                                         source1_encoder_output)
        initial_source2_decoder_attention = self.source2_attention_layer(initial_decoder_hidden,
                                                                         source2_encoder_output)

        initial_source1_decoder_attention_score = torch.softmax(initial_source1_decoder_attention, -1)
        initial_source2_decoder_attention_score = torch.softmax(initial_source2_decoder_attention, -1)

        initial_source1_weighted_context = weighted_sum(source1_encoder_output, initial_source1_decoder_attention_score)
        initial_source2_weighted_context = weighted_sum(source2_encoder_output, initial_source2_decoder_attention_score)

        model_state["source1_weighted_context"] = initial_source1_weighted_context
        model_state["source2_weighted_context"] = initial_source2_weighted_context

        return model_state


    def decode_step(self, previous_token_id, model_state):

        previous_s1_weighted_context = model_state["source1_weighted_context"]
        previous_s2_weighted_context = model_state["source2_weighted_context"]
        previous_hidden_state = model_state["decoder_hidden_state"]
        previous_cell_state = model_state["decoder_cell_state"]

        type_id = torch.ones_like(previous_token_id).long().to(self.device)

        zeros = torch.zeros_like(previous_token_id)
        ones = torch.ones_like(previous_token_id)
        attention_mask = torch.where(previous_token_id == 0, zeros, ones).long().to(self.device)
        previous_token_id = previous_token_id.long().to(self.device)

        previous_token_embedding = self.embedding_roberta(previous_token_id, attention_mask, type_id).squeeze(1)

        current_decoder_input = torch.cat((previous_token_embedding, previous_s1_weighted_context,
                                          previous_s2_weighted_context), dim=-1)

        decoder_hidden_state, decoder_cell_state = self.decoder_cell(current_decoder_input,
                                                                     (previous_hidden_state, previous_cell_state))
        if self.flag_use_layernorm:
            decoder_hidden_state = self.decoder_hidden_layernorm(decoder_hidden_state)
            decoder_cell_state = self.decoder_cell_layernorm(decoder_cell_state)
        model_state["decoder_hidden_state"] = decoder_hidden_state
        model_state["decoder_cell_state"] = decoder_cell_state

        source1_encoder_output = model_state["source1_encoder_output"]
        source2_encoder_output = model_state["source2_encoder_output"]
        source1_attention_output = self.source1_attention_layer(decoder_hidden_state, source1_encoder_output)
        source2_attention_output = self.source2_attention_layer(decoder_hidden_state, source2_encoder_output)
        source1_attention_score = torch.softmax(source1_attention_output, -1)
        source2_attention_score = torch.softmax(source2_attention_output, -1)
        model_state["source1_attention_score"] = source1_attention_score
        model_state["source2_attention_score"] = source2_attention_score

        source1_weighted_context = weighted_sum(source1_encoder_output, source1_attention_output)
        source2_weighted_context = weighted_sum(source2_encoder_output, source2_attention_output)
        model_state["source1_weighted_context"] = source1_weighted_context
        model_state["source2_weighted_context"] = source2_weighted_context

        gate_input = torch.cat((previous_token_embedding, source1_weighted_context, source2_weighted_context,
                                decoder_hidden_state), dim=-1)
        gate_projected = self.gate_projection_layer(gate_input).squeeze(-1)
        gate_score = torch.sigmoid(gate_projected)
        model_state["gate_score"] = gate_score

        return model_state

    def get_predict_id(self, model_state, source1_input_ids, source2_input_ids):
        gate1 = model_state["gate_score"].unsqueeze(-1)
        source1_attention_score = model_state["source1_attention_score"].squeeze(1)
        source2_attention_score = model_state["source2_attention_score"].squeeze(1)

        log_score1 = (source1_attention_score + 1e-20).log()
        log_score2 = (source2_attention_score + 1e-20).log()
        log_gate1 = (gate1 + 1e-20).log()
        log_gate2 = (1 - gate1 + 1e-20).log()

        distribution1 = log_score1 + log_gate1
        distribution2 = log_score2 + log_gate2

        position = torch.argmax(torch.cat((distribution1, distribution2), -1), -1).int()

        source1_length = self.source1_max_length
        token_ids = []

        for i in range(position.shape[0]):
            if position[i] < source1_length:
                real_position = position[i]
                token_id = source1_input_ids[i][real_position]
            else:
                real_position = position[i] - source1_length
                token_id = source2_input_ids[i][real_position]
            token_ids.append(token_id)
        token_ids = torch.Tensor(token_ids).unsqueeze(-1).to(self.device)


        return token_ids

    def get_summarization(self, batch_input):
        target_len = self.target_max_length   # 15

        t_input_ids = batch_input['true_name_input_ids']
        source1_input_ids = batch_input['source_l_input_ids']
        source2_input_ids = batch_input['source_s_input_ids']
        model_state = self.get_decoder_initial_state(batch_input)
        predict_tokens = torch.Tensor([]).to(self.device)

        for step in range(target_len):
            previous_token_id = t_input_ids[:, step].unsqueeze(1)
            model_state = self.decode_step(previous_token_id, model_state)
            predict_step_tokens = self.get_predict_id(model_state, source1_input_ids, source2_input_ids)
            predict_tokens = torch.cat((predict_tokens, predict_step_tokens), dim=1)

        predict_tokens = predict_tokens[:, 1:-1]
        start_token = (torch.ones(self.batch_size, 1) * 101).to(self.device)
        end_token = (torch.ones(self.batch_size, 1) * 102).to(self.device)
        predict_tokens = torch.cat((start_token, predict_tokens, end_token), dim=1).long().cpu()

        non_zeros = []
        for i in range(self.batch_size):
            tokens = predict_tokens[i][~np.isin(predict_tokens[i], torch.Tensor([0]))]
            non_zeros.append(tokens)
        predict_tokens = pad_sequence(non_zeros, batch_first=True, padding_value=0).long().to(self.device)

        if predict_tokens.shape[1] < self.source2_max_length:
            zeros = torch.zeros((self.batch_size, self.source2_max_length - predict_tokens.shape[1])).long().to(self.device)
            predict_tokens = torch.cat((predict_tokens, zeros), dim=-1)


        token_types = torch.zeros_like(predict_tokens).to(self.device)

        zeros = torch.zeros((self.batch_size, 1)).long().to(self.device)
        ones = torch.ones((self.batch_size, 1)).long().to(self.device)
        attention_mask = torch.where(predict_tokens == 0, zeros, ones).to(self.device)

        summa_embedding = self.embedding_roberta(predict_tokens.to(self.device), attention_mask, token_types)
        return summa_embedding

    def merge_final_log_probs(self, source1_attention_score, source2_attention_score, source1_token_ids, source2_token_ids, gate):
        source2_max_len = source2_token_ids.shape[1]
        gate_1 = gate.expand(self.source1_max_length, -1).t()
        gate_2 = (1 - gate).expand(source2_max_len, -1).t()

        source1_attention_score = source1_attention_score.squeeze(1) * gate_1
        source2_attention_score = source2_attention_score.squeeze(1) * gate_2

        log_probs_1 = (source1_attention_score + 1e-45).log()
        log_probs_2 = (source2_attention_score + 1e-45).log()

        batch_size = source1_token_ids.shape[0]
        final_log_probs = (source1_attention_score.new_zeros((batch_size, self.source1_max_length +
                                                              self.source2_max_length)) + 1e-45).log()

        for i in range(self.source1_max_length):
            log_probs_slice = log_probs_1[:, i].unsqueeze(-1).to(self.device)
            source_to_target_slice = source1_token_ids[:, i].unsqueeze(-1).to(self.device)
            selected_log_probs = final_log_probs.gather(-1, source_to_target_slice).to(self.device)
            combined_scores = logsumexp(torch.cat((selected_log_probs, log_probs_slice), dim=-1)).unsqueeze(-1)
            final_log_probs = final_log_probs.scatter(-1, source_to_target_slice, combined_scores)

        for i in range(source2_max_len):
            log_probs_slice = log_probs_2[:, i].unsqueeze(-1)
            source_to_target_slice = source2_token_ids[:, i].unsqueeze(-1).to(self.device)
            selected_log_probs = final_log_probs.gather(-1, source_to_target_slice).to(self.device)
            combined_scores = logsumexp(torch.cat((selected_log_probs, log_probs_slice), dim=-1)).unsqueeze(-1)
            final_log_probs = final_log_probs.scatter(-1, source_to_target_slice, combined_scores)

        return final_log_probs

    def take_search_step(self, previous_token_ids, model_state):
        model_state = self.decode_step(previous_token_ids, model_state)

        final_log_probs = self.merge_final_log_probs(model_state["source1_attention_score"],
                                                                   model_state["source2_attention_score"],
                                                                   model_state["merged_input1"],
                                                                   model_state["merged_input2"],
                                                                   model_state["gate_score"])
        return final_log_probs, model_state

    def forward_beam_search(self, batch_input, model_state, ranking=False):
        if ranking:
            start_token_ids = (torch.ones((1, 1)) * 101).to(self.device)
        else:
            start_token_ids = (torch.ones((self.batch_size, 1)) * 101).to(self.device)
        all_top_k_predictions, _ = self.beam_search.search(start_token_ids, batch_input, model_state,
                                                           self.take_search_step)

        return all_top_k_predictions

    def get_summarization_bs(self, batch_input, ranking=False):
        model_state = self.get_decoder_initial_state(batch_input, train=False)
        predicted_tokens = self.forward_beam_search(batch_input, model_state, ranking)
        predicted_tokens = predicted_tokens[:, 0, :].long().to(self.device)
        merged_index = batch_input['merged_index'].to(self.device)
        predicted_tokens = merged_index.gather(-1, predicted_tokens).to(self.device)
        if ranking:
            ones = (torch.ones((1, 1)) * 101).long().to(self.device)
            ones2 = (torch.ones((1, 1)) * 102).long().to(self.device)

        else:
            ones = (torch.ones((self.batch_size, 1)) * 101).long().to(self.device)
            ones2 = (torch.ones((self.batch_size, 1)) * 102).long().to(self.device)
        predicted_tokens = torch.cat((ones, predicted_tokens, ones2), dim=-1).cpu()

        non_zeros = []
        for i in range(predicted_tokens.shape[0]):
            tokens = predicted_tokens[i][~np.isin(predicted_tokens[i], torch.Tensor([0]))]
            non_zeros.append(tokens)

        predicted_tokens = pad_sequence(non_zeros, batch_first=True, padding_value=0).to(self.device)

        if predicted_tokens.shape[1] < self.source2_max_length:
            zeros = torch.zeros((predicted_tokens.shape[0], self.source2_max_length - predicted_tokens.shape[1])).long().to(self.device)
            predicted_tokens = torch.cat((predicted_tokens, zeros), dim=-1).to(self.device)

        token_types = torch.zeros_like(predicted_tokens).to(self.device)  # .new_ones((self.batch_size, self.source2_max_length))

        zeros = torch.zeros((predicted_tokens.shape[0], 1)).long().to(self.device)
        ones = torch.ones((predicted_tokens.shape[0], 1)).long().to(self.device)
        attention_mask = torch.where(predicted_tokens == 0, zeros, ones).to(self.device)

        summa_embedding = self.embedding_roberta(predicted_tokens.to(self.device), attention_mask, token_types)

        return summa_embedding

    def get_matching_result_train(self, batch_input):
        summa_embedding = self.get_summarization(batch_input)
        label_embedding = self.embedding_roberta(batch_input["target_input_ids"].to(self.device),
                                                 batch_input["target_attention_mask"].to(self.device),
                                                 batch_input["target_token_type_ids"].to(self.device))

        summa_encoder_output, _ = self.summarization_encoder(summa_embedding)
        label_encoder_output, _ = self.label_encoder(label_embedding)

        if self.flag_use_dropout:
            summa_encoder_output = self.summarization_dropout_layer(summa_encoder_output)
            label_encoder_output = self.label_dropout_layer(label_encoder_output)

        if self.flag_pool == 1:
            summa_after_pooling = self.max_pooling_layer(summa_encoder_output)
            label_after_pooling = self.max_pooling_layer(label_encoder_output)
            summa_inner_weight = self.inner_attention_layer_pool_summa(summa_after_pooling, summa_encoder_output)
            label_inner_weight = self.inner_attention_layer_pool_label(label_after_pooling, label_encoder_output)
        elif self.flag_pool == 2:
            summa_after_pooling = self.mean_pooling_layer(summa_encoder_output)
            label_after_pooling = self.mean_pooling_layer(label_encoder_output)
            summa_inner_weight = self.inner_attention_layer_pool_summa(summa_after_pooling, summa_encoder_output)
            label_inner_weight = self.inner_attention_layer_pool_label(label_after_pooling, label_encoder_output)
        else:
            summa_inner_weight = self.inner_attention_layer_summa(summa_encoder_output, summa_encoder_output)
            label_inner_weight = self.inner_attention_layer_label(label_encoder_output, label_encoder_output)

        summa_inner_attention_score = torch.softmax(summa_inner_weight, -1)
        label_inner_attention_score = torch.softmax(label_inner_weight, -1)

        summa_after_inner = weighted_sum(summa_encoder_output, summa_inner_attention_score)
        label_after_inner = weighted_sum(label_encoder_output, label_inner_attention_score)

        summa_inter_weight = self.inter_attention_layer(label_after_inner, summa_after_inner)
        label_inter_weight = self.inter_attention_layer2(summa_after_inner, label_after_inner)

        summa_inter_attention_score = torch.softmax(summa_inter_weight, -1)
        label_inter_attention_score = torch.softmax(label_inter_weight, -1)

        summa_after_inter = weighted_sum(summa_after_inner, summa_inter_attention_score)
        label_after_inter = weighted_sum(label_after_inner, label_inter_attention_score)

        summa_concat = torch.cat((summa_after_inner, summa_after_inter), dim=-1)
        label_concat = torch.cat((label_after_inner, label_after_inter), dim=-1)

        summa_aggregated, _ = self.aggregation_layer1(summa_concat)
        label_aggregated, _ = self.aggregation_layer2(label_concat)

        if self.flag_use_dropout:
            summa_aggregated = self.summarization_dropout_layer2(summa_aggregated)
            label_aggregated = self.label_dropout_layer2(label_aggregated)

        if self.flag_use_cosine == 1:

            summa_flatten = torch.flatten(summa_aggregated, start_dim=1)
            label_flatten = torch.flatten(label_aggregated, start_dim=1)
            matching_score = F.cosine_similarity(summa_flatten, label_flatten)

        elif self.flag_use_cosine == 2:
            summa_flatten = torch.flatten(summa_aggregated, start_dim=1)
            label_flatten = torch.flatten(label_aggregated, start_dim=1)
            matching_score = torch.exp(1 - nn.PairwiseDistance(p=2)(summa_flatten, label_flatten)) *1000

        else:
            all_concat = torch.cat((summa_aggregated, label_aggregated), dim=-1)
            all_concat = torch.flatten(all_concat, start_dim=1)
            matching_score = self.classification(all_concat)

        return matching_score

    def get_matching_result_eval(self, batch_input, ranking=False):
        summa_embedding = self.get_summarization_bs(batch_input, ranking)

        label_embedding = self.embedding_roberta(batch_input["target_input_ids"].to(self.device),
                                                 batch_input["target_attention_mask"].to(self.device),
                                                 batch_input["target_token_type_ids"].to(self.device))

        summa_encoder_output, _ = self.summarization_encoder(summa_embedding)
        label_encoder_output, _ = self.label_encoder(label_embedding)

        if self.flag_use_dropout:
            summa_encoder_output = self.summarization_dropout_layer(summa_encoder_output)
            label_encoder_output = self.label_dropout_layer(label_encoder_output)
        if self.flag_pool == 1:
            summa_after_pooling = self.max_pooling_layer(summa_encoder_output)
            label_after_pooling = self.max_pooling_layer(label_encoder_output)
            summa_inner_weight = self.inner_attention_layer_pool_summa(summa_after_pooling, summa_encoder_output)
            label_inner_weight = self.inner_attention_layer_pool_label(label_after_pooling, label_encoder_output)
        elif self.flag_pool == 2:
            summa_after_pooling = self.mean_pooling_layer(summa_encoder_output)
            label_after_pooling = self.mean_pooling_layer(label_encoder_output)
            summa_inner_weight = self.inner_attention_layer_pool_summa(summa_after_pooling, summa_encoder_output)
            label_inner_weight = self.inner_attention_layer_pool_label(label_after_pooling, label_encoder_output)
        else:
            summa_inner_weight = self.inner_attention_layer_summa(summa_encoder_output, summa_encoder_output)
            label_inner_weight = self.inner_attention_layer_label(label_encoder_output, label_encoder_output)

        summa_inner_attention_score = torch.softmax(summa_inner_weight, -1)
        label_inner_attention_score = torch.softmax(label_inner_weight, -1)

        summa_after_inner = weighted_sum(summa_encoder_output, summa_inner_attention_score)
        label_after_inner = weighted_sum(label_encoder_output, label_inner_attention_score)

        summa_inter_weight = self.inter_attention_layer(label_after_inner, summa_after_inner)
        label_inter_weight = self.inter_attention_layer2(summa_after_inner, label_after_inner)

        summa_inter_attention_score = torch.softmax(summa_inter_weight, -1)
        label_inter_attention_score = torch.softmax(label_inter_weight, -1)

        summa_after_inter = weighted_sum(summa_after_inner, summa_inter_attention_score)
        label_after_inter = weighted_sum(label_after_inner, label_inter_attention_score)

        label_concat = torch.cat((label_after_inner, label_after_inter), dim=-1)
        summa_concat = torch.cat((summa_after_inner, summa_after_inter), dim=-1)

        summa_aggregated, _ = self.aggregation_layer1(summa_concat)
        label_aggregated, _ = self.aggregation_layer2(label_concat)

        if self.flag_use_dropout:
            summa_aggregated = self.summarization_dropout_layer2(summa_aggregated)
            label_aggregated = self.label_dropout_layer2(label_aggregated)


        if self.flag_use_cosine == 1:
            summa_flatten = torch.flatten(summa_aggregated, start_dim=1)
            label_flatten = torch.flatten(label_aggregated, start_dim=1)
            matching_score = F.cosine_similarity(summa_flatten, label_flatten)

        elif self.flag_use_cosine == 2:
            summa_flatten = torch.flatten(summa_aggregated, start_dim=1)
            label_flatten = torch.flatten(label_aggregated, start_dim=1)
            matching_score = torch.exp(1 - nn.PairwiseDistance(p=2)(summa_flatten, label_flatten)) * 1000

        else:
            all_concat = torch.cat((summa_aggregated, label_aggregated), dim=-1)
            all_concat = torch.flatten(all_concat, start_dim=1)
            matching_score = self.classification(all_concat)

        return matching_score

    def normalize_zero_one(self, matching_score):
        x = torch.zeros_like(matching_score).to(self.device)
        y = torch.ones_like(matching_score).to(self.device)
        result = torch.where(matching_score <= 0, x, y).to(self.device)
        return result

    def get_batch_loss(self, batch_input, train=True):
        real_tag = batch_input['tag'].to(self.device)
        if train:
            self.train()
            matching_score = self.get_matching_result_train(batch_input)
        else:
            self.eval()
            matching_score = self.get_matching_result_eval(batch_input)

        if self.flag_use_cosine != 0:
            matching_score = matching_score
            matching_after_sigmoid = self.sigmoid(matching_score)
            batch_loss = self.loss_function_bi(matching_after_sigmoid, real_tag.float().to(self.device))

            matching_result = self.normalize_zero_one(matching_score)

            accuracy = torch.true_divide((self.batch_size - torch.sum(torch.abs(matching_result - real_tag))),
                                         self.batch_size)

        else:
            batch_loss = self.loss_function(matching_score, real_tag.to(self.device))

            matching_after_softmax = self.softmax(matching_score)
            matching_result = torch.argmax(matching_after_softmax, dim=1)
            accuracy = torch.true_divide((self.batch_size - torch.sum(torch.abs(matching_result-real_tag))), self.batch_size)

        return batch_loss, matching_result, accuracy.item()


    def get_batch_matching_result(self, batch_input):
        self.eval()
        if self.flag_use_cosine != 0:
            matching_result = self.get_matching_result_eval(batch_input, ranking=True)
            matching_result = matching_result
        else:
            matching_score = self.get_matching_result_eval(batch_input, ranking=True)
            matching_after_softmax = self.softmax(matching_score)
            matching_result = matching_after_softmax[:, 1]

        return matching_result



if __name__ == "__main__":
    print(torch.cuda.is_available())



















