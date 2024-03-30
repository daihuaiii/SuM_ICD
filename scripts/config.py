# -*- coding:utf-8 -*-

import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-random_seed", "--random_seed", type=int, default=1000, help='random seed.')
    parser.add_argument("-numpy_random_seed", "--numpy_random_seed", type=int, default=1001, help='numpy random seed.')
    parser.add_argument("-torch_random_seed", "--torch_random_seed", type=int, default=1002, help='torch random seed.')

    group = parser.add_argument_group("path")
    group.add_argument("-train_data", "--train_data", type=str, default="", help='train data path.')
    group.add_argument("-val_data", "--val_data", type=str, default="", help='validation data path.')
    group.add_argument("-test_data", "--test_data", type=str, default="", help='test data path.')
    group.add_argument("-result_dir", "--result_dir", type=str, default="", help='result directory path.')
    group.add_argument("-model_dir", "--model_dir", type=str, default="", help='model directory path.')
    group.add_argument("-library_dir", "--library_dir", type=str, default="", help='library directory path.')

    group = parser.add_argument_group("data")
    group.add_argument("-batch_size", "--batch_size", type=int, default=16, help='the size of batch.')
    group.add_argument("-beam_size", "--beam_size", type=int, default=6, help='the size of batch.')
    group.add_argument("-source1_max_length", "--source1_max_length", type=int, default=1024, help='max length of source1.')
    group.add_argument("-source2_max_length", "--source2_max_length", type=int, default=20, help='max length of source2.')
    group.add_argument("-target_max_length", "--target_max_length", type=int, default=20, help='max length of target.')

    group = parser.add_argument_group("model")
    group.add_argument("-lr", "--lr", type=float, default=1e-3, help='learning rate.')
    group.add_argument("-emb_dim", "--emb_dim", type=int, default=256, help='Dimension of embeddings.')
    group.add_argument("-target_embedding_dim", "--target_embedding_dim", type=int, default=128, help='Dimension of target embeddings.')
    group.add_argument("-hidden_dim", "--hidden_dim", type=int, default=128, help='Dimension of hidden nodes.')
    group.add_argument("-encoder_output_dim", "--encoder_output_dim", type=int, default=128, help='Dimension of encoder output.')
    group.add_argument("-decoder_output_dim", "--decoder_output_dim", type=int, default=128, help='Dimension of decoder output.')
    group.add_argument("-flag_use_layernorm", "--flag_use_layernorm", type=bool, default=True, help='layernorm.')
    group.add_argument("-flag_use_dropout", "--flag_use_dropout", type=bool, default=True, help='dropout.')

    group.add_argument("-flag_pool", "--flag_pool", type=int, default=2, help='0: none ; 1: max ; 2: mean.')
    group.add_argument("-flag_inter_attention", "--flag_inter_attention", type=int, default=0, help='0: Addictive. 1: Cosine')
    group.add_argument("-flag_use_cosine", "--flag_use_cosine", type=int, default=0, help='1Cosine 2 Manhattam 0Else MLP')

    group = parser.add_argument_group("training")
    group.add_argument("-cuda_device", "--cuda_device", type=int, default=-1, help="Id of CUDA device.")
    group.add_argument("-max_epoch", "--max_epoch", type=int, default=30, help='max number of epoch')
    group.add_argument("-training_patience", "--training_patience", type=int, default=3, help='Early stopping patience.')
    group.add_argument("-dropout_keep_prob", "--dropout_keep_prob", type=float, default=0.5, help='dropout prob.')
    group.add_argument("-grad_norm", "--grad_norm", type=float, default=5.0, help='Max gradient norm.')
    group.add_argument("-lr_patience", "--lr_patience", type=int, default=2, help='Patience for reduce lr.')
    group.add_argument("-flag_clip_grad", "--flag_clip_grad", type=bool, default=True, help='clip the grad.')
    group.add_argument("-flag_lr_schedule", "--flag_lr_schedule", type=bool, default=True, help='scheduling lr.')

    group = parser.add_argument_group("baseline")

    group.add_argument("-baseline_model", "--baseline_model", type=str, default='ConvNet', help='baseline_model_type')

    args = parser.parse_args()
    return args







