# -*- coding:utf-8 -*-
from scripts.model import SuM_Model
from scripts.utils import set_cuda_device
from scripts.data_process import *
from scripts.config import config
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def train(args):

    model = trans_to_cuda(SuM_Model(args))

    train_path = args.train_data
    test_path = args.test_data
    val_path = args.val_data

    optimizer = optim.RMSprop(model.parameters(), args.lr)
    lr_scheduler = None
    if args.flag_lr_schedule:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=args.lr_patience, verbose=True)

    iteration = 0

    print("Start training...")

    model_dir = args.model_dir

    checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch0 = checkpoint['epoch']

    for epoch in range(args.max_epoch):
        print(epoch)
        loss = 0.0
        acc_all = 0.0
        counter = 0

        train_gen = batch_data_encode(train_path, batch_size=args.batch_size, train=True)

        model.train()
        for batch in tqdm(train_gen):
            iteration += 1
            optimizer.zero_grad()
            train_loss, train_matching_result, train_acc = model.get_batch_loss(batch)
            train_loss.backward()
            if args.flag_clip_grad and args.grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm, norm_type=2)
            optimizer.step()
            batch_loss = train_loss * args.batch_size
            loss += batch_loss
            acc_all += train_acc
            counter += 1

            if counter % 100 == 0:
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                batch_printer(epoch, counter, loss, acc_all, current_lr)
                print(train_matching_result)

        loss = loss / counter
        acc_all = acc_all / counter
        
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
            f'{model_dir}_{str(epoch)}_acc_{acc_all}.pt')

        print("Start validation...")
        model.eval()
        val_loss, val_acc = do_eval(model, val_path, args.batch_size)
        print("Val Loss: %.5f\t Val Acc: %.5f" % (val_loss, val_acc))

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if args.flag_lr_schedule:
            lr_scheduler.step(val_loss)

        print("Epoch %d\t Train Loss: %.5f\t Acc: %.5f\t Learning Rate: %.5f" % (epoch, loss, acc_all, current_lr))
        torch.save(model.state_dict(), f'{model_dir}_{str(epoch)}_acc_{acc_all}.pt')
        print("Model saved!")

    print("Model training completed.")

    print("Start testing...")
    _, test_acc = do_eval(model, test_path, args.batch_size)
    print("Test acc:", test_acc)


def do_eval(model, path, batch_size):
    loss = 0.0
    acc = 0.0
    val_gen = batch_data_encode(path, batch_size=batch_size)
    num = 0
    for batch in tqdm(val_gen):
        num += 1
        val_loss, _, val_acc = model.get_batch_loss(batch, train=False)
        loss += val_loss
        acc += val_acc

    loss = loss / num
    acc = acc / num

    return loss, acc


def batch_printer(epoch, counter, loss, acc, current_lr):
    print("Epoch %d\t Batch %d\t Train Loss: %.5f\t Acc: %.5f\t Learning Rate: %.5f" % (epoch, counter, loss / counter,
                                                                                        acc / counter, current_lr))


def ranking(args, data_path, model_path):
    '''

    :param args:
    :param  path:
    :param range0:
    :param range1:
    :return: answer, ranking_result, main_dia, full_text
    '''

    df = pd.read_csv(data_path)

    full_text = df['data']
    main_dia = df['diagnosis']
    answer = df['diagnosis_nation']

    lib = pd.read_csv(args.library_dir)
    lib_name = lib["index"].values[:]

    model = trans_to_cuda(SuM_Model(args))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    ranking_result = []

    print("Start coding ALL...")
    for t, s, md in tqdm(zip(full_text, answer, main_dia)):
        sim_rank1 = 0.0
        sim_rank2 = 0.0
        sim_rank3 = 0.0
        sim_rank4 = 0.0
        sim_rank5 = 0.0
        sim_rank10, sim_rank9, sim_rank8, sim_rank7, sim_rank6 = 0.0, 0.0, 0.0, 0.0, 0.0
        sim_result1 = 0
        sim_result2 = 0
        sim_result3 = 0
        sim_result4 = 0
        sim_result5 = 0
        sim_result10, sim_result9, sim_result8, sim_result7, sim_result6 = 0, 0, 0, 0, 0

        for i, name in enumerate(lib_name):
                data_input = get_single_token_seperated(md, t, name)
                sim00 = calculate_matching(model, data_input)

                if sim00 > sim_rank1:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim_rank7
                    sim_rank7 = sim_rank6
                    sim_rank6 = sim_rank5
                    sim_rank5 = sim_rank4
                    sim_rank4 = sim_rank3
                    sim_rank3 = sim_rank2
                    sim_rank2 = sim_rank1
                    sim_rank1 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = sim_result7
                    sim_result7 = sim_result6
                    sim_result6 = sim_result5
                    sim_result5 = sim_result4
                    sim_result4 = sim_result3
                    sim_result3 = sim_result2
                    sim_result2 = sim_result1
                    sim_result1 = name
                elif sim00 > sim_rank2:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim_rank7
                    sim_rank7 = sim_rank6
                    sim_rank6 = sim_rank5
                    sim_rank5 = sim_rank4
                    sim_rank4 = sim_rank3
                    sim_rank3 = sim_rank2
                    sim_rank2 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = sim_result7
                    sim_result7 = sim_result6
                    sim_result6 = sim_result5
                    sim_result5 = sim_result4
                    sim_result4 = sim_result3
                    sim_result3 = sim_result2
                    sim_result2 = name
                elif sim00 > sim_rank3:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim_rank7
                    sim_rank7 = sim_rank6
                    sim_rank6 = sim_rank5
                    sim_rank5 = sim_rank4
                    sim_rank4 = sim_rank3
                    sim_rank3 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = sim_result7
                    sim_result7 = sim_result6
                    sim_result6 = sim_result5
                    sim_result5 = sim_result4
                    sim_result4 = sim_result3
                    sim_result3 = name
                elif sim00 > sim_rank4:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim_rank7
                    sim_rank7 = sim_rank6
                    sim_rank6 = sim_rank5
                    sim_rank5 = sim_rank4
                    sim_rank4 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = sim_result7
                    sim_result7 = sim_result6
                    sim_result6 = sim_result5
                    sim_result5 = sim_result4
                    sim_result4 = name
                elif sim00 > sim_rank5:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim_rank7
                    sim_rank7 = sim_rank6
                    sim_rank6 = sim_rank5
                    sim_rank5 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = sim_result7
                    sim_result7 = sim_result6
                    sim_result6 = sim_result5
                    sim_result5 = name
                elif sim00 > sim_rank6:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim_rank7
                    sim_rank7 = sim_rank6
                    sim_rank6 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = sim_result7
                    sim_result7 = sim_result6
                    sim_result6 = name
                elif sim00 > sim_rank7:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim_rank7
                    sim_rank7 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = sim_result7
                    sim_result7 = name
                elif sim00 > sim_rank8:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim_rank8
                    sim_rank8 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = sim_result8
                    sim_result8 = name
                elif sim00 > sim_rank9:
                    sim_rank10 = sim_rank9
                    sim_rank9 = sim00
                    sim_result10 = sim_result9
                    sim_result9 = name
                elif sim00 > sim_rank10:
                    sim_rank10 = sim00
                    sim_result10 = name

        final_result = [sim_result1, sim_result2, sim_result3, sim_result4, sim_result5, sim_result6, sim_result7,
                        sim_result8, sim_result9, sim_result10]

        print("Input1", t[:30], "Input2", md, "Gold standard", s)
        print(final_result)

        ranking_result.append(final_result)
    print("Finish coding...")

    return answer, ranking_result, main_dia, full_text

def calculate_matching(model, data_input):

    model.eval()

    similarity_result = model.get_batch_matching_result(data_input)
    similarity_result = similarity_result.item()

    return similarity_result





if __name__ == "__main__":
    config = config()
    print(torch.cuda.is_available())

    data_path = config.test_dir
    model_path = config.model_dir

    answer, ranking_result, main_dia, full_text = ranking(config, data_path, model_path)

    pred_list = {'answer': answer,
                 'ranking_result': ranking_result,
                 'main_dia': main_dia,
                 'full_text': full_text}
    pred_list = pd.DataFrame(pred_list)
    pred_list.to_csv(
            f'{config.result_dir}_all.csv',
            index=False)

    



