import pandas as pd
import re
from tqdm import tqdm
import numpy as np

def MetricsinTopK(list, gtruth, k):
    r = 0
    mrr = 0
    for i in range(k):
        m = list[i]
        if gtruth == m:
            r = 1
            mrr = float(1/(i+1))
            break

    return r, mrr


def F1_score_cal(list, gtruth, k, TP, FP, FN, DN, PN):
    num_dn_t = DN.get(gtruth, 0)
    num_dn_t += 1
    DN[gtruth] = num_dn_t

    if list.count(gtruth) != 0:
        index = list.index(gtruth)

        if index <= k-1:
            num_tp_t = TP.get(gtruth, 0)
            num_tp_t += 1
            TP[gtruth] = num_tp_t
        else:
            num_fn_t = FN.get(gtruth, 0)
            num_fn_t += 1
            FN[gtruth] = num_fn_t
    else:
        num_fn_t = FN.get(gtruth, 0)
        num_fn_t += 1
        FN[gtruth] = num_fn_t

    for i in range(k):
        num_pn_t = PN.get(list[i], 0)
        num_pn_t += 1
        PN[list[i]] = num_pn_t

        if list[i] != gtruth:
            num_fp = FP.get(list[i], 0)
            num_fp += 1
            FP[list[i]] = num_fp

    return TP, FP, FN, DN, PN


def F1_score_single(list, gtruth, k):
    tp = 0
    recall = 0
    for i in range(k):
        if gtruth == list[i]:
            tp = 1
            recall = 1
            break
    pre = float(tp) / float(k)

    return recall, pre


def TN_score_cal(predict_i, result, k, TN, res_list):
    res = predict_i[:k]
    for j in res_list:
        if j != result:
            if j not in res:
                num_tn = TN.get(j, 0)
                num_tn += 1
                TN[j] = num_tn

    return TN

def EvaluateScore(result, predict, k):
    l_r1 = []
    l_mrr1 = []
    l_re = []
    l_pre = []

    TP = {}
    FP = {}
    FN = {}
    DN = {}
    PN = {}
    TN = {}

    res_list = list(set(result))
    for i, predict_i in enumerate(predict):
        r1, mrr1 = MetricsinTopK(predict_i, result[i], k)
        re, pre = F1_score_single(predict_i, result[i], k)
        l_r1.append(r1)
        l_mrr1.append(mrr1)
        l_re.append(re)
        l_pre.append(pre)
        TP, FP, FN, DN, PN = F1_score_cal(predict_i, result[i], k, TP, FP, FN, DN, PN)
        TN = TN_score_cal(predict_i, result[i], k, TN, res_list)

    f1_key = []
    f1_marco = []

    ttp = 0
    tfp = 0
    tfn = 0
    tdn = 0
    tpn = 0
    ttn = 0

    for key in TP.keys():
        tp = TP.get(key, 0)
        ttp += tp
    for key in FP.keys():
        fp = FP.get(key, 0)
        tfp += fp
    for key in FN.keys():
        fn = FN.get(key, 0)
        tfn += fn
    for key in DN.keys():
        dn = DN.get(key, 0)
        tdn += dn

    for key in TN.keys():
        tn = TN.get(key, 0)
        ttn += tn

    for key in PN.keys():
        pn = PN.get(key, 0)
        tpn += pn

    for key in TP.keys():
        tp = TP.get(key, 0)
        fp = FP.get(key, 0)
        fn = FN.get(key, 0)
        dn = DN.get(key, 0)
        pn = PN.get(key, 0)

        p = float(tp) / float(tp + fp)
        r = float(tp) / float(tp + fn)
        f1 = 2*p*r/(p+r)
        f1_key.append(f1)

        f1_m = float(2*tp) / float(dn + pn)
        f1_marco.append(f1_m)

    r_r1 = float(sum(l_r1) / len(l_r1))
    r_mrr1 = float(sum(l_mrr1) / len(l_mrr1))
    mip = float(ttp) / float(ttp + tfp)
    mir = float(ttp) / float(ttp + tfn)
    tpr = mir
    fpr = float(tfp) / float(len(result) * (len(res_list) - 1))
    auc = float(1 + tpr - fpr) * 0.5
    f1_mirco = float(2 * ttp) / float(tdn + tpn)

    return r_r1, r_mrr1, f1_marco, f1_mirco, mip, mir, fpr, auc

if __name__ == '__main__':

        df = pd.read_csv('result file path')

        answer = df['answer'].values
        pred_text = df['ranking_result'].values

        r1, mrr1, f1_marco_1, f1_mirco_1, mip1, mir1, fpr1, aoc1 = EvaluateScore(answer, pred_text, 1)
        r5, mrr5, f1_marco_5, f1_mirco_5, mip5, mir5, fpr5, aoc5 = EvaluateScore(answer, pred_text, 5)
        r10, mrr10, f1_marco_10, f1_mirco_10, mip10, mir10, fpr10, aoc10 = EvaluateScore(answer, pred_text, 10)

        _, _, _, _, _, mir2, fpr2, _, = EvaluateScore(answer, pred_text, 2)
        _, _, _, _, _, mir3, fpr3, _, = EvaluateScore(answer, pred_text, 3)
        _, _, _, _, _, mir4, fpr4, _, = EvaluateScore(answer, pred_text, 4)
        _, _, _, _, _, mir6, fpr6, _, = EvaluateScore(answer, pred_text, 6)
        _, _, _, _, _, mir7, fpr7, _, = EvaluateScore(answer, pred_text, 7)
        _, _, _, _, _, mir8, fpr8, _, = EvaluateScore(answer, pred_text, 8)
        _, _, _, _, _, mir9, fpr9, _, = EvaluateScore(answer, pred_text, 9)


        print("Evaluate Metrics:")
        print("MRR@1:", mrr1)
        print("F1_Mirco@1", f1_mirco_1)
        print("F1_Mirco_Precision,Recall@1", mip1, mir1)
        print("-----------------------")
        print("MRR@5:", mrr5)
        print("F1_Mirco@5", f1_mirco_5)
        print("F1_Mirco_Precision,Recall@5", mip5, mir5)
        print("-----------------------")
        print("MRR@10:", mrr10)
        print("F1_Mirco@10", f1_mirco_10)
        print("F1_Mirco_Precision,Recall@10", mip10, mir10)

        aucall = 0.5 * float(
            mir1 * fpr2 - mir2 * fpr1 +
            mir2 * fpr3 - mir3 * fpr2 +
            mir3 * fpr4 - mir4 * fpr3 +
            mir4 * fpr5 - mir5 * fpr4 +
            mir5 * fpr6 - mir6 * fpr5 +
            mir6 * fpr7 - mir7 * fpr6 +
            mir7 * fpr8 - mir8 * fpr7 +
            mir8 * fpr9 - mir9 * fpr8 +
            mir9 * fpr10 - mir10 * fpr9 +
            1 + mir10 - fpr10)

        print("AUC", aucall)




