import numpy as np

def Accuracy(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the accuracy for prediction.
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - pred  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - Acc : (scalar, float), Computed accuracy score
    # ========================= EDIT HERE =========================
    corr = label[label == pred].shape[0]
    Acc = corr / label.shape[0]
    # =============================================================
    return Acc

def Precision(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Precision for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - pred  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - precision : (scalar, float), Computed precision score
    # ========================= EDIT HERE =========================
    filt = label[pred == 1]
    precision = 1 if filt.shape[0] == 0 else (filt[filt == 1].shape[0] / filt.shape[0]) 
    # =============================================================
    return precision

def Recall(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Recall for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - pred  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - recall : (scalar, float), Computed recall score
    # ========================= EDIT HERE =========================
    filt = pred[label == 1]
    recall = 1 if filt.shape[0] == 0 else (filt[filt == 1].shape[0] / filt.shape[0])
    # =============================================================
    return recall

def F_measure(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the F-measure score for prediction.
    #         you can erase the code. (F_score = 0.)
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - pred  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - F_score : (scalar, float), Computed F-score score
    # ========================= EDIT HERE =========================
    prec = Precision(label, pred)
    recl = Recall(label, pred)

    F_score = 1 if prec == 0 and recl == 0 else (2 * prec * recl / (prec + recl))
    # =============================================================
    return F_score

def MAP(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the code to calculate the MAP for prediction.
    #         Notice that, hypo is the real value array in (0, 1)
    #         MAP (at = 10) means MAP @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed MAP score
    # ========================= EDIT HERE =========================
    def AP (label, hypo, at):
        label = label[np.argsort(hypo)[::-1]]
        label_c = label.cumsum()
        return ((label * label_c)[:at] / np.arange(1, at + 1)).sum() / label_c[-1]
    
    Map = np.mean([AP(l, h, at) for (l, h) in zip(label, hypo)])
    # =============================================================
    return Map

def nDCG(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the each code to calculate the nDCG for prediction.
    #         you can erase the code. (dcg, idcg, ndcg = 0.)
    #         Notice that, hypo is the real value array in (0, 1)
    #         nDCG (at = 10 ) means nDCG @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed nDCG score


    def DCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        dcg = (label[np.argsort(hypo)[::-1]][:at] / np.log2(np.arange(2, at + 2))).sum()
        # =============================================================
        return dcg

    def IDCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        idcg = (np.sort(label)[::-1][:at] / np.log2(np.arange(2, at + 2))).sum()
        # =============================================================
        return idcg
    # ========================= EDIT HERE =========================
    ndcg = np.mean([DCG(l, h, at) / IDCG(l, h, at) for (l, h) in zip(label, hypo)])
    # =============================================================
    return ndcg

# =============================================================== #
# ===================== DO NOT EDIT BELOW ======================= #
# =============================================================== #

def evaluation_test1(label, pred, at = 10):
    result = {}

    result['Accuracy '] = Accuracy(label, pred)
    result['Precision'] = Precision(label, pred)
    result['Recall   '] = Recall(label, pred)
    result['F_measure'] = F_measure(label, pred)

    return result

def evaluation_test2(label, hypo, at = 10):
    result = {}

    result['MAP  @%d'%at] = MAP(label, hypo, at)
    result['nDCG @%d'%at] = nDCG(label, hypo, at)

    return result
