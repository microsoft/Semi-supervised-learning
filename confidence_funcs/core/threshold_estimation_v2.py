from sklearn.metrics import accuracy_score
import numpy as np


def get_threshold():
    pass


def determine_threshold(
    classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold=0.01
):

    # sort the scores in ascending order
    scores = inf_out[auto_lbl_conf.score_type]
    scores_sorted_index = np.argsort(scores)
    scores = scores[scores_sorted_index]
    
    scores_mat = np.tile(scores, (len(scores), 1))

    # reorder the labels so that they match the scores
    y_true = val_ds.Y[scores_sorted_index]
    y_pred = inf_out["labels"][scores_sorted_index]

    for class_ in classes:
        loc = y_pred == class_
        # check how many scores at loc are greater than the threshold
        scor
