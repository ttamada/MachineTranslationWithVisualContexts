

import numpy as np


def get_data_set(target_tok_id_file, visual_vec_file, num_targets):
    # store lines of both target ids and visual vectors
    with open(target_tok_id_file, "r") as f:
        target_id_lines = [[int(id) for id in l.split()] for l in f.read().splitlines() if l]
    with open(visual_vec_file, "r") as f:
        visual_vec_lines = [[int(num) for num in l.split()] for l in f.read().splitlines() if l]
    if len(target_id_lines) != len(visual_vec_lines):
        raise Exception("Not all of target sentence have a visual context!")


    # create data set
    X = [] #input set (visual contexts)
    T = [] #target set (target words)
    for i in range(0, len(target_id_lines)):
        t_vec = np.zeros(num_targets)
        ids_in_line = target_id_lines[i]
        for id in ids_in_line:
            t_vec[id] = 1
        X.append(np.array(visual_vec_lines[i]))
        T.append(np.array(t_vec))

    return X, T


