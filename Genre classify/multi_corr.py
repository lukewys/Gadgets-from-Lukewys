import multiprocessing
import time
import copy
import numpy as np


def corr_two_seq(seq1, seq2, gate_size):
    corr = []
    for m in range(len(seq1) - gate_size):
        seqa = seq1[m:m + gate_size]
        corr.append(np.correlate(seqa, seq2))
    return corr


def mean_corr_normalize(corr):
    means = []
    for pieces in corr:
        # pieces = pieces / np.max(pieces)
        means.append(np.mean(pieces))
    return np.mean(np.array(means))


def combine(seq1, seq2, gate_size):
    corr = corr_two_seq(seq1, seq2, gate_size)
    return mean_corr_normalize(corr)


def cor_score(data_1, data_2):
    i = 0
    s = 0

    for piece1 in data_1:
        for piece2 in data_2:
            s += combine(piece1, piece2, 64)
            i += 1
    s = s / i
    return s


def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


if __name__ == "__main__":
    data_jazz = np.load('jazz_g64_notes.npy')
    data_bach = np.load('bach_g64_notes.npy')
    seq1 = data_bach
    seq2 = data_jazz
    num_threads = 8
    data_span = 10
    SCORE_LIST = np.zeros((num_threads))
    FINISH_FLAG = np.zeros((num_threads))
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_threads)
    n = 1000
    results = []

    for i in range(num_threads):
        results.append(
            pool.apply_async(cor_score,
                             (seq1[n:n + data_span],
                              seq2)))
        n += data_span

    pool.close()
    pool.join()
    SCORE_LIST = []
    for result in results:
        SCORE_LIST.append(result.get())

    '''    while np.sum(FINISH_FLAG) != len(FINISH_FLAG):
        if np.random.rand()>0.99999:
            print(FINISH_FLAG)'''

    print(elapsed(time.time() - start_time))
    print(SCORE_LIST)
    print(np.mean(SCORE_LIST))
