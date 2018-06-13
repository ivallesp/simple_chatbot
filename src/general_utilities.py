import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]

def batching(list_of_iterables, n=1, infinite=False, return_incomplete_batches=False):
    list_of_iterables = [list_of_iterables] if type(list_of_iterables) is not list else list_of_iterables
    assert(len({len(it) for it in list_of_iterables}) == 1)
    l = len(list_of_iterables[0])
    while 1:
        for ndx in range(0, l, n):
            if not return_incomplete_batches:
                if (ndx+n) > l:
                    break
            yield [iterable[ndx:min(ndx + n, l)] for iterable in list_of_iterables]

        if not infinite:
            break

def exponential_decay_generator(start, finish, decay=0.999):
    x=start
    while 1:
        x = x*decay + finish*(1-decay)
        yield x
        
def get_batcher(dialog_codes, BATCH_SIZE, shuffle=True):
    if shuffle:
        np.random.shuffle(dialog_codes)
    questions, answers = zip(*dialogs_codes)
    answers = [tuple([go_symbol]+list(seq)) for seq in answers] # Preppend go symbol
    questions = np.expand_dims(np.row_stack(questions), 2)
    answers = np.expand_dims(np.row_stack(answers), 2)
    batcher = batching([questions, answers], n=BATCH_SIZE)
    return(batcher)