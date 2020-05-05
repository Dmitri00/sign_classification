import tqdm
class Logger:
    def __init__(self, total_len):
        initial_desc = "ITERATION - loss: 0"
        self.pbar = tqdm.tqdm(initial=0, leave=False, total=total_len, desc=initial_desc)
        self.tqdm = tqdm