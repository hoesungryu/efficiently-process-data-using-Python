from time import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time()
        result = original_fn(*args, **kwargs)
        end_time = time()
        print("WorkingTime [{:<15s}]: {:.2f} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn
    