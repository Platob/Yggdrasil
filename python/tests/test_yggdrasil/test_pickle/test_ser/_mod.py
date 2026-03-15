DEFAULT_VALUE = 41

def remote_decorator(func):
    def wrapper(*args, **kwargs):
        print("remote_decorator")
        print(DEFAULT_VALUE)
        return func(*args, **kwargs)

    return wrapper