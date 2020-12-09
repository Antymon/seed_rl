import tensorflow as tf
import time
from seed_rl import grpc
import threading
import random

max_i = 10

from multiprocessing import Process
import os


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name):
    # info('function f')
    print('hello', name)
    client = grpc.Client("0.0.0.0:6011")
    id = threading.get_ident()
    for i in range(5):
        time.sleep(random.random())
        val = client.inference(id)
        print("Thread {} request {} got reply {}".format(id, i, val))


class Counter(object):
    def __init__(self):
        self.i = 0


if __name__ == '__main__':
    print("hello")

    var = tf.Variable(initial_value=1, trainable=False, dtype=tf.int64)


    inference_specs = [
        tf.TensorSpec([], tf.int64, 'env_id'),
    ]

    c = Counter()
    c.i = 1


    @tf.function(input_signature=inference_specs)
    def inference(env_id):
        tf.print(env_id)
        while var % 2 == 0:
            pass
        return var


    server = grpc.Server(["0.0.0.0:6011"])

    print("lol0")

    server.bind(inference)
    print("lol")

    server.start()

    print("lol2")

    info('main line')
    p = Process(target=f, args=('bob',))
    p2 = Process(target=f, args=('alice',))

    var = var - 1
    p.start()
    p2.start()

    while var.numpy()[0] < max_i:
        var = var + 1
        print(var)
        time.sleep(1)

    p.join()
    p2.join()

    server.shutdown()

    print("Done")
