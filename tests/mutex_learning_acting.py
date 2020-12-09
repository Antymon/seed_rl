import tensorflow as tf
import time
from seed_rl import grpc
import random

max_tick_tocks = 10

from multiprocessing import Process
import os


def print_process_info(title):
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


def worker_loop(name):
    print_process_info(name)
    client = grpc.Client("0.0.0.0:6011")
    for i in range(max_tick_tocks):
        time.sleep(random.random())
        val = client.inference(name).numpy()
        print("{}'s request no {} got {} reply: {}".format(name, i, "correct" if val % 2 == 1 else "INCORRECT", val))


class DummyModel(tf.Module):
    inference_specs = [tf.TensorSpec([], tf.string, 'env_id'), ]

    def __init__(self):
        self.var = tf.Variable(initial_value=1, trainable=False, dtype=tf.int64)

    @tf.function(input_signature=inference_specs)
    def inference(self, env_id):

        v = self.var
        while v % 2 == 0:
            v = self.var
            # tf.print(env_id, "waits")
        return v

    @tf.function
    def inc(self):
        return self.var.assign_add(1)

    @tf.function
    def dec(self):
        return self.var.assign_sub(1)


if __name__ == '__main__':

    print("Processes should stay silent on TOCK")

    m = DummyModel()
    server = grpc.Server(["0.0.0.0:6011"])
    server.bind(m.inference)
    server.start()

    print_process_info('main line')
    p = Process(target=worker_loop, args=('bob',))
    p2 = Process(target=worker_loop, args=('alice',))

    m.dec()
    p.start()
    p2.start()

    while m.var.read_value() < max_tick_tocks:
        m.inc()
        count = m.var.read_value().numpy()
        print("{} {}".format("tick" if count % 2 == 0 else "tock", count))
        time.sleep(1)

    p.join()
    p2.join()

    server.shutdown()

    print("Done")
