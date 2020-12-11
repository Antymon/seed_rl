import tensorflow as tf
import time
from seed_rl import grpc
import random

from multiprocessing import Process
import os

ACTOR_NAMES = ["Alice", "Bob", "Celine", "Donald", "Elisa", "Francois", "Greta", "Henric", "Isa", "John", "Kay"]

def print_process_info(title):
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


def actor_loop(name, id):
    print_process_info("actor {}".format(name))
    client = grpc.Client("0.0.0.0:6011")
    for i in range(10):
        time.sleep(random.random())
        val = client.infer(id).numpy()
        print("{}'s request no {} got {} reply".format(name, i, "correct" if val == id else "INCORRECT"))


def learner_loop():
    print_process_info('lerner')
    for i in range(10):
        training = i % 2 == 0

        if training:
            model.train()

        print("{} {}".format("tick" if i % 2 == 0 else "tock", i))
        time.sleep(1)


class DummyModel(tf.Module):

    def __init__(self):
        self.is_training = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)
        self.is_inferring = tf.Variable(initial_value=tf.zeros([len(ACTOR_NAMES)], dtype=tf.bool), trainable=False,
                                        dtype=tf.bool)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32, 'env_id'), ])
    def infer(self, env_id):
        while self.is_training:
            pass

        self.is_inferring[env_id].assign(True)
        for _ in range(int(1e4)):
            pass

        self.is_inferring[env_id].assign(False)

        return env_id

    @tf.function
    def train(self):
        self.is_training.assign(True)
        while tf.reduce_sum(tf.cast(self.is_inferring, tf.int32)) != 0:
            pass

        tf.print("training start")

        for _ in range(int(1e5)):
            pass

        tf.print("training end")

        self.is_training.assign(False)

        return


if __name__ == '__main__':

    model = DummyModel()

    server = grpc.Server(["0.0.0.0:6011"])
    server.bind(model.infer)
    server.start()

    workers = [Process(target=actor_loop, args=(name, idx)) for idx, name in enumerate(ACTOR_NAMES)]
    for w in workers:
        w.start()

    learner_loop()

    for w in workers:
        w.join()

    server.shutdown()

    print("Done")
