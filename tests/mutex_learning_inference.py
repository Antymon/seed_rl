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


def learner_loop(model):
    print_process_info('learner')
    for i in range(10):
        training = i % 2 == 0

        if training:
            model.train()

        print("{} {}".format("tick" if i % 2 == 0 else "tock", i))
        time.sleep(1)


class MutexDummyModel(tf.Module):

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

class MutexViaInferenceInvalidationDummyModel(tf.Module):

    def __init__(self):
        self.is_training = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32, 'env_id'), ])
    def infer(self, env_id):
        # invariant: act of inference is never longer than act of training as a timespan between sync primitive toggle
        # if violated, inference outcome is unknown and likely to contribute noise
        # do inference only if training has not started and accept result only if training is not on when returning
        current_inference_invalid = tf.constant(True)
        while current_inference_invalid:
            while self.is_training:
                pass
            # imitate work
            for _ in range(int(1e4)):
                pass

            current_inference_invalid = self.is_training

        return env_id

    @tf.function
    def train(self):
        self.is_training.assign(True)

        tf.print("training start")

        for _ in range(int(1e5)):
            pass

        tf.print("training end")

        self.is_training.assign(False)

        return

def run_experiment(model):
    server = grpc.Server(["0.0.0.0:6011"])
    server.bind(model.infer)
    server.start()

    workers = [Process(target=actor_loop, args=(name, idx)) for idx, name in enumerate(ACTOR_NAMES)]
    for w in workers:
        w.start()

    learner_loop(model)

    for w in workers:
        w.join()

    server.shutdown()

    print("Done")

if __name__ == '__main__':


    # run_experiment(MutexDummyModel())
    run_experiment(MutexViaInferenceInvalidationDummyModel())

