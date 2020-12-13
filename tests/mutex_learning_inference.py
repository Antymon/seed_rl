import tensorflow as tf
import time
from seed_rl import grpc
import random
from abc import abstractmethod

from multiprocessing import Process
import os

ACTOR_NAMES = ["Alice", "Bob"]  # , "Celine", "Donald", "Elisa", "Francois", "Greta", "Henric", "Isa", "John", "Kay"]


def print_process_info(title):
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


def actor_loop(name, id):
    # print_process_info("actor {}".format(name))
    client = grpc.Client("0.0.0.0:6011")
    for i in range(5):
        time.sleep(random.random())
        val = client.infer(id)[0].numpy()
        print("{}'s request no {} got {} reply".format(name, i, "correct" if val == id else "INCORRECT"))


def learner_loop(model):
    # print_process_info('learner')
    for i in range(10):
        training = i % 2 == 0

        if training:
            model.train()

        print("{} {}".format("tick" if i % 2 == 0 else "tock", i))
        time.sleep(1)


class DummyModel(tf.Module):

    @abstractmethod
    def infer(self, env_id):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()


class MutexDummyModel(DummyModel):

    def __init__(self):
        print("MutexDummyModel")
        self.is_training = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)
        self.is_inferring = tf.Variable(initial_value=tf.zeros([len(ACTOR_NAMES)], dtype=tf.bool), trainable=False,
                                        dtype=tf.bool)

    def simulate_work(self, steps, prints_count=0, on_count_msg=None, on_finish_fn=None):
        condition = lambda i: tf.less(i, int(steps))

        def increment(i):
            if prints_count != 0 and on_count_msg is not None:
                if i % int(steps // prints_count) == 0:
                    tf.print(on_count_msg)
            return tf.add(i, 1)

        result = tf.while_loop(condition, increment, [tf.constant(0)])

        if on_finish_fn is not None and result == tf.constant(int(steps), tf.int32):
            on_finish_fn()

        return result

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32, 'env_id'), ])
    def infer(self, env_id):

        wait_cycles = tf.constant(0)
        while self.is_training:
            if wait_cycles % int(1e5) == 0:
                tf.print(env_id,"waiting", wait_cycles)
            wait_cycles += 1

        # always true, just to forbid compiler to move this before wait
        if wait_cycles >= 0:
            self.is_inferring[env_id].assign(True)

        tf.print(env_id, "waited",wait_cycles, tf.reduce_sum(tf.cast(self.is_inferring, tf.int32)))

        def on_finish_fn():
            self.is_inferring[env_id].assign(False)
            tf.print(env_id, "done", self.is_inferring[env_id])

        r = self.simulate_work(5e5, 10, ("working", env_id), on_finish_fn)

        return env_id, r

    @tf.function
    def train(self):
        self.is_training.assign(True)

        wait_cycles = tf.constant(0)
        while tf.reduce_sum(tf.cast(self.is_inferring, tf.int32)) != 0:
            if wait_cycles % int(1e5) == 0:
                tf.print("training waiting", wait_cycles, self.is_inferring)
            wait_cycles += 1


        tf.print("training start", wait_cycles, self.is_training)

        def on_finish_fn():
            self.is_training.assign(False)
            tf.print("training end", self.is_training)

        self.simulate_work(5e5, 10, "training", on_finish_fn)

        return


class MutexViaInferenceInvalidationDummyModel(DummyModel):

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
    run_experiment(MutexDummyModel())
    # run_experiment(MutexViaInferenceInvalidationDummyModel())
