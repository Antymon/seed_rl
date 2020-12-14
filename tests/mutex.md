#Which approach to the on-policy training/inference synchronization is best?

Hi,
I am thinking of a synchronization technique that would be suitable to introduce an on-policy algorithm such as PPO
which alternates between data collection and training. I came up with 3 basic ideas and would very much appreciate it if anyone could help me judge if any of them is valid. I am leaning towards the last one as it doesn't use busy waits (i.e., empty while loops on sync variables),
although I am not entirely sure how (un)acceptable is a busy loop on a GPU. Although below I present simple pseudocode
to demonstrate ideas, I implemented them minimally outside of SEED to try out synchronization via tf variables
inside of TF functions called through SEED's gRPC framework to examine potential problems. One of such problems is the
default mode of autograph which unaware of my synchronization efforts (re)moves stuff thus compromising my intents, but I did
some artificial dependencies hackery to go around this problem (guess optimizations can be controlled but ideally I would
hope for sth like much-hated C++ `volatile` qualifier). In any case: thanks for any feedback, if any of the below approaches, are valid
and/or ideas for alternatives.

```
# Model is a tf.module with a constructor and 2 tf.functions: infer and train called with gRPC as in SEED.
# tf variables live on cpu of a sole host associated with learner

#1 busy waits, many sync variables
class Model() {

    def Model() {
        var training = tf.Var(False)
        var inferring = tf.Var([False]*NUM_ACTORS)
    }

    def infer(id) {
        while training:
            pass

        inferring[id] = True
        ...
        inferring[id] = False

        return result
    }

    def train() {
        training = True
        while sum(inferring) > 0:
            pass

        ...
        training = False
    }
}


#2 busy waits, a single sync flag, invalidating inference result
class Model() {

    def Model() {
        var training = tf.Var(False)
    }

    # invariant: time(inference) < time(training)
    def infer(id) {
        do {
            while training:
                pass

            ...

        } while not training

        return result
    }

    def train {
        training = True
        ...
        training = False
    }
}

#3 no busy waits, single sync variable, client needs to balance inference calls and reject some of the returns
class Model {

    def Model() {
        var training = tf.Var(False)
    }

    # invariant: time(inference) < time(training)
    def infer(id) {

        if training:
            return None

        ...

        if training:
            return None

        return result
    }

    def train {
        training = True
        ...
        training = False
    }
}
```