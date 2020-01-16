import copy

import numpy as np

from game import Game
from player import Player

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

POSITIONS_PER_ITERATION = 20000
SAVE_ITERATIONS = 1
TRAIN_EPOCHS = 1
BATCH_SIZE = 128
WINDOW_SIZE = 200000
POSITIONS_PER_EPOCH = 100000
EPSILON = 0.01
MINIMUM_TRAIN_SIZE = 50000


class BotModel(Model):
    def __init__(self):
        super(BotModel, self).__init__()
        self.d1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.d2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.d2 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.d3 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.d4 = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.0001))

    def call(self, x, **kwargs):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


def convert_to_input(x):
    p = x[0]
    v = x[1]
    a = x[2]
    return tf.concat(
        [tf.reshape(p, [-1]),
         tf.reshape(tf.square(p), [-1]),
         tf.reshape(v, [-1]),
         tf.reshape(tf.square(v), [-1]),
         tf.reshape(a, [-1]),
         tf.reshape(tf.square(a), [-1])], axis=0)


def convert_to_input_pair(x, y):
    return convert_to_input(x), tf.reshape(y, [-1])


class NNBot(Player):
    def __init__(self, model=None):
        super().__init__()
        if model is None:
            model = load_model('model')
        self.model = model
        self.skip = 0
        self.last_score = 0.5

    def get_action(self, game) -> np.ndarray:
        if self.skip != 0:
            self.skip -= 1
            return game.last_action[game.turn]
        self.skip = 3
        actions = [np.random.uniform(-1, 1, size=2) for _ in range(7)]
        actions.append(game.p[1 - game.turn] - game.p[game.turn])
        states = []
        for action in actions:
            next_game = copy.deepcopy(game)
            next_game.step(action)
            for _ in range(9):
                next_game.step(next_game.last_action[next_game.turn])
            if next_game.turn == 0:
                states.append([next_game.p, next_game.v, next_game.last_action])
            else:
                states.append([[next_game.p[1], next_game.p[0]], [next_game.v[1], next_game.v[0]],
                               [next_game.last_action[1], next_game.last_action[0]]])

        inputs = tf.convert_to_tensor([convert_to_input(random_flip(s, None)[0]) for s in states])
        predictions = self.model(inputs, training=False)
        best_id = np.argmax(predictions)
        self.last_score = predictions[best_id].numpy()[0]
        return actions[best_id]

    def get_score(self):
        return self.last_score


loss_object = tf.keras.losses.binary_crossentropy
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')


@tf.function
def train_step(x, y, model):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)


@tf.function
def test_step(x, y, model):
    predictions = model(x)
    t_loss = loss_object(y, predictions)

    test_loss(t_loss)
    test_accuracy(y, predictions)


def random_flip(x, y):
    if np.random.uniform(0, 1) < 0.5:
        mult = tf.constant([[[1, -1], [1, -1]], [[1, -1], [1, -1]], [[1, -1], [1, -1]]], dtype=tf.float64)
        x = tf.multiply(x, mult)

    if np.random.uniform(0, 1) < 0.5:
        mult = tf.constant([[[-1, 1], [-1, 1]], [[-1, 1], [-1, 1]], [[-1, 1], [-1, 1]]], dtype=tf.float64)
        x = tf.multiply(x, mult)

    return x, y


def train(train_x, train_y, test_x, test_y, model):
    print('Start train')
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)) \
        .shuffle(1000000) \
        .take(POSITIONS_PER_EPOCH) \
        .map(random_flip, num_parallel_calls=8) \
        .map(convert_to_input_pair, num_parallel_calls=8) \
        .batch(BATCH_SIZE) \
        .prefetch(4)

    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)) \
        .shuffle(1000000) \
        .take(POSITIONS_PER_EPOCH // 10) \
        .map(random_flip, num_parallel_calls=8) \
        .map(convert_to_input_pair, num_parallel_calls=8) \
        .batch(BATCH_SIZE)

    for epoch in range(TRAIN_EPOCHS):
        print('Epoch', epoch)
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x, y in train_ds:
            train_step(x, y, model)

        for x, y in test_ds:
            test_step(x, y, model)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


def pick_action(game, bot: NNBot):
    if np.random.uniform(0, 1) <= EPSILON:
        if np.random.uniform(0, 1) <= 0.5:
            return np.random.uniform(-1, 1, size=2)
        return game.p[1 - game.turn] - game.p[game.turn]
    return bot.get_action(game)


def play_one_game(bot: NNBot):
    history = []
    game = Game()

    while game.state != Game.STATE_FINISHED:
        action = pick_action(game, bot)
        actions = np.copy(game.last_action)
        actions[game.turn] = action
        history.append((np.copy(game.p), np.copy(game.v), actions, game.turn))
        game.step(action)

    x = []
    y = []

    if game.winner == 2:
        return [], []

    for p, v, a, turn in history[-1000:]:
        if turn == 1:
            p = np.array([p[1], p[0]])
            v = np.array([v[1], v[0]])
            a = np.array([a[1], a[0]])

        if game.winner == 2:
            z = 0.5
        elif game.winner == turn:
            z = 1.0
        else:
            z = 0.0

        x.append((p, v, a))
        y.append(z)

    return x, y


def load_model(path):
    model = BotModel()
    model.compile(loss=loss_object, optimizer=optimizer)
    game = Game()
    x, y = convert_to_input_pair([game.p, game.v, game.last_action], 0.0)
    model.train_on_batch(tf.convert_to_tensor([x]), tf.convert_to_tensor([y]))
    try:
        model.load_weights(path)
    except Exception as e:
        print(e)
        model = BotModel()

    return model


def main():
    last_saved = 0

    train_x = []
    train_y = []

    test_x = []
    test_y = []

    model_path = 'model'
    model = load_model(model_path)

    bot = NNBot(model)

    iteration = 0

    while True:
        iteration += 1
        print('Iteration', iteration)
        added = 0
        while added < POSITIONS_PER_ITERATION:
            # print('added', added)
            x, y = play_one_game(bot)
            added += len(y)
            train_x.extend(x)
            train_y.extend(y)

        print('To', len(train_y))

        train_x = train_x[-WINDOW_SIZE:]
        train_y = train_y[-WINDOW_SIZE:]

        added = 0
        while added < POSITIONS_PER_ITERATION // 10:
            x, y = play_one_game(bot)
            added += len(y)
            test_x.extend(x)
            test_y.extend(y)

        test_x = test_x[-WINDOW_SIZE // 10:]
        test_y = test_y[-WINDOW_SIZE // 10:]

        print('Train:', len(train_y))

        if len(train_y) >= MINIMUM_TRAIN_SIZE:
            train(train_x, train_y, test_x, test_y, model)
        else:
            print('SKIP')

        last_saved += 1
        if last_saved >= SAVE_ITERATIONS:
            last_saved = 0
            model.save_weights(model_path, save_format='tf')


if __name__ == '__main__':
    main()
