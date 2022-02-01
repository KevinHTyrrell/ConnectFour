import numpy as np
import pandas as pd
from random import randint, choices
from tensorflow import keras
from abc import ABC


class BaseBot:
    def __init__(self, state_shape):
        self._action_space_size     = 7
        self._alpha                 = 0.005  # learning rate
        self._batch_size            = 32
        self._epsilon               = 1.0
        self._epsilon_decay         = 0.99
        self._epsilon_limit         = 0.2
        self._gamma                 = 0.01
        self._model                 = None
        self._memory                = list()
        self._model_parameters      = None
        self._train_epochs          = 32
        self._train_sample_size     = 128
        self._state_shape           = state_shape
        self._build_model()

    def get_action(self, current_state):
        if self._epsilon > self._epsilon_limit:
            insert_col = randint(0, self._action_space_size - 1)
            self._epsilon *= self._epsilon_decay
        else:
            current_state = np.expand_dims(current_state, axis=-1)
            insert_col = self._compute_column(current_state)

        if len(self._memory) != 0 and len(self._memory) % self._train_sample_size == 0:
            self._train()
        return insert_col

    def store_memory(self, initial_state, column, next_state, reward):
        initial_state = np.expand_dims(initial_state, axis=(0, -1))
        next_state = np.expand_dims(next_state, axis=(0, -1))
        self._memory.append((initial_state, column, next_state, reward))

    def _compute_column(self, curr_state):
        curr_state = np.expand_dims(curr_state, axis=0)
        model_predictions = self._model.predict(curr_state).squeeze()
        column = np.argmax(model_predictions)
        return column

    def _build_model(self):
        input_layer = keras.layers.Input(shape=self._state_shape)
        x = keras.layers.Conv2D(filters=64,
                                kernel_size=(4, 4),
                                strides=(1, 1),
                                padding='same',
                                kernel_regularizer=keras.regularizers.l1_l2(5e-4, 5e-4))(input_layer)
        x = keras.layers.LeakyReLU(0.3)(x)
        x = keras.layers.Conv2D(filters=64,
                                kernel_size=(2, 2),
                                strides=(1, 1),
                                padding='same',
                                kernel_regularizer=keras.regularizers.l1_l2(5e-4, 5e-4))(x)
        x = keras.layers.LeakyReLU(0.3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=100,
                               kernel_regularizer=keras.regularizers.l1_l2(5e-4, 5e-4))(x)
        x = keras.layers.LeakyReLU(0.3)(x)
        output_layer = keras.layers.Dense(units=self._action_space_size,
                                          activation='softmax')(x)
        model = keras.models.Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        self._model = model

    def _train(self):
        train_x_list = list()
        train_y_list = list()
        sample_memory = choices(self._memory, k=self._train_sample_size)

        for initial_state, action, next_state, reward in sample_memory:
            # current_rewards = self._model.predict(np.expand_dims(initial_state, axis=0))
            current_rewards = self._model.predict(initial_state).squeeze()
            q_val = self._get_q_value(initial_state, action, reward, next_state)
            current_rewards[action] = q_val
            current_rewards = np.expand_dims(current_rewards, axis=0)
            train_x_list.append(initial_state)
            train_y_list.append(current_rewards)
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        self._model.fit(train_x, train_y, batch_size=self._batch_size)

    def _get_q_value(self, initial_state, action, reward, next_state):
        # initial_state_predictions = self._model.predict(np.expand_dims(initial_state, axis=0))
        # next_state_predictions = self._model.predict(np.expand_dims(next_state, axis=0))
        initial_state_predictions = self._model.predict(initial_state).squeeze()
        next_state_predictions = self._model.predict(next_state).squeeze()
        initial_q_max = initial_state_predictions[action]
        next_q_max = next_state_predictions[np.argmax(next_state_predictions)]
        q_current = (1 - self._alpha) * initial_q_max
        q_future = self._alpha * (reward * self._gamma * next_q_max)
        return q_current + q_future