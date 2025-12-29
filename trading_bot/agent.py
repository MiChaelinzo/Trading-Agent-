import random
import os

from collections import deque

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense, Input
from keras.optimizers import Adam


@keras.saving.register_keras_serializable()
def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy

        # agent config
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(units=128, activation="relu"),
            Dense(units=256, activation="relu"),
            Dense(units=256, activation="relu"),
            Dense(units=128, activation="relu"),
            Dense(units=self.action_size)
        ])

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        Uses batch prediction for improved performance.
        """
        mini_batch = random.sample(self.memory, batch_size)
        
        # Prepare batch data
        states = np.array([sample[0][0] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        actions = [sample[1] for sample in mini_batch]
        rewards = [sample[2] for sample in mini_batch]
        dones = [sample[4] for sample in mini_batch]
        
        # DQN
        if self.strategy == "dqn":
            # Batch predict for both states and next_states
            q_values = self.model.predict(states, verbose=0)
            next_q_values = self.model.predict(next_states, verbose=0)
            
            for i in range(batch_size):
                if dones[i]:
                    target = rewards[i]
                else:
                    target = rewards[i] + self.gamma * np.amax(next_q_values[i])
                q_values[i][actions[i]] = target
            
            X_train = states
            y_train = q_values

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())
            
            # Batch predict
            q_values = self.model.predict(states, verbose=0)
            next_q_values = self.target_model.predict(next_states, verbose=0)
            
            for i in range(batch_size):
                if dones[i]:
                    target = rewards[i]
                else:
                    target = rewards[i] + self.gamma * np.amax(next_q_values[i])
                q_values[i][actions[i]] = target
            
            X_train = states
            y_train = q_values

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())
            
            # Batch predict from both networks
            q_values = self.model.predict(states, verbose=0)
            next_q_values_model = self.model.predict(next_states, verbose=0)
            next_q_values_target = self.target_model.predict(next_states, verbose=0)
            
            for i in range(batch_size):
                if dones[i]:
                    target = rewards[i]
                else:
                    # Use model to select action, target to evaluate
                    best_action = np.argmax(next_q_values_model[i])
                    target = rewards[i] + self.gamma * next_q_values_target[i][best_action]
                q_values[i][actions[i]] = target
            
            X_train = states
            y_train = q_values
                
        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            X_train, y_train,
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        """Save model in Keras 3 format with .keras extension"""
        filepath = "models/{}_{}.keras".format(self.model_name, episode)
        self.model.save(filepath)

    def load(self):
        """Load model - supports both new .keras and legacy formats"""
        model_path = "models/" + self.model_name
        
        # Try new .keras format first
        if model_path.endswith('.keras'):
            return load_model(model_path, custom_objects=self.custom_objects)
        
        # Check if .keras version exists
        keras_path = model_path + ".keras"
        if os.path.exists(keras_path):
            return load_model(keras_path, custom_objects=self.custom_objects)
        
        # For legacy models, check if the file exists and try to load
        if os.path.exists(model_path):
            # Legacy models from TF1 SavedModel format need special handling
            try:
                # Try loading as legacy H5
                return load_model(model_path, custom_objects=self.custom_objects)
            except ValueError:
                # If legacy format fails, try using TFSMLayer for inference only
                import keras
                print(f"Warning: Model {model_path} is in legacy format. Creating a new model instead.")
                print("For inference with old models, you may need to convert them or retrain.")
                # Return a new model since the old format is incompatible
                return self._model()
        
        # Check for path with .keras extension added
        raise FileNotFoundError(f"Model not found: {model_path} or {keras_path}")
