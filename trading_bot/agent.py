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


# ═══════════════════════════════════════════════════════════════════════════════
#  NETRUNNER TRADING DAEMON v2.077
#  "Wake up, Samurai. We have markets to burn." - Johnny Silverhand
# ═══════════════════════════════════════════════════════════════════════════════


@keras.saving.register_keras_serializable()
def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber Loss - Cyberware Loss Function for Q-Learning
    
    Smooths the gradient when error is large - like Sandevistan for your neural updates.
    
    Links:  https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ 
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  NETRUNNER TRADING DAEMON                                             ║
    ║  Night City's Premier Market Exploitation System                      ║
    ║  "In Night City, you either extract eddies or get flatlined."         ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, state_size, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy

        # ══════════════════════════════════════════════════════════════════
        # DAEMON CONFIGURATION - Neural Architecture Specs
        # ══════════════════════════════════════════════════════════════════
        self.state_size = state_size        # Normalized market observation window
        self.action_size = 3                # [HOLD, BUY, SELL] - Combat actions
        self.model_name = model_name
        self.inventory = []                 # Active positions (chrome in the bank)
        self.memory = deque(maxlen=10000)   # Experience buffer (braindance recordings)
        self.first_iter = True

        # ══════════════════════════════════════════════════════════════════
        # NEURAL IMPLANT CONFIGURATION - Learning Parameters
        # ══════════════════════════════════════════════════════════════════
        self.model_name = model_name
        self.gamma = 0.95                   # Long-term reward affinity (future eddies)
        self.epsilon = 1.0                  # Exploration rate (street chaos factor)
        self.epsilon_min = 0.01             # Minimum chaos threshold
        self.epsilon_decay = 0.995          # Chaos decay per cycle
        self.learning_rate = 0.001          # Neural plasticity coefficient
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # ══════════════════════════════════════════════════════════════════
        # STRATEGY CONFIGURATION - Combat Protocol Selection
        # ══════════════════════════════════════════════════════════════════
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # Target network (shadow neural backup)
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Constructs the Neural Cyberware Architecture
        
        Four-layer deep network — premium Arasaka-grade chrome.
        """
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(units=128, activation="relu"),   # Input cortex
            Dense(units=256, activation="relu"),   # Processing layer 1
            Dense(units=256, activation="relu"),   # Processing layer 2
            Dense(units=128, activation="relu"),   # Decision cortex
            Dense(units=self.action_size)          # Action output
        ])

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Records experience to memory — Braindance recording protocol
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Execute combat action based on current market state
        
        Actions: 0=HOLD, 1=BUY, 2=SELL
        """
        # Chaos mode: Random action to diversify experience (street samurai style)
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1  # First move: Always BUY (gotta start somewhere, choom)

        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Neural Training Protocol — Experience Replay
        
        Replays braindance recordings to optimize neural weights.
        Uses batch prediction for overclocked performance.
        """
        mini_batch = random.sample(self.memory, batch_size)
        
        # Prepare batch data (loading braindance sequence)
        states = np.array([sample[0][0] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        actions = [sample[1] for sample in mini_batch]
        rewards = [sample[2] for sample in mini_batch]
        dones = [sample[4] for sample in mini_batch]
        
        # ══════════════════════════════════════════════════════════════════
        # VANILLA DQN — Basic Neural Protocol
        # ══════════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════════
        # T-DQN — Fixed Target Protocol (Stabilized Targeting System)
        # ══════════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════════
        # DOUBLE DQN — Dual-Core Neural Protocol
        # ══════════════════════════════════════════════════════════════════
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())
            
            # Batch predict from both networks (dual cortex processing)
            q_values = self.model.predict(states, verbose=0)
            next_q_values_model = self.model.predict(next_states, verbose=0)
            next_q_values_target = self.target_model.predict(next_states, verbose=0)
            
            for i in range(batch_size):
                if dones[i]:
                    target = rewards[i]
                else:
                    # Use primary model to select action, target to evaluate
                    best_action = np.argmax(next_q_values_model[i])
                    target = rewards[i] + self.gamma * next_q_values_target[i][best_action]
                q_values[i][actions[i]] = target
            
            X_train = states
            y_train = q_values
                
        else:
            raise NotImplementedError()

        # Update Q-function parameters based on Huber loss gradient
        loss = self.model.fit(
            X_train, y_train,
            epochs=1, verbose=0
        ).history["loss"][0]

        # Decay chaos factor — agent becomes more calculated over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        """Save neural shard in Keras 3 format (.keras extension)"""
        filepath = "models/{}_{}.keras".format(self.model_name, episode)
        self.model.save(filepath)

    def load(self):
        """Load neural shard — supports both new .keras and legacy formats"""
        model_path = "models/" + self.model_name
        
        # Try new .keras format first (current Arasaka standard)
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
                # Try loading as legacy H5 (old chrome)
                return load_model(model_path, custom_objects=self.custom_objects)
            except ValueError:
                # If legacy format fails, try using TFSMLayer for inference only
                import keras
                print(f"⚠️  WARNING: Model {model_path} is legacy format. Initializing fresh neural shard.")
                print("For inference with old chrome, you may need to convert or retrain.")
                # Return a new model since the old format is incompatible
                return self._model()
        
        # Check for path with .keras extension added
        raise FileNotFoundError(f"💀 FLATLINE: Neural shard not found: {model_path} or {keras_path}")
