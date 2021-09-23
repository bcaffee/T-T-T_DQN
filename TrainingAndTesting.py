import os
from TicTacToeEnvironment import TicTacToeEnvironment
from Agent import Agent
import pandas as pd
import numpy as np
import keras
from keras import layers


def main():
    debugging = False

    # Set up the agent in the training and testing environment
    agent = Agent(TicTacToeEnvironment, random_aspect=0.1, discount=0.9, agent_tag='X')

    # Q-Learning agent starts to train

    # print("Initiated learning stats:")
    # agent.demoGameStats()

    agent.learn_game(1000)
    print("After 1,000 learning games...")

    agent.learn_game(5000)
    print("After 5,000 learning games...")

    agent.learn_game(10000)
    print("After 10,000 learning games...")

    agent.learn_game(20000)
    print("After 20,000 learning games...")

    agent.learn_game(40000)
    print("After 40,000 learning games...")

    agent.learn_game(80000)
    print("After 80,000 learning games...")

    # agent.learnGame(100000)
    # print("After 100,000 learning games...")
    # agent.demoGameStats() # (Takes a really long time because it uses the neural network)

    agent.round_v()

    # Create a CSV file from the Q-Learning train data to feed into neural network
    # Note: Instead of using a CSV file I can directly get the state/reward information from the agent's dictionary
    agent.save_v_table()

    # Reading CSV file and converting it to a Pandas DataFrame
    data_frame = pd.read_csv("state_values.csv")
    data_frame = data_frame.sample(frac=1)

    if debugging:
        print("\n")
        print("DataFrame of state_values.csv: ")
        print("\n")
        print(data_frame)

    # Create numpy array from Pandas DataFrame for better accessibility
    input_board_states = np.array([data_frame['TopLeft'], data_frame['TopMid'],
                                   data_frame['TopRight'], data_frame['MidLeft'], data_frame['Center'],
                                   data_frame['MidRight'], data_frame['BottomLeft'],
                                   data_frame['BottomMid'], data_frame['BottomRight']])

    actual_rewards = np.array([data_frame['RewardValue']])

    # Transpose to improve accessibility
    input_board_states = np.transpose(input_board_states)
    actual_rewards = np.transpose(actual_rewards)

    # Create neural network and pass in board states (keys) from CSV file

    model = keras.Sequential([
        # 9 neurons each
        layers.Dense(36, activation=keras.activations.linear, input_shape=[9]),  # Input shape defines input layer
        layers.Dropout(0.1),
        layers.Dense(36, activation=keras.activations.linear),
        layers.Dropout(0.1),
        layers.Dense(36, activation=keras.activations.linear),
        layers.Dropout(0.1),
        layers.Dense(36, activation=keras.activations.linear),
        layers.Dropout(0.1),
        layers.Dense(36, activation=keras.activations.linear),
        layers.Dropout(0.1),
        layers.Dense(36, activation=keras.activations.linear),
        layers.Dropout(0.1),
        layers.Dense(1)  # Output layer
    ])

    optimizer = keras.optimizers.Adam(0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

    # Fits the train method input to correspond  the specific output
    model.fit(input_board_states, actual_rewards, epochs=10, batch_size=64, shuffle=True)

    path = "./CSV_QL_Data/"

    if not os.path.exists(path):
        os.makedirs(path)

    model.save(path + "TicTacToeModel.h5")

    '''
    initializer = tf.contrib.layers.xavier_initializer()

    # Get the input for the neural net
    input_board_states = boardStateRewardMatrix[0]

    if debugging:
        print(input_board_states)

    hiddenLayer = tf.layers.dense(input_board_states, 8, activation=tf.nn.relu, kernel_initializer=initializer)
    hiddenLayer2 = tf.layers.dense(hiddenLayer, 8, activation=tf.nn.relu, kernel_initializer=initializer)

    # Output of neural net
    outputLayer = tf.layers.dense(hiddenLayer2, activation=None)

    rewardGuesses = tf.nn.softmax(outputLayer)

    if debugging:
        print(rewardGuesses)

    # Get actual values
    actual_rewards = boardStateRewardMatrix[1]

    if debugging:
        print(actual_rewards)

    # Calculate error for optimizer
    inputs = tf.placeholder(dtype=tf.float32, shape=None)
    guessedOutputs = tf.placeholder(dtype=tf.float32, shape=None)
    actualOutputs = tf.placeholder(dtype=tf.float32, shape=None)

    error = tf.square(guessedOutputs - actualOutputs)
    
    errors = []

    for i in range(len(rewardGuesses)):
        errors[i] = math.pow((rewardGuesses[i] - actual_rewards[i]), 2)

    loss = tf.reduce_mean(error)

    # Weight that gets changed over the training process
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-3)

    # Storing the optimizer.minimize(loss) call in a variable
    train = optimizer.minimize(loss)

    # TODO: Create training loop

    tf.reset_default_graph()

    loadFromCheckpoint = False

    # Saving the neural network train data into a folder

    path = "./CSV_QL_Data/"

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=3)

    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        sess.run(init)

        if loadFromCheckpoint:
            checkpoint = tf.train.get_checkpoint_state(path)
            saver.restore(sess, checkpoint.model_checkpoint_path)

        else:
            tf.global_variables_initializer()

        tf.local_variables_initializer()

        # Total # of possible board states: 19,683
        for i in range(5000):
            sess.run(train, feed_dict={inputs: input_board_states, actualOutputs: actual_rewards})

        sess.close()
'''


if __name__ == '__main__':
    main()
