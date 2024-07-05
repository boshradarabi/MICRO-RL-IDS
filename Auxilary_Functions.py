import torch
import pandas as pd
from Agent import Agent
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import RocCurveDisplay


path = '/home/art/proposal/darabi/IDS-RL-Code/'

def Dataset(dataset):
    df = pd.read_csv(dataset)
    df = df.sample(frac=1)
    print(df.head(5))
    print(df.tail(5))

    # split dataset into train and test set
    train_df, test_df = train_test_split(df, train_size=0.8)

    # seperate data and labels
    x_train_pd = train_df.iloc[:, :-1]
    y_train_pd = train_df.iloc[:, -1:]
    x_test_pd = test_df.iloc[:, :-1]
    y_test_pd = test_df.iloc[:, -1:]

    # convert to a numpy array
    x_train = x_train_pd.to_numpy()
    y_train = y_train_pd.to_numpy()
    x_test = x_test_pd.to_numpy()
    y_test = y_test_pd.to_numpy()

    return x_train, y_train, x_test, y_test



def dqn(env, n_episodes= 250, max_t = 200,
        eps_start=1.0, eps_end = 0.001,eps_decay=0.9996):
    """Deep Q-Learning
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
    """

    env_train = env
    state_size = env_train.observation_space.shape[0]
    action_size = env_train.action_space.n
    agent = Agent(state_size,action_size,seed=0)
    scores = [] # list containing score from each episode
    eps = eps_start
    score_max_all=0

    temp1 = 0
    for i_episode in range(1, n_episodes+1):
        print(f"episode: {i_episode}")
        state = env_train.reset()
        score = 0
        score_cnt=0
        score_max=0

        for t in range(max_t):
            # print("t:  ", t)
            temp1 = t
            action = agent.act(state,eps)
            next_state,reward,done,_ = env_train.step(action)
            agent.step(state,action,reward,next_state,done)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our
            ## replay buffer.
            state = next_state
            score += reward
            if reward==1:
              score_cnt+=1
            else:
              score_cnt=0
            # scores.append(score) ## see the most recent score
            eps = max(eps*eps_decay,eps_end)## decrease the epsilon

            if t==299 and i_episode%50==0:
              print('\rEpisode {}\t Score_max {:.2f}   Score {:.2f}   epsilon {:.2f}'.format(i_episode,score_max,score, eps))
            
            if score_cnt>score_max:
                score_max=score_cnt

            if score_cnt >= score_max_all:
                score_max_all=score_cnt
                torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')

            if done:
                break
        scores.append(score)

        # print(f"final t: {temp1}")
        # print(f"episode * t = {i_episode} * {temp1} = {i_episode*temp1}")
    
    scores1 = np.array(scores)
    min_val = np.min(scores1)
    max_val = np.max(scores1)
    normal_reward = (scores1-min_val) / (max_val-min_val) * (1 - (-1)) + (-1)

    plt.plot(np.arange(len(scores)), normal_reward)
    plt.ylabel('normalized reward')
    plt.xlabel('episode')
    plt.savefig(f"{path}/Results/NSL_KDD/diagram.png")

    plt.show()

    return scores


y_pred = []
def evaluate(model, env, y_pred):
    """custom evaluate function"""
    attempts, correct = 0, 0
    state = env.reset()
    done = False

    while not done:
        action = model.act(state)
        next_state, reward, done, _ = env.step(action)
        y_pred.append(action)
        attempts += 1
        if reward > 0:
            correct += 1
        state = next_state

    accuracy = float(correct) / attempts
    print('Accuracy:', accuracy, '   correct:', correct,'   attempts:', attempts)
    return accuracy



def get_classification_report(y_set, y_pred, target_names):
    y_data = y_set
    y_predict = y_pred
    path = '/home/art/proposal/darabi/IDS-RL-Code/'
    np.savetxt('/home/art/proposal/darabi/IDS-RL-Code/Results/y_set.csv', y_data, delimiter=',')
    np.savetxt('/home/art/proposal/darabi/IDS-RL-Code/Results/y_pred.csv', y_data, delimiter=',')

    report = metrics.classification_report(y_set, y_pred, output_dict=True, target_names=target_names)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report



def test_metrics(env_test, y_test, test_len, target_names):
    y_pred = []
    # Initialize the environment with the test dataset
    env = env_test
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load the trained model
    agent = Agent(state_size,action_size,seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    # Evaluate the model
    accuracy = evaluate(agent, env, y_pred)
    matrix = confusion_matrix(y_test[:test_len], y_pred[:test_len])
    report = get_classification_report(y_test[:test_len], y_pred[:test_len], target_names)

    # fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    # roc_auc = metrics.auc(fpr, tpr)
    
    path = '/home/art/proposal/darabi/IDS-RL-Code/'
    

    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig(f"{path}/Results/ROC.png")
    # plt.show()


    return matrix, report, accuracy