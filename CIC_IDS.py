import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from Environment import IntrusionEnv
from Auxilary_Functions import Dataset, dqn, test_metrics
import warnings
warnings.filterwarnings("ignore")


path = '/home/art/proposal/darabi/IDS-RL-Code/'
test_acc = []
action_size = 7
state_size = 79
number_of_iteration = 5
dataset = path + 'Datasets/CIC_IDS2018/normalized_CIC_IDS.csv'
target_names = ['Benign', 'DDoS', 'DoS', 'Bot', 'BruteForce', 'Infilteration', 'Attack']


for i in range(number_of_iteration):
    print(f"FOLD {i+1}\n")
    start_time = time.time()
    # import dataset
    x_train, y_train, x_test, y_test = Dataset(dataset)
    test_len = len(y_test)-2
    train_len = len(y_train)-2
    print(f"train len: {train_len},   test_len: {test_len}")

    # train model
    env_train = IntrusionEnv(x_train, y_train, len=47000  , n_action=action_size, n_state=state_size, random=True)

    dqn(env_train, max_t=47000)
    end_time = time.time()
    print(f"process time of fold {i + 1} = {end_time - start_time}")

    # evluate test set
    env_test = IntrusionEnv(x_test, y_test, len=test_len, n_action=action_size, n_state=state_size, random=False)

    # evaluate test set
    print("Test Results:")
    test_matrix, test_report, test_accuracy = test_metrics(env_test, y_test, test_len, target_names)
    df_test_matrix = pd.DataFrame(test_matrix, columns=target_names, index=target_names)
    test_acc.append(test_accuracy)

    print(f"test report:\n{test_report}")

    ax = sn.heatmap(df_test_matrix, annot=True, cmap="YlGnBu")
    plt.tight_layout()
    test_report.to_csv(f"{path}/Results/CIC_IDS2018/classification_report/Fold_{i+1}.csv", index=True, header=True)
    plt.savefig(f"{path}/Results/CIC_IDS2018/confusion_matrix/Fold_{i+1}.png")
    df_test_matrix.to_csv(f"{path}/Results/CIC_IDS2018/confusion_matrix/Fold_{i+1}.csv", index=True, header=True)
    plt.show()


    print("\n---------------------------------------------------\n")