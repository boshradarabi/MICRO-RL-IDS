import pandas as pd
import time
import seaborn as sn
import matplotlib.pyplot as plt
from Environment import IntrusionEnv
from Auxilary_Functions import Dataset, dqn, test_metrics
import warnings
warnings.filterwarnings("ignore")



path = '/home/art/proposal/darabi/IDS-RL-Code/'
test_acc = []
action_size = 5
state_size = 122
number_of_iteration = 5
dataset = path + 'Datasets/NSL_KDD/Normalized_NSL_KDD.csv'
target_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


for i in range(number_of_iteration):
    start_time = time.time()

    cfm_plot = None
    print(f"FOLD {i+1}\n")
    # import dataset
    x_train, y_train, x_test, y_test = Dataset(dataset)
    test_len = len(y_test)-2
    train_len = len(y_train)-2
    print(f"train len: {train_len},   test_len: {test_len}")
    # train model
    env_train = IntrusionEnv(x_train, y_train, len=3000, n_action=action_size, n_state=state_size, random=True)

    dqn(env_train, max_t=3000)
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
    test_report.to_csv(f"{path}/Results/NSL_KDD/classification_report/Fold_{i+1}.csv", index=True, header=True)
    plt.savefig(f"{path}/Results/NSL_KDD/confusion_matrix/Fold_{i+1}.png")
    df_test_matrix.to_csv(f"{path}/Results/NSL_KDD/confusion_matrix/Fold_{i+1}.csv", index=True, header=True)
    plt.show()


    print("\n---------------------------------------------------\n")