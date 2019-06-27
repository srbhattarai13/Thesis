import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_accuracies =   event_acc.Scalars('train_loss_shanghaitech')
    # validation_accuracies = event_acc.Scalars('validation_accuracy')

    steps = 100
    x = np.zeros([steps, 2])
    y = np.arange(steps)

    for i in range(steps):
        x[i, 0] = training_accuracies[i][2] # value
        # y[i, 1] = validation_accuracies[i][2]

    plt.plot(x[:,0], y, label='shanghaitech training accuracy')
    # plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    log_file = "logs/shanghaitech/train_64_0.0001_fnet_shaTech_chck/events.out.tfevents.1558938545.cvg21"
    plot_tensorflow_log(log_file)
