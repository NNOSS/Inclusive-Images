from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

MODELS_NAME = ['Classical', 'SOTA', 'Ours']
MODELS_COLORS = ['r', 'b', 'g']
WHICH_MODEL = 2
SUMMARY_FILEPATH ='/Models/FashionMNIST/'+ MODELS_NAME[WHICH_MODEL] + '/Summaries/'

def return_values(filepath, var_name):
    event_acc = EventAccumulator(filepath)
    event_acc.Reload()
    return zip(*event_acc.Scalars(var_name))

def list_var_names(filepath):
    event_acc = EventAccumulator(filepath)
    event_acc.Reload()
    tag_dict = event_acc.Tags()
    for i,v in tag_dict.items():
        print(i + ':')
        if isinstance(v, list):
            for val in v:
                print(val)
        else:
            print(v)
        print('------------')

def compare_models(var_name):
    plt.figure()
    for i in range(len(MODELS_NAME)):
        summary_filepath = '/Models/FashionMNIST/'+ MODELS_NAME[i] + '/Summaries/'
        w_times, step_nums, vals = return_values(summary_filepath, var_name)
        plt.plot(step_nums,vals, MODELS_COLORS[i])
    plt.legend(MODELS_NAME)
    plt.show()

if __name__ == '__main__':
    # list_var_names('/Models/FashionMNIST/'+ MODELS_NAME[0] + '/Summaries/')
    compare_models('logistics/train_accuracy')

# Show all tags in the log file


# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
