from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

MODELS_NAME = ['Classical', 'SOTA', 'Ours']
MODELS_COLORS = ['r', 'b', 'g']
WHICH_MODEL = 2
TOP_FILEPATH ='/Models/FashionMNIST/'

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

def compare_models(var_name, trials):
    #some of the naming is weird sorry
    plt.figure()
    for i in range(len(MODELS_NAME)):
        all_res = []
        WHICH_MODEL = i
        for j in range(trials):
            summary_filepath = TOP_FILEPATH+ MODELS_NAME[WHICH_MODEL]\
                + '/Trial'+ str(j)+'/Summaries/' #filepaths to model and summaries
            print(summary_filepath)
            w_times, step_nums, vals = return_values(summary_filepath, var_name)
            all_res.append(vals)
        for val in all_res:
            print(len(val))
        all_res = np.array(all_res)
        mean_result = np.mean(all_res, axis = 0)
        plt.plot(step_nums,mean_result, MODELS_COLORS[i])
    plt.legend(MODELS_NAME)
    plt.show()

def show_trials(var_name, trials, which_model):
    plt.figure()
    for j in range(trials):
        summary_filepath = TOP_FILEPATH+ MODELS_NAME[which_model]\
            + '/Trial'+ str(j)+'/Summaries/' #filepaths to model and summaries
        print(summary_filepath)
        w_times, step_nums, vals = return_values(summary_filepath, var_name)
        plt.plot(step_nums,vals)
    plt.title(MODELS_NAME[which_model])
    plt.show()


if __name__ == '__main__':
    # list_var_names('/Models/FashionMNIST/'+ MODELS_NAME[0] + '/Summaries/')
    # compare_models('logistics/train_accuracy')
    # show_trials('logistics/train_accuracy', 5, 0)
    # show_trials('logistics/train_accuracy', 5, 1)
    # show_trials('logistics/train_accuracy', 5, 2)
    compare_models('logistics/train_accuracy', 5)


# Show all tags in the log file


# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
