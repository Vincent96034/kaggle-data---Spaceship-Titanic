from sklearn.metrics import classification_report
from sklearn import metrics
import json




def model_report(model, y_pred, y_val, print_out = True, fname=None):
    '''
    Function that documents the model specs and its performance.
    Data is returned as a dictionary and optionally saved as a .json file.
    Args:
        model: model estimator object
        y_pred: predicted target values
        y_val: validation target values
        print_out (bool): if true, acc, pred, recall and f1 score are printed out
        fname (str): Filename of output file; If not None, json file containing model report is created
    '''

    output_dict = {}
    output_dict["model"] = type(model).__name__
    output_dict["model_config"] = model.get_params()
    output_dict["model_performance"] = classification_report(y_val, y_pred, output_dict = True)
    output_dict["model_performance"]["confusion_matrix"] = metrics.confusion_matrix(y_val, y_pred).tolist()

    if print_out:
        print("Accuracy: " + str(output_dict["model_performance"]['accuracy']))
        print("Precision: " + str(output_dict["model_performance"]['1']["precision"]))
        print("Recall: " + str(output_dict["model_performance"]['1']["recall"]))
        print("F1-Score: " + str(output_dict["model_performance"]['1']["f1-score"]))

    if fname != None:
        out_file = open(str(fname), "w")
        json.dump(output_dict, out_file, indent = 6) 
        out_file.close()
        print("Created .json file '" + str(fname) + "'")

    return output_dict