import evaluation
import training


def full_cycle():
    filename = input('file name (with .csv)')
    clf = training.train_model(filename)
    auroc = evaluation.perform_evaluation(filename, clf)
    print('AUROC: ', auroc)

