import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
from time import time
import auxil.save_report as save_report
import numpy as np
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
import sys
import warnings
warnings.filterwarnings('always')


def set_params(args):
    if args.dataset in ["IP", "DIP", "DIPr", "KSC"]:
        args.C = 1
    elif args.dataset in ["UP", "DUP", "DUPr"]:
        args.C = 10
    elif args.dataset == "SV":
        args.C = 10
    elif args.dataset == "UH":
        args.C = 83  # best result C=1e5 and preprocess minmax
    else:
        print("ERROR params")
        exit()
    return args


def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr", "KSC"],
        help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr, KSC)')

    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components',
                        default=None,
                        type=int,
                        help='dimensionality reduction')
    parser.add_argument('--preprocess',
                        default="standard",
                        type=str,
                        help='Preprocessing')
    parser.add_argument('--splitmethod',
                        default="sklearn",
                        type=str,
                        help='Method for split datasets')
    parser.add_argument(
        '--random_state',
        default=None,
        type=int,
        help=
        'The seed of the pseudo random number generator to use when shuffling the data'
    )
    parser.add_argument('--tr_percent',
                        default=0.15,
                        type=float,
                        help='samples of train set')
    #########################################
    parser.add_argument('--set_parameters',
                        action='store_false',
                        help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--C',
                        default=1,
                        type=int,
                        help='Inverse of regularization strength')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    if args.set_parameters:
        args = set_params(args)

    pixels, labels, num_class = mydata.loadData(args.dataset,
                                                num_components=args.components,
                                                preprocessing=args.preprocess)
    pixels = pixels.reshape(-1, pixels.shape[-1])
    pixels_bak = np.copy(pixels)
    labels_bak = np.copy(labels)

    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((args.repeat, num_class))
    for pos in range(args.repeat):
        pixels = pixels_bak
        labels = labels_bak
        if args.dataset in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:
            x_train, x_test, y_train, y_test = mydata.load_split_data_fix(
                args.dataset, pixels)  #, rand_state=args.random_state+pos)
        else:
            labels = labels.reshape(-1)
            pixels = pixels[labels != 0]
            labels = labels[labels != 0] - 1
            rstate = args.random_state + pos if args.random_state is not None else None
            x_train, x_test, y_train, y_test = mydata.split_data(
                pixels, labels, args.tr_percent, rand_state=rstate)
        tic1 = time()
        clf = LogisticRegression(penalty='l2',
                                 dual=False,
                                 tol=1e-20,
                                 C=args.C,
                                 solver='lbfgs',
                                 multi_class='multinomial',
                                 max_iter=5000,
                                 fit_intercept=True,
                                 n_jobs=-1).fit(x_train, y_train)
        toc1 = time()
        tic2 = time()
        _, _, overall_acc, average_acc, kappa, each_acc = mymetrics.reports(
            clf.predict(x_test), y_test)
        toc2 = time()

        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[pos, :] = each_acc

    print(OA, AA, KAPPA)
    save_report.record_output(
        save_report.args_to_text(vars(args)), OA, AA, KAPPA, ELEMENT_ACC,
        TRAINING_TIME, TESTING_TIME, './report/' + 'logistic_regression_' +
        args.dataset + '_' + str(args.tr_percent) + '.txt')


if __name__ == '__main__':
    main()
