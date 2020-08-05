import argparse
import sys
import warnings
from time import time

import numpy as np
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import auxil.save_report as save_report

warnings.filterwarnings('always')


def set_params(args):
    if args.dataset in ["IP", "DIP", "DIPr", "KSC"]:
        args.n_est = 200
        args.m_s_split = 3
        args.max_feat = 10
        args.depth = 10
    elif args.dataset in ["UP", "DUP", "DUPr"]:
        args.n_est = 200
        args.m_s_split = 2
        args.max_feat = 40
        args.depth = 60
    elif args.dataset == "SV":
        args.n_est = 200
        args.m_s_split = 2
        args.max_feat = 10
        args.depth = 10
    elif args.dataset == "UH":
        args.n_est = 200
        args.m_s_split = 2
        args.max_feat = 8
        args.depth = 20
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
    parser.add_argument('--n_est',
                        default=200,
                        type=int,
                        help='The number of trees in the forest')
    parser.add_argument(
        '--m_s_split',
        default=2,
        type=int,
        help='The minimum number of samples required to split an internal node'
    )
    parser.add_argument(
        '--max_feat',
        default=40,
        type=int,
        help=
        'The number of features to consider when looking for the best split')
    parser.add_argument('--depth',
                        default=60,
                        type=int,
                        help='The maximum depth of the tree')
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
            rstate = args.random_state + pos if args.random_state != None else None
            x_train, x_test, y_train, y_test = mydata.split_data(
                pixels, labels, args.tr_percent, rand_state=rstate)
        tic1 = time()
        clf = RandomForestClassifier(n_estimators=args.n_est,
                                     min_samples_split=args.m_s_split,
                                     max_features=args.max_feat,
                                     max_depth=args.depth).fit(
                                         x_train, y_train)
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
        TRAINING_TIME, TESTING_TIME, './report/' + 'random_forest_' +
        args.dataset + '_' + str(args.tr_percent) + '.txt')


if __name__ == '__main__':
    main()
