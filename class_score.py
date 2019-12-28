import re
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np; np.random.seed(0)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def main(opts):
    speaker = []
    class_ = []
    acierto = 0
    fallo = 0

    results = open(opts.r)
    for line in results:
        temp = re.split("\t",line)    
        class_.append(temp[1][3:-1])
        temp2 = re.split("/",temp[0])
        speaker.append(temp2[1][3:])

    print(classification_report(speaker, class_))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NN in classification')   
    parser.add_argument('--r', type=str,
                        help='classification result file') 

    opts = parser.parse_args()
    if opts.r is None:
        raise ValueError('Result file not specified!')
    main(opts)


