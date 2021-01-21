from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import basename, dirname, join, isdir

import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from pycm import ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def traverse(directory):
    results = []

    for d in listdir(directory):
        obj_path = join(directory, d)
        if isdir(obj_path):
            results.extend(traverse(obj_path))
        else:
            results.append(obj_path)

    return results


def save_html(confusion_matrix_folder, ground_truths, predictions, model):
    html_folder = join(confusion_matrix_folder, 'html')
    makedirs(html_folder, exist_ok=True)

    cm = ConfusionMatrix(actual_vector=ground_truths, predict_vector=predictions)

    filename = f'{model}.html'
    cm.save_html(join(html_folder, filename), summary=True)


def save_png(confusion_matrix_folder, ground_truths, predictions, model):
    png_folder = join(confusion_matrix_folder, 'png')
    makedirs(png_folder, exist_ok=True)

    CM = confusion_matrix(ground_truths, predictions)

    plt.figure(dpi=1200)
    plt.matshow(CM, cmap='GnBu')
    plt.savefig(join(png_folder, f'{model}.png'), dpi=1200)


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', required=True, type=str, help='Where to find results files to be reported')
    parser.add_argument('-f', '--file', default=None, help='Where to save report file (as CSV)')
    parser.add_argument('-cm', '--confusion_matrix', default=None, help='Where to save confusion matrices')
    args = parser.parse_args()
    root_directory = str(args.dir)
    save_file = args.file
    conf_mat_path = args.confusion_matrix

    files = traverse(root_directory)

    scores = {}
    for file in files:
        loaded = torch.load(file)
        model_name = f'{basename(dirname(file))}-{basename(file).split(".")[0].split("_results")[0]}'

        scores[model_name] = [
            accuracy_score(loaded['ground_truths'], loaded['predictions']) * 100,
            precision_score(loaded['ground_truths'], loaded['predictions'], average='macro') * 100,
            recall_score(loaded['ground_truths'], loaded['predictions'], average='macro') * 100,
            f1_score(loaded['ground_truths'], loaded['predictions'], average='macro') * 100
        ]

        if conf_mat_path:
            save_html(conf_mat_path, loaded['ground_truths'], loaded['predictions'], model_name)
            save_png(conf_mat_path, loaded['ground_truths'], loaded['predictions'], model_name)

    df_report = DataFrame.from_dict(scores, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    print(df_report)

    if save_file:
        if not save_file.lower().endswith('csv'):
            save_file = f'{save_file.split(".")[0]}.csv'
        df_report.to_csv(save_file, index_label='Model')
