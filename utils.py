from sklearn import preprocessing
import joblib
import numpy as np
import csv


def load_data(file_path, mode):
    input_data = []
    output_classes = []
    with open(file_path, 'r', newline='') as csvfile:
        log_reader = csv.reader(csvfile, delimiter=',')
        next(log_reader)
        for row in log_reader:
            if mode == "train":
                input_data.append(row[:-1])
                output_classes.append(row[-1])
            else:
                input_data.append(row)
    input_data = np.array(input_data, dtype=int)
    return input_data, output_classes


def save_result(input_data, result, result_file):
    data = [row for row in zip(input_data, result)]
    np.savetxt(result_file, data, delimiter=',', fmt='%s')


def pre_processing(output_classes):
    encoder = preprocessing.LabelEncoder()
    output_classes = encoder.fit_transform(output_classes).astype(int)  # 0: anomaly  1: normal
    return encoder, output_classes


def save_components(model, pca, encoder, components_file):
    joblib.dump([model, pca, encoder], components_file)


def load_components(components_file):
    model, pca, encoder = joblib.load(components_file)
    return model, pca, encoder


