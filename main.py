import timeit
import sys
import os

from utils import load_data, pre_processing, save_components, load_components, save_result
from classifier import RF_run, RF_test


def main(mode):
    components_file = "RF_components.pkl"
    result_file = "RF_result.csv"
    file_path = input("Please enter the path to your csv-file \n")
    name, extension = os.path.splitext(file_path)
    if extension == '.csv':
        start = timeit.default_timer()
        input_data, labels = load_data(file_path, mode)
        if mode == "train":
            encoder, labels = pre_processing(labels)
            model, pca = RF_run(input_data, labels, k_fold=10, n_components=5, n_models=10)
            save_components(model, pca, encoder, components_file)
        elif mode == "predict":
            model, pca, encoder = load_components(components_file)
            result = RF_test(model, pca, input_data)
            result = encoder.inverse_transform(result)
            save_result(input_data, result, result_file)
        stop = timeit.default_timer()
        print("Run Time: %.2f" % (stop - start))
    else:
        print("File appears not to be in CSV format")
        exit(-1)


if len(sys.argv) == 2 and sys.argv[1] in ["train", "predict"]:
    main(sys.argv[1])
else:
    print('Please run the code in the following format: main.py  [train/predict]')
    exit(-1)
