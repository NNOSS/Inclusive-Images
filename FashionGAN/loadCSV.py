import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv

FILEPATH = '/Data/Open/CSVData/'
CSV_FILENAME = FILEPATH + 'train_labels.csv'
batch_size= 1000000

def load_csv(CSV_FILENAME, batch_size):
    file_dict = {}
    types_labels = set()
    with open(CSV_FILENAME) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are' + ", ".join(row))
                line_count += 1
            else:
                image_name = row[0]
                label = row[2]
                confidence = float(row[3])
                if confidence != 1:
                    continue
                else:
                    line_count += 1

                if image_name in file_dict:
                    file_dict[image_name].append(label)
                else:
                    file_dict[image_name] = [label]

                if label not in types_labels:
                    types_labels.add(label)
                if not line_count % batch_size:
                    yield file_dict
                    file_dict = {}


            # if not line_count%20:
            #     print('Line Count: ' + str(line_count))
    yield file_dict



if __name__ == '__main__':
    files_dict_gen = load_csv(CSV_FILENAME, batch_size)
    i = 1
    iterations = 0
    while i is not None:
        i = next(files_dict_gen, None)
        iterations += 1
        print(iterations)
