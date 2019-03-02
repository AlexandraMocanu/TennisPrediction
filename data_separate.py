from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

ATP_file_features = 'D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_atp_modified_features.csv'
ATP_file_labels = 'D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_atp_modified_labels.csv'

global ATP_features_train, ATP_features_test, ATP_labels_train, ATP_labels_test

def separate_atp():
    features = pd.read_csv(ATP_file_features, encoding = "ISO-8859-1", dtype={'tourney_id': np.str, 'player2_id': np.str,
                                                                                'surface': np.str, 'draw_size': np.str})
    labels = pd.read_csv(ATP_file_labels, encoding = "ISO-8859-1", dtype={'tourney_id': np.str, 'player2_id': np.str,
                                                                            'surface': np.str, 'draw_size': np.str})

    global ATP_features_train, ATP_features_test, ATP_labels_train, ATP_labels_test
    ATP_features_train, ATP_features_test, ATP_labels_train, ATP_labels_test = train_test_split(features, labels, test_size=0.15)
    
    ATP_features_train.fillna(0, inplace=True)
    ATP_features_test.fillna(0, inplace=True)

    # print('\nTrain:')
    # print('\n\tfeatures=')
    # print(ATP_features_train.shape)
    # print('\n\t')
    # # print(ATP_features_train.head())
    # print('\n\tlabels=')
    # print(ATP_labels_train.shape)
    # print('\n\t')
    # # print(ATP_labels_train.head())
    # print('\n\nTest:')
    # print('\n\tfeatures=')
    # print(ATP_features_test.shape)
    # print('\n\t')
    # # print(ATP_features_test.head())
    # print('\n\tlabels=')
    # print(ATP_labels_test.shape)
    # print('\n\t')
    # # print(ATP_labels_test.head())


def to_categorical_atp():
    return


def main():
    separate_atp()
    # separate_wta()
    # to_categorical_atp()


if __name__ == "__main__":
    main()