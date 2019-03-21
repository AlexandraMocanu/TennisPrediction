import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline

ATP_file_features = 'D:\\EXPERT\\lab_se\\TennisPrediction\\combined_atp_modified_features.csv'
ATP_file_labels = 'D:\\EXPERT\\lab_se\\TennisPrediction\\combined_atp_modified_labels.csv'
WTA_file_features = 'D:\\EXPERT\\lab_se\\TennisPrediction\\combined_wta_modified_features.csv'
WTA_file_labels = 'D:\\EXPERT\\lab_se\\TennisPrediction\\combined_wta_modified_labels.csv'

RESULTS = 'D:\\EXPERT\\lab_se\\results'

global ATP_features_train, ATP_features_test, ATP_labels_train, ATP_labels_test
global WTA_features_train, WTA_features_test, WTA_labels_train, WTA_labels_test
global mean, std, cs, lb1, lb2, lb3, lb4, lb5, cs2

def separate_atp():
    features = pd.read_csv(ATP_file_features, encoding = "ISO-8859-1", 
                                            dtype={'tourney_id': np.str, 'year': np.str, 'surface': np.str, 'player2_id': np.str, 'draw_size': np.str}, 
                                            parse_dates=False)
    labels = pd.read_csv(ATP_file_labels, encoding = "ISO-8859-1", 
                                        dtype={'score': np.str, 'set1_w': np.str, 'set3_w': np.str, 'set3_l': np.str}, 
                                        parse_dates=False)

    features.fillna(0, inplace=True)
    labels.fillna(0, inplace=True)

    # features, labels = one_hot_encode_features(features, labels)
    features = scale_continous_discrete(features)
    # print(features[1])

    global ATP_features_train, ATP_features_test, ATP_labels_train, ATP_labels_test
    ATP_features_train, ATP_features_test, ATP_labels_train, ATP_labels_test = train_test_split(features, labels, test_size=0.1)

    # normalize_atp()

    eval_shapes_atp()


def separate_wta():
    features = pd.read_csv(WTA_file_features, encoding = "ISO-8859-1", 
                                            dtype={'tourney_id': np.str, 'year': np.str, 'surface': np.str, 
                                            'player2_id': np.str, 'draw_size': np.str, 'player2_seed': np.str,
                                            'player1_seed': np.str}, 
                                            parse_dates=False)
    labels = pd.read_csv(WTA_file_labels, encoding = "ISO-8859-1", 
                                        dtype={'score': np.str, 'set1_w': np.str, 'set3_w': np.str, 'set3_l': np.str}, 
                                        parse_dates=False)

    features.fillna(0, inplace=True)
    labels.fillna(0, inplace=True)

    # features, labels = one_hot_encode_features(features, labels)
    features = scale_continous_discrete(features)
    # print(features[1])

    global WTA_features_train, WTA_features_test, WTA_labels_train, WTA_labels_test
    WTA_features_train, WTA_features_test, WTA_labels_train, WTA_labels_test = train_test_split(features, labels, test_size=0.15)

    # normalize_wta()

    eval_shapes_wta()


def scale_continous_discrete(features):
    # [0,1]
        #['year', 'tourney_id', 'player2_id', 'player1_id',
        # 'surface', 'draw_size', 'best_of', 'player2_seed', 'player2_hand', 'player2_ht',
        # 'player2_age', 'player2_rank', 'player1_seed', 'player1_hand', 'player1_ht', 
        # 'player1_age', 'player1_rank']
        # 
        # discrete features: surface, draw_size, best_of, player2_hand, player1_hand,
        # continous features: year, tourney_id, player2_id, player1_id, player2_seed, player2_ht, 
        #                       player2_age, player2_rank, player1_seed, player1_ht, player1_age, player1_rank

    # set_C = "-QWERTYUIOPASDFGHJKLZXCVBNM"
    # for index, row in features.iterrows():
    #     id = row['tourney_id']
    #     for c in set_C:
    #         if c in id:
    #             id = id.split(c)[0] + id.split(c)[1]
    #     # print(id)
    #     # if 'Q' in id:
    #     #     print(id)
    #     features.set_value(index, 'tourney_id', id)

    for index, row in features.iterrows():
        r1 = str(row['player1_seed'])
        r2 = str(row['player2_seed'])
        if 'Q' in r1:
            features.set_value(index, 'player1_seed', 0)
        if 'Q' in r2:
            features.set_value(index, 'player2_seed', 0)

    # atp
    # cont = ['year', 'tourney_id', 'player2_id', 'player1_id', 'player2_seed', 'player2_ht',
    #             'player2_age', 'player2_rank', 'player1_seed', 'player1_ht', 'player1_age', 'player1_rank']
    
    # atp
    # cs = MinMaxScaler()
    # features_cont = cs.fit_transform(features[cont])
    
    ##
        # lb1 = LabelBinarizer()
        # draw_size_B = lb1.fit(features["draw_size"].astype(str))
        # lb2 = LabelBinarizer()
        # surface_B = lb2.fit(features["surface"].astype(str))
        # lb3 = LabelBinarizer()
        # player2_hand_B = lb3.fit(features["player2_hand"].astype(str))
        # lb4 = LabelBinarizer()
        # player1_hand_B = lb4.fit(features["player1_hand"].astype(str))
        # 
        # lb5 = LabelBinarizer()
        # best_of = lb5.fit(features["best_of"].astype(str))
        # set1_w = lb.fit(labels["set1_w"].astype(str))
        # set1_l = lb.fit(labels["set1_l"].astype(str))
        # set2_w = lb.fit(labels["set2_w"].astype(str))
        # set2_l = lb.fit(labels["set2_l"].astype(str))
        # set3_w = lb.fit(labels["set3_w"].astype(str))
        # set3_l = lb.fit(labels["set3_l"].astype(str))
        # set4_w = lb.fit(labels["set4_w"].astype(str))
        # set4_l = lb.fit(labels["set4_l"].astype(str))
        # set5_w = lb.fit(labels["set5_w"].astype(str))
        # set5_l = lb.fit(labels["set5_l"].astype(str))
        # t1 = lb.fit(labels["t1"].astype(str))
        # t2 = lb.fit(labels["t2"].astype(str))
        # t3 = lb.fit(labels["t3"].astype(str))
        # t4 = lb.fit(labels["t4"].astype(str))
        # t5 = lb.fit(labels["t5"].astype(str))
        # 
        # draw_size_trans = draw_size_B.transform(features["draw_size"].astype(str))
        # surface_trans = surface_B.transform(features["surface"].astype(str))
        # player2_hand_trans = player2_hand_B.transform(features["player2_hand"].astype(str))
        # player1_hand_trans = player1_hand_B.transform(features["player1_hand"].astype(str))
        # best_of_trans = best_of.transform(features["best_of"].astype(str))
        # set1_w_trans = set1_w.transform(labels["set1_w"].astype(str))
        # set1_l_trans = set1_l.transform(labels["set1_l"].astype(str))
        # set2_w_trans = set2_w.transform(labels["set2_w"].astype(str))
        # set2_l_trans = set2_l.transform(labels["set2_l"].astype(str))
        # set3_w_trans = set3_w.transform(labels["set3_w"].astype(str))
        # set3_l_trans = set3_l.transform(labels["set3_l"].astype(str))
        # set4_w_trans = set4_w.transform(labels["set4_w"].astype(str))
        # set4_l_trans = set4_l.transform(labels["set4_l"].astype(str))
        # set5_w_trans = set5_w.transform(labels["set5_w"].astype(str))
        # set5_l_trans = set5_l.transform(labels["set5_l"].astype(str))
        # t1_trans = t1.transform(labels["t1"].astype(str))
        # t2_trans = t2.transform(labels["t2"].astype(str))
        # t3_trans = t3.transform(labels["t3"].astype(str))
        # t4_trans = t4.transform(labels["t4"].astype(str))
        # t5_trans = t5.transform(labels["t5"].astype(str))
        #
        # features = np.hstack([features_cont, draw_size_trans, surface_trans, player1_hand_trans, player2_hand_trans, 
        #                         best_of_trans, set1_w_trans, set1_l_trans, set2_w_trans, set2_l_trans, set3_w_trans, 
        #                         set3_l_trans, set4_w_trans, set4_l_trans, set5_w_trans, set5_l_trans, t1_trans, 
        #                         t2_trans, t3_trans, t4_trans, t5_trans])
        #
        # string features: surface, player2_hand, player1_hand
        #    'surface': ['Clay', 'Hard', 'Grass', 'Carpet'],
        #    'player2_hand': ['R', 'L', 'U'],
        #    'player1_hand' : ['R', 'L', 'U']

    for index, row in features.iterrows():
        draw = str(row['draw_size'])
        if '32' == draw or '32.0' == draw:
            features.set_value(index, 'draw_size', '32')
        elif '8' == draw or '8.0' == draw:
            features.set_value(index, 'draw_size', '8')
        elif '9' == draw or '9.0' == draw:
            features.set_value(index, 'draw_size', '9')
    
    draw_size_f = pd.get_dummies(features['draw_size'], prefix='draw_size')
    features = pd.concat([features, draw_size_f], axis=1)
    features.drop(['draw_size'], axis = 1, inplace=True)
    # print(draw_size_f.head(0))
    surface_f = pd.get_dummies(features['surface'], prefix='surface')
    features = pd.concat([features, surface_f], axis=1)
    features.drop(['surface'], axis = 1, inplace=True)
    # print(surface_f.head(0))
    player1_hand_f =pd.get_dummies(features['player1_hand'], prefix='player1_hand')
    features = pd.concat([features, player1_hand_f], axis=1)
    features.drop(['player1_hand'], axis = 1, inplace=True)
    # print(player1_hand_f.head(0))
    player2_hand_f = pd.get_dummies(features['player2_hand'], prefix='player2_hand')
    features = pd.concat([features, player2_hand_f], axis=1)
    features.drop(['player2_hand'], axis = 1, inplace=True)
    # print(player2_hand_f.head(0))
    best_of_f = pd.get_dummies(features['best_of'], prefix='best_of')
    features = pd.concat([features, best_of_f], axis=1)
    features.drop(['best_of'], axis = 1, inplace=True)
    # print(best_of_f.head(0))

        #['year', 'tourney_id', 'player2_id', 'player1_id',
        # 'surface', 'draw_size', 'best_of', 'player2_seed', 'player2_hand', 'player2_ht',
        # 'player2_age', 'player2_rank', 'player1_seed', 'player1_hand', 'player1_ht', 
        # 'player1_age', 'player1_rank']

        # cont = ['year', 'tourney_id', 'player2_id', 'player1_id', 'player2_seed', 'player2_ht',
        #         'player2_age', 'player2_rank', 'player1_seed', 'player1_ht', 'player1_age', 'player1_rank']

    rs = MinMaxScaler()
    features_cont = rs.fit_transform(features)

    # print(features.head(0))
    # print(features_cont[0])

    # features = np.hstack([features_cont['year'], features_cont['tourney_id'], features_cont['player2_id'],
    #                     features_cont['player1_id'], surface_f, draw_size_f, best_of_f,
    #                     features_cont['player2_seed'], player2_hand_f, features_cont['player2_ht'], 
    #                     features_cont['player2_age'], features_cont['player2_rank'], features_cont['player1_seed'], 
    #                     player1_hand_f, features_cont['player1_ht'], features_cont['player1_age'],
    #                     features_cont['player1_rank']])
    # print(features[0:5])
    return features_cont


def one_hot_encode_features(features, labels):
    # string features: surface, player2_hand, player1_hand
    #    'surface': ['Clay', 'Hard', 'Grass', 'Carpet'],
    #    'player2_hand': ['R', 'L', 'U'],
    #    'player1_hand' : ['R', 'L', 'U']

    # features = pd.concat([features, pd.get_dummies(features['surface'], prefix='surface')], axis=1)
    # features.drop(['surface'], axis=1, inplace=True)

    # features = pd.concat([features, pd.get_dummies(features['player1_hand'], prefix='player1_hand')], axis=1)
    # features.drop(['player1_hand'], axis=1, inplace=True)

    # features = pd.concat([features, pd.get_dummies(features['player2_hand'], prefix='player2_hand')], axis=1)
    # features.drop(['player2_hand'], axis=1, inplace=True)

    # first get rid of '-D '
    set_C = "-DMO"
    for index, row in features.iterrows():
        id = row['tourney_id']
        for c in set_C:
            if c in id:
                id = id.split(c)[0] + id.split(c)[1]
        features.set_value(index, 'tourney_id', id)
    
    # string to numeric -> tourney_id
    # features['tourney_id'] = pd.to_numeric(features['tourney_id'], errors='coerce')
    # labels['tourney_id'] = pd.to_numeric(labels['tourney_id'], errors='coerce')

    # set_C2 = "-() "
    # for index, row in labels.iterrows():
    #     id = row['score']
    #     if id != 0:
    #         if 'RET' in id:
    #             id = id.replace("RET", "9")
    #         if 'W/O' in id:
    #             id = id.replace("W/O", "00")
    #         if 'Played and abandoned' in id:
    #             id = id.replace("Played and abandoned", "111")
    #         if 'Played and unfinished' in id:
    #             id = id.replace("Played and unfinished", "222")
    #         if 'Unfinished' in id:
    #             id = id.replace("Unfinished", "333")
    #         if 'In Progress' in id:
    #             id = id.replace("In Progress", "0")
    #         if 'DEF' in id:
    #             id = id.replace("DEF", "444")
    #         if 'Walkover' in id:
    #             id = id.replace("Walkover", "555")
    #         date = ['3-Jun', '3Jun']
    #         for d in date: 
    #             if d in id:
    #                 print("REPLACED THE 3-JUN!!!")
    #                 id = id.replace(d, "6300")
    #     for c in set_C2:
    #         if id != 0 and c in id:
    #                 id = id.split(c)[0] + id.split(c)[1]
    #     labels.set_value(index, 'score', id)

    # print(features.head(5))
    # print(labels.head(5))

    return features, labels


# not used
def normalize_atp():
    global ATP_features_train, ATP_features_test
    global mean, std

    mean = 0
    std = 0

    mean = ATP_features_train.mean(axis=0)
    ATP_features_train -= mean
    std = ATP_features_train.std(axis=0)
    ATP_features_train /= std
    
    ATP_features_test -= mean
    ATP_features_test /= std


# not used
def normalize_wta():
    global WTA_features_train, WTA_features_test
    global mean, std

    mean = 0
    std = 0

    mean = WTA_features_train.mean(axis=0)
    WTA_features_train -= mean
    std = WTA_features_train.std(axis=0)
    WTA_features_train /= std
    
    WTA_features_test -= mean
    WTA_features_test /= std


# not used
def denormalize(predictions):
    global mean, std

    predictions = predictions * std
    predictions = predictions + mean

    return predictions


def eval_shapes_atp():
    global ATP_features_train, ATP_features_test, ATP_labels_train, ATP_labels_test
    print(ATP_features_train.shape)
    print(ATP_features_test.shape)
    print(ATP_labels_test.shape)
    print(ATP_labels_train.shape)

    df1 = pd.DataFrame(data=ATP_features_train,
                    columns=['year', 'tourney_id', 'player2_id', 'player1_id', 'player2_seed', 'player2_ht',
                            'player2_age', 'player2_rank', 'player1_seed', 'player1_ht', 'player1_age', 'player1_rank',
                            'draw_size_0', 'draw_size_10', 'draw_size_12', 'draw_size_128', 'draw_size_16', 'draw_size_24', 
                            'draw_size_28', 'draw_size_32', 'draw_size_4', 'draw_size_48', 'draw_size_56', 
                            'draw_size_6', 'draw_size_64', 'draw_size_7', 'draw_size_8', 'draw_size_9', 
                            'draw_size_96', 'surface_0', 'surface_Carpet', 'surface_Clay', 'surface_Grass', 
                            'surface_Hard', 'surface_None', 'player1_hand_0', 'player1_hand_L', 
                            'player1_hand_R', 'player1_hand_U', 'player2_hand_0', 'player2_hand_L', 'player2_hand_R', 
                            'player2_hand_U', 'best_of_3', 'best_of_5'])
    df1.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\ATP_features_train.csv", sep='\t', encoding='utf-8')
    df2 = pd.DataFrame(data=ATP_features_test,
                    columns=['year', 'tourney_id', 'player2_id', 'player1_id', 'player2_seed', 'player2_ht',
                            'player2_age', 'player2_rank', 'player1_seed', 'player1_ht', 'player1_age', 'player1_rank',
                            'draw_size_0', 'draw_size_10', 'draw_size_12', 'draw_size_128', 'draw_size_16', 'draw_size_24', 
                            'draw_size_28', 'draw_size_32', 'draw_size_4', 'draw_size_48', 'draw_size_56', 
                            'draw_size_6', 'draw_size_64', 'draw_size_7', 'draw_size_8', 'draw_size_9', 
                            'draw_size_96', 'surface_0', 'surface_Carpet', 'surface_Clay', 'surface_Grass', 
                            'surface_Hard', 'surface_None', 'player1_hand_0', 'player1_hand_L', 
                            'player1_hand_R', 'player1_hand_U', 'player2_hand_0', 'player2_hand_L', 'player2_hand_R', 
                            'player2_hand_U', 'best_of_3', 'best_of_5'])
    df2.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\ATP_features_test.csv", sep='\t', encoding='utf-8')
    
    df3 = pd.DataFrame(data=ATP_labels_train)
    df3.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\ATP_labels_train.csv", sep='\t', encoding='utf-8')
    df4 = pd.DataFrame(data=ATP_labels_test)
    df4.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\ATP_labels_test.csv", sep='\t', encoding='utf-8')

    # print(ATP_features_test.values[:,0].shape, ATP_labels_test.values[:,0].shape)


def eval_shapes_wta():
    global WTA_features_train, WTA_features_test, WTA_labels_train, WTA_labels_test
    print(WTA_features_train.shape)
    print(WTA_features_test.shape)
    print(WTA_labels_test.shape)
    print(WTA_labels_train.shape)

    df1 = pd.DataFrame(data=WTA_features_train,
                    columns=['year', 'tourney_id', 'player2_id', 'player1_id', 'player2_seed', 'player2_ht',
                            'player2_age', 'player2_rank', 'player1_seed', 'player1_ht', 'player1_age', 'player1_rank',
                            'draw_size_12', 'draw_size_128', 'draw_size_15', 'draw_size_16', 'draw_size_28', 'draw_size_30', 
                            'draw_size_31', 'draw_size_32', 'draw_size_4', 'draw_size_48', 'draw_size_54', 'draw_size_55', 
                            'draw_size_56', 'draw_size_60', 'draw_size_64', 'draw_size_8', 'draw_size_96', 
                            'surface_0', 'surface_Carpet', 'surface_Clay', 'surface_Grass', 
                            'surface_Hard', 'player1_hand_0', 'player1_hand_L', 
                            'player1_hand_R', 'player1_hand_U', 'player2_hand_0', 'player2_hand_L', 'player2_hand_R', 
                            'player2_hand_U', 'best_of_3', 'best_of_5'])
    df1.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\WTA_features_train.csv", sep='\t', encoding='utf-8')
    df2 = pd.DataFrame(data=WTA_features_test,
                    columns=['year', 'tourney_id', 'player2_id', 'player1_id', 'player2_seed', 'player2_ht',
                            'player2_age', 'player2_rank', 'player1_seed', 'player1_ht', 'player1_age', 'player1_rank',
                            'draw_size_12', 'draw_size_128', 'draw_size_15', 'draw_size_16', 'draw_size_28', 'draw_size_30', 
                            'draw_size_31', 'draw_size_32', 'draw_size_4', 'draw_size_48', 'draw_size_54', 'draw_size_55', 
                            'draw_size_56', 'draw_size_60', 'draw_size_64', 'draw_size_8', 'draw_size_96', 
                            'surface_0', 'surface_Carpet', 'surface_Clay', 'surface_Grass', 
                            'surface_Hard', 'player1_hand_0', 'player1_hand_L', 
                            'player1_hand_R', 'player1_hand_U', 'player2_hand_0', 'player2_hand_L', 'player2_hand_R', 
                            'player2_hand_U', 'best_of_3', 'best_of_5'])
    df2.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\WTA_features_test.csv", sep='\t', encoding='utf-8')
    
    df3 = pd.DataFrame(data=WTA_labels_train)
    df3.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\WTA_labels_train.csv", sep='\t', encoding='utf-8')
    df4 = pd.DataFrame(data=WTA_labels_test)
    df4.to_csv("D:\\EXPERT\\lab_se\\TennisPrediction\\WTA_labels_test.csv", sep='\t', encoding='utf-8')


def baseline_model_atp():
    model = Sequential()
    model.add(Dense(128, input_dim=45, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='linear'))
    model.add(Dense(35, kernel_initializer='normal', activation='linear'))

    # rms = optimizers.RMSprop(lr=0.005)
    
    model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=['mse', 'mae', 'mape', 'cosine', 'accuracy'])
    return model


def baseline_model_wta():
    model = Sequential()
    model.add(Dense(128, input_dim=44, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='linear'))
    model.add(Dense(35, kernel_initializer='normal', activation='linear'))

    # sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.8, nesterov=True)
    # rms = optimizers.RMSprop(lr=0.005)
    # adam = optimizers.Adam(lr=0.01)
    # nadam = optimizers.Nadam(lr=0.01)
    # adagrad = optimizers.Adagrad()
    
    model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=['mse', 'mae', 'mape', 'cosine', 'accuracy'])
    return model


def evaluate_1():
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=baseline_model_atp, epochs=200, batch_size=256, verbose=1)

    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, ATP_features_train, ATP_labels_train, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def evaluate_2():
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=baseline_model_wta, epochs=200, batch_size=256, verbose=1)

    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, WTA_features_train, WTA_labels_train, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def plot_train(history):
    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # acc
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # mse
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('mean squared error')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # mae
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('mean absolute error')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # mape
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title('mean absolute percentage error')
    plt.ylabel('mape')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # cosine
    plt.plot(history.history['cosine_proximity'])
    plt.plot(history.history['val_cosine_proximity'])
    plt.title('cosine proximity')
    plt.ylabel('cosine')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_test_atp(model):
    global ATP_features_test, ATP_labels_test
    print('\nTesting ------------')
    cost = model.evaluate(ATP_features_test, ATP_labels_test, batch_size=40)
    print('test cost:', cost)
    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)


def plot_test_wta(model):
    global WTA_features_test, WTA_labels_test
    print('\nTesting ------------')
    cost = model.evaluate(WTA_features_test, WTA_labels_test, batch_size=40)
    print('test cost:', cost)
    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)


def plot_predict_atp(model):
    global ATP_features_test, ATP_labels_test

    Y_pred = model.predict(ATP_features_test)
    # print(Y_pred)
    print(Y_pred.shape, ATP_labels_test.shape)
    compare_predictions_atp(Y_pred)
    # plt.scatter(ATP_features_test, ATP_labels_test)
    # plt.plot(ATP_features_test, Y_pred)
    # plt.show()


def plot_predict_wta(model):
    global WTA_features_test, WTA_labels_test

    Y_pred = model.predict(WTA_features_test)
    # print(Y_pred)
    print(Y_pred.shape, WTA_labels_test.shape)
    compare_predictions_wta(Y_pred)


def compare_predictions_atp(predictions):
    global ATP_labels_test, RESULTS

    # show the inputs and predicted outputs
    print(ATP_labels_test.shape, predictions.shape)
    dave_p = pd.DataFrame(data=predictions,
                        columns=['winner',
                                'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                                't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                                'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                                'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                                'player1_bpFaced'])  # 1st row as the column names
    dave_p.to_csv(os.path.join(RESULTS, 'predictions_atp.csv'), sep='\t', encoding='utf-8')
    # for i in range(len(predictions)):
	#     print("Actual=%s, Predicted=%s" % (ATP_labels_test.values[:,0][i], predictions['score'][i]))


def compare_predictions_wta(predictions):
    global WTA_labels_test, RESULTS

    # show the inputs and predicted outputs
    print(WTA_labels_test.shape, predictions.shape)
    dave_p = pd.DataFrame(data=predictions,
                        columns=['winner',
                                'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                                't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                                'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                                'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                                'player1_bpFaced'])  # 1st row as the column names
    dave_p.to_csv(os.path.join(RESULTS, 'predictions_wta.csv'), sep='\t', encoding='utf-8')


def train_test_baseline_atp():
    global RESULTS

    model = baseline_model_atp()
    model.summary()

    checkpoint_name = os.path.join(RESULTS, 'Weights-ATP-{epoch:03d}--{val_loss:.5f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]

    global ATP_features_train, ATP_labels_train

    # print(ATP_features_train.head(5))
    # print(ATP_labels_train.head(5))

    history = model.fit(ATP_features_train, ATP_labels_train, epochs=200, batch_size=1024, validation_split = 0.2, callbacks=callbacks_list, verbose=1)
    
    plot_train(history)

    plot_test_atp(model)

    plot_predict_atp(model)

    model.save(os.path.join(RESULTS, "best_model_atp.h5"))


def train_test_baseline_wta():
    global RESULTS

    model = baseline_model_wta()
    model.summary()

    checkpoint_name = os.path.join(RESULTS, 'Weights-WTA-{epoch:03d}--{val_loss:.5f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]

    global WTA_features_train, WTA_labels_train

    history = model.fit(WTA_features_train, WTA_labels_train, epochs=200, batch_size=256, validation_split = 0.15, callbacks=callbacks_list, verbose=1)
    
    plot_train(history)

    plot_test_wta(model)

    plot_predict_wta(model)

    model.save(os.path.join(RESULTS, "best_model_wta.h5"))


def predict_best_atp():
    global ATP_features_test, ATP_labels_test, RESULTS

    model = load_model(os.path.join(RESULTS, "best_model_atp.h5"))

    Y_pred = model.predict(ATP_features_test)
    # print(Y_pred)
    print(Y_pred.shape, ATP_labels_test.shape)
    # plt.scatter(ATP_features_test, ATP_labels_test)
    # plt.plot(ATP_features_test, Y_pred)
    # plt.show()

    Y_pred_df = pd.DataFrame(data=Y_pred, 
                            columns=['winner',
                            'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                            't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                            'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                            'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                            'player1_bpFaced'])
    
    ATP_labels_test_df = pd.DataFrame(data=ATP_labels_test, 
                        columns=['winner',
                        'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                        't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                        'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                        'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                        'player1_bpFaced'])

    with open(os.path.join(RESULTS, "predictions_compared_atp.csv"), 'w') as output:
        filewriter_output = csv.writer(output, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        filewriter_output.writerow(['type', 'winner',
                                    'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 
                                    'set5_w', 'set5_l', 't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 
                                    'player2_svpt', 'player2_1stIn', 'player2_1stWon', 'player2_2ndWon', 'player2_SvGms', 
                                    'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 'player1_svpt', 
                                    'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved', 
                                    'player1_bpFaced'])
        
        for i, j in zip(Y_pred_df.index, ATP_labels_test_df.index): #df.loc[i,'Age']
            filewriter_output.writerow(['predicted', # predicted
                                        Y_pred_df.loc[i, 'winner'],
                                        Y_pred_df.loc[i, 'set1_w'], Y_pred_df.loc[i, 'set1_l'], Y_pred_df.loc[i, 'set2_w'], Y_pred_df.loc[i, 'set2_l'], 
                                        Y_pred_df.loc[i, 'set3_w'], Y_pred_df.loc[i, 'set3_l'], Y_pred_df.loc[i, 'set4_w'], Y_pred_df.loc[i, 'set4_l'], 
                                        Y_pred_df.loc[i, 'set5_w'], Y_pred_df.loc[i, 'set5_l'], Y_pred_df.loc[i, 't1'], Y_pred_df.loc[i, 't2'], Y_pred_df.loc[i, 't3'], 
                                        Y_pred_df.loc[i, 't4'], Y_pred_df.loc[i, 't5'], Y_pred_df.loc[i, 'minutes'], Y_pred_df.loc[i, 'player2_ace'], 
                                        Y_pred_df.loc[i, 'player2_df'], Y_pred_df.loc[i, 'player2_svpt'], Y_pred_df.loc[i, 'player2_1stIn'], 
                                        Y_pred_df.loc[i, 'player2_1stWon'], Y_pred_df.loc[i, 'player2_2ndWon'], Y_pred_df.loc[i, 'player2_SvGms'], 
                                        Y_pred_df.loc[i, 'player2_bpSaved'], Y_pred_df.loc[i, 'player2_bpFaced'], Y_pred_df.loc[i, 'player1_ace'], 
                                        Y_pred_df.loc[i, 'player1_df'], Y_pred_df.loc[i, 'player1_svpt'], Y_pred_df.loc[i, 'player1_1stIn'], 
                                        Y_pred_df.loc[i, 'player1_1stWon'], Y_pred_df.loc[i, 'player1_2ndWon'], Y_pred_df.loc[i, 'player1_SvGms'], 
                                        Y_pred_df.loc[i, 'player1_bpSaved'], Y_pred_df.loc[i, 'player1_bpFaced']])
            filewriter_output.writerow(['actual',  # actual
                                        ATP_labels_test_df.loc[j, 'winner'],
                                        ATP_labels_test_df.loc[j, 'set1_w'], ATP_labels_test_df.loc[j, 'set1_l'], ATP_labels_test_df.loc[j, 'set2_w'], ATP_labels_test_df.loc[j, 'set2_l'], 
                                        ATP_labels_test_df.loc[j, 'set3_w'], ATP_labels_test_df.loc[j, 'set3_l'], ATP_labels_test_df.loc[j, 'set4_w'], ATP_labels_test_df.loc[j, 'set4_l'], 
                                        ATP_labels_test_df.loc[j, 'set5_w'], ATP_labels_test_df.loc[j, 'set5_l'], ATP_labels_test_df.loc[j, 't1'], ATP_labels_test_df.loc[j, 't2'], 
                                        ATP_labels_test_df.loc[j, 't3'], ATP_labels_test_df.loc[j, 't4'], ATP_labels_test_df.loc[j, 't5'], ATP_labels_test_df.loc[j, 'minutes'], 
                                        ATP_labels_test_df.loc[j, 'player2_ace'], ATP_labels_test_df.loc[j, 'player2_df'], ATP_labels_test_df.loc[j, 'player2_svpt'], 
                                        ATP_labels_test_df.loc[j, 'player2_1stIn'], ATP_labels_test_df.loc[j, 'player2_1stWon'], ATP_labels_test_df.loc[j, 'player2_2ndWon'], 
                                        ATP_labels_test_df.loc[j, 'player2_SvGms'], ATP_labels_test_df.loc[j, 'player2_bpSaved'], ATP_labels_test_df.loc[j, 'player2_bpFaced'], 
                                        ATP_labels_test_df.loc[j, 'player1_ace'], ATP_labels_test_df.loc[j, 'player1_df'], ATP_labels_test_df.loc[j, 'player1_svpt'], 
                                        ATP_labels_test_df.loc[j, 'player1_1stIn'], ATP_labels_test_df.loc[j, 'player1_1stWon'], ATP_labels_test_df.loc[j, 'player1_2ndWon'], 
                                        ATP_labels_test_df.loc[j, 'player1_SvGms'], ATP_labels_test_df.loc[j, 'player1_bpSaved'], ATP_labels_test_df.loc[j, 'player1_bpFaced']])


def predict_best_wta():
    global WTA_features_test, WTA_labels_test, RESULTS

    model = load_model(os.path.join(RESULTS, "best_model_wta.h5"))

    Y_pred = model.predict(WTA_features_test)
    # print(Y_pred)
    print(Y_pred.shape, WTA_labels_test.shape)

    Y_pred_df = pd.DataFrame(data=Y_pred, 
                            columns=['winner',
                            'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                            't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                            'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                            'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                            'player1_bpFaced'])
    
    WTA_labels_test_df = pd.DataFrame(data=WTA_labels_test, 
                        columns=['winner',
                        'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                        't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                        'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                        'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                        'player1_bpFaced'])

    with open(os.path.join(RESULTS, "predictions_compared_wta.csv"), 'w') as output:
        filewriter_output = csv.writer(output, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        filewriter_output.writerow(['type', 'winner',
                                    'set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 
                                    'set5_w', 'set5_l', 't1', 't2', 't3', 't4', 't5', 'minutes', 'player2_ace', 'player2_df', 
                                    'player2_svpt', 'player2_1stIn', 'player2_1stWon', 'player2_2ndWon', 'player2_SvGms', 
                                    'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 'player1_svpt', 
                                    'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved', 
                                    'player1_bpFaced'])

        for i, j in zip(Y_pred_df.index, WTA_labels_test_df.index): #df.loc[i,'Age']
            filewriter_output.writerow(['predicted', # predicted
                                        Y_pred_df.loc[i, 'winner'],
                                        Y_pred_df.loc[i, 'set1_w'], Y_pred_df.loc[i, 'set1_l'], Y_pred_df.loc[i, 'set2_w'], Y_pred_df.loc[i, 'set2_l'], 
                                        Y_pred_df.loc[i, 'set3_w'], Y_pred_df.loc[i, 'set3_l'], Y_pred_df.loc[i, 'set4_w'], Y_pred_df.loc[i, 'set4_l'], 
                                        Y_pred_df.loc[i, 'set5_w'], Y_pred_df.loc[i, 'set5_l'], Y_pred_df.loc[i, 't1'], Y_pred_df.loc[i, 't2'], Y_pred_df.loc[i, 't3'], 
                                        Y_pred_df.loc[i, 't4'], Y_pred_df.loc[i, 't5'], Y_pred_df.loc[i, 'minutes'], Y_pred_df.loc[i, 'player2_ace'], 
                                        Y_pred_df.loc[i, 'player2_df'], Y_pred_df.loc[i, 'player2_svpt'], Y_pred_df.loc[i, 'player2_1stIn'], 
                                        Y_pred_df.loc[i, 'player2_1stWon'], Y_pred_df.loc[i, 'player2_2ndWon'], Y_pred_df.loc[i, 'player2_SvGms'], 
                                        Y_pred_df.loc[i, 'player2_bpSaved'], Y_pred_df.loc[i, 'player2_bpFaced'], Y_pred_df.loc[i, 'player1_ace'], 
                                        Y_pred_df.loc[i, 'player1_df'], Y_pred_df.loc[i, 'player1_svpt'], Y_pred_df.loc[i, 'player1_1stIn'], 
                                        Y_pred_df.loc[i, 'player1_1stWon'], Y_pred_df.loc[i, 'player1_2ndWon'], Y_pred_df.loc[i, 'player1_SvGms'], 
                                        Y_pred_df.loc[i, 'player1_bpSaved'], Y_pred_df.loc[i, 'player1_bpFaced']])
            filewriter_output.writerow(['actual',  # actual
                                        WTA_labels_test_df.loc[j, 'winner'],
                                        WTA_labels_test_df.loc[j, 'set1_w'], WTA_labels_test_df.loc[j, 'set1_l'], WTA_labels_test_df.loc[j, 'set2_w'], WTA_labels_test_df.loc[j, 'set2_l'], 
                                        WTA_labels_test_df.loc[j, 'set3_w'], WTA_labels_test_df.loc[j, 'set3_l'], WTA_labels_test_df.loc[j, 'set4_w'], WTA_labels_test_df.loc[j, 'set4_l'], 
                                        WTA_labels_test_df.loc[j, 'set5_w'], WTA_labels_test_df.loc[j, 'set5_l'], WTA_labels_test_df.loc[j, 't1'], WTA_labels_test_df.loc[j, 't2'], 
                                        WTA_labels_test_df.loc[j, 't3'], WTA_labels_test_df.loc[j, 't4'], WTA_labels_test_df.loc[j, 't5'], WTA_labels_test_df.loc[j, 'minutes'], 
                                        WTA_labels_test_df.loc[j, 'player2_ace'], WTA_labels_test_df.loc[j, 'player2_df'], WTA_labels_test_df.loc[j, 'player2_svpt'], 
                                        WTA_labels_test_df.loc[j, 'player2_1stIn'], WTA_labels_test_df.loc[j, 'player2_1stWon'], WTA_labels_test_df.loc[j, 'player2_2ndWon'], 
                                        WTA_labels_test_df.loc[j, 'player2_SvGms'], WTA_labels_test_df.loc[j, 'player2_bpSaved'], WTA_labels_test_df.loc[j, 'player2_bpFaced'], 
                                        WTA_labels_test_df.loc[j, 'player1_ace'], WTA_labels_test_df.loc[j, 'player1_df'], WTA_labels_test_df.loc[j, 'player1_svpt'], 
                                        WTA_labels_test_df.loc[j, 'player1_1stIn'], WTA_labels_test_df.loc[j, 'player1_1stWon'], WTA_labels_test_df.loc[j, 'player1_2ndWon'], 
                                        WTA_labels_test_df.loc[j, 'player1_SvGms'], WTA_labels_test_df.loc[j, 'player1_bpSaved'], WTA_labels_test_df.loc[j, 'player1_bpFaced']])


def main():
    # separate_atp()
    # train_test_baseline_atp()
    # predict_best_atp()
    separate_wta()
    train_test_baseline_wta()
    predict_best_wta()
    # evaluate_1()
    # evaluate_2()


if __name__ == "__main__":
    main()