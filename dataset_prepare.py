import os
import pandas as pd
import csv
import numpy as np

# ATP_path = "D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\atp-matches-dataset"
ATP_path = "D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\tennis_atp\\matches"
WTA_path = "D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\tennis_wta"

RENAME = 1

def read_datasets_atp():

    # original:
    # winner_seed   winner_name	    winner_rank	    score
    # winner_hand	winner_ht	    winner_ioc	    winner_rank_points	
    # loser_seed	loser_name	    loser_hand	    loser_rank	    
    # loser_ht	    loser_ioc	    loser_rank_points	
    # best_of	    minutes	        w_ace	        w_df	        w_svpt	        l_SvGms	
    # w_1stIn	    w_1stWon	    w_2ndWon	    w_SvGms	        w_bpSaved	    w_bpFaced	
    # l_ace	        l_df	        l_svpt	        l_1stIn	        l_1stWon	    l_2ndWon	        
    # l_bpSaved	    l_bpFaced
    # tourney_id, tourney_date, match_num, winner_id, winner_entry, loser_id, loser_entry, round
    # tourney_name, draw_size, surface, tourney_level, player_1, player_2, winner_age, loser_age

    #!Remember: winner => player2 ; loser => player1

    # final: 
    # tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, match_num,
    # player2_id, player2_seed, player2_entry, player2_name, player2_hand, player2_ht, player2_ioc, 
    # player2_age, player2_rank, player2_rank_points, player1_id, player1_seed, player1_entry, 
    # player1_name, player1_hand, player1_ht, player1_ioc, player1_age, player1_rank, player1_rank_points, 
    # score, best_of, round, minutes, player2_ace, player2_df, player2_svpt, player2_1stIn, player2_1stWon, 
    # player2_2ndWon, player2_SvGms, player2_bpSaved, player2_bpFaced, player1_ace, player1_df, player1_svpt, 
    # player1_1stIn, player1_1stWon, player1_2ndWon, player1_SvGms, player1_bpSaved, player1_bpFaced

    # input = tourney_name, tourney_id, player2_id, player1_id, surface, draw_size, player2_seed, player2_name, player2_hand, player2_ht, player2_ioc,
    #           player2_age, player2_rank, player1_seed, player1_name, player1_hand, player1_ht, player1_ioc, player1_age, player1_rank, best_of
    
    # dont care = tourney_level, tourney_date, match_num, player2_entry, player2_rank_points,
    #               player1_entry, player1_rank_points, round
    #           tourney_name, player1_name, player2_name, player2_ioc, player1_ioc
    # output = score, minutes, player2_ace, player2_df, player2_svpt, player2_1stIn, player2_1stWon, 
    #           player2_2ndWon, player2_SvGms, player2_bpSaved, player2_bpFaced, player1_ace, player1_df, player1_svpt, 
    #           player1_1stIn, player1_1stWon, player1_2ndWon, player1_SvGms, player1_bpSaved, player1_bpFaced

    # score -> set1_w, set1_l, set2_w, set2_l, set3_w, set3_l, set4_w, set4_l, set5_w, set5_l, t1, t2, t3, t4, t5

    with open('D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_atp_modified_features.csv', 'w') as features,\
            open('D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_atp_modified_labels.csv', 'w') as labels:
        filewriter_features = csv.writer(features, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_labels = csv.writer(labels, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter_features.writerow(['year', 'tourney_id', 'player2_id', 'player1_id',
                                    'surface', 'draw_size', 'best_of', 'player2_seed', 'player2_hand', 'player2_ht',
                                    'player2_age', 'player2_rank', 'player1_seed', 'player1_hand', 'player1_ht', 
                                    'player1_age', 'player1_rank'])
        filewriter_labels.writerow(['set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                                    't1', 't2', 't3', 't4', 't5',
                                    'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                                    'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                                    'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                                    'player1_bpFaced'])

        for file in os.listdir(ATP_path):
            # if "atp_matches" in file and "2015" not in file and "2016" not in file and "2017" not in file:
            if "atp_matches" in file and "futures" not in file:
                read_file = pd.read_csv(os.path.join(ATP_path, file), parse_dates=False, encoding = "ISO-8859-1")
                print("Processing ... " + os.path.join(ATP_path, file))
                if RENAME == 1:
                    read_file.rename(index=str, columns={
                        "winner_seed" : "player2_seed",
                        "winner_name" : "player2_name",
                        "winner_rank" : "player2_rank",
                        "winner_hand" : "player2_hand",
                        "winner_ht" : "player2_ht",
                        "winner_ioc" : "player2_ioc",
                        "winner_rank_points" : "player2_rank_points",
                        "winner_id" : "player2_id",
                        "winner_entry" : "player2_entry",
                        "winner_age" : "player2_age",
                        "w_ace" : "player2_ace",
                        "w_df" : "player2_df",
                        "w_svpt" : "player2_svpt",
                        "w_bpFaced" : "player2_bpFaced",
                        "w_bpSaved" : "player2_bpSaved",
                        "w_SvGms" : "player2_SvGms",
                        "w_2ndWon" : "player2_2ndWon",
                        "w_1stWon" : "player2_1stWon",
                        "w_1stIn" : "player2_1stIn",
                        "loser_seed" : "player1_seed",
                        "loser_name" : "player1_name",
                        "loser_rank" : "player1_rank",
                        "loser_hand" : "player1_hand",
                        "loser_ht" : "player1_ht",
                        "loser_ioc" : "player1_ioc",
                        "loser_rank_points" : "player1_rank_points",
                        "loser_id" : "player1_id",
                        "loser_entry" : "player1_entry",
                        "loser_age" : "player1_age",
                        "l_ace" : "player1_ace",
                        "l_df" : "player1_df",
                        "l_svpt" : "player1_svpt",
                        "l_bpFaced" : "player1_bpFaced",
                        "l_bpSaved" : "player1_bpSaved",
                        "l_SvGms" : "player1_SvGms",
                        "l_2ndWon" : "player1_2ndWon",
                        "l_1stWon" : "player1_1stWon",
                        "l_1stIn" : "player1_1stIn"
                    }, inplace=True)
                    # print("New Header: ")
                    # print(read_file.head(1))

                for index, row in read_file.iterrows():
                    # if 'futures' in file: year = file.split('_')[3].split('.')[0]
                    if 'qual_chall' in file: year = file.split('_')[4].split('.')[0]
                    else: year = file.split('_')[2].split('.')[0]
                    filewriter_features.writerow([year,
                                                row['tourney_id'], row['player2_id'], row['player1_id'],
                                                row['surface'], row['draw_size'], row['best_of'],
                                                row['player2_seed'], row['player2_hand'], 
                                                row['player2_ht'], row['player2_age'], 
                                                row['player2_rank'], row['player1_seed'],
                                                row['player1_hand'], row['player1_ht'],
                                                row['player1_age'], row['player1_rank']])
                for index, row in read_file.iterrows():
                    set1_w, set1_l, set2_w, set2_l, set3_w, set3_l, set4_w, set4_l, set5_w, set5_l,\
                        t1, t2, t3, t4, t5 = get_set_values(str(row['score']), file)
                    
                    filewriter_labels.writerow([set1_w, set1_l, set2_w, set2_l, set3_w, set3_l, set4_w, set4_l, set5_w, set5_l,
                                                t1, t2, t3, t4, t5,
                                                row['minutes'], row['player2_ace'], row['player2_df'], 
                                                row['player2_svpt'], row['player2_1stIn'], row['player2_1stWon'], 
                                                row['player2_2ndWon'], row['player2_SvGms'], row['player2_bpSaved'], 
                                                row['player2_bpFaced'], row['player1_ace'], row['player1_df'], 
                                                row['player1_svpt'], row['player1_1stIn'], row['player1_1stWon'], 
                                                row['player1_2ndWon'], row['player1_SvGms'], row['player1_bpSaved'],
                                                row['player1_bpFaced']])


def read_datasets_wta():

    # original
        # players: player_id	first_name	last_name	hand	birth_date	country_code
        # matches: best_of	draw_size	loser_age	loser_entry	loser_hand	loser_ht	loser_id	
        #           loser_ioc	loser_name	loser_rank	loser_rank_points	loser_seed	match_num	
        #           minutes	round	score	surface	tourney_date	tourney_id	tourney_level	tourney_name	
        #           winner_age	winner_entry	winner_hand	winner_ht	winner_id	winner_ioc	winner_name	
        #           winner_rank	winner_rank_points	winner_seed	year
        # rankings: ranking_date	ranking	player_id	ranking_points
    
    # modifiy: winner => player2; loser => player1

    # input = tourney_name, tourney_id, player2_id, player1_id, surface, draw_size, best_of, player2_seed, player2_name, player2_hand, player2_ht, player2_ioc,
        #           player2_age, player2_rank, player1_seed, player1_name, player1_hand, player1_ht, player1_ioc, player1_age, player1_rank,
        # dont care = tourney_level, tourney_date, match_num, player2_entry, player2_rank_points,
        #               player1_entry, player1_rank_points, round
        # output = score, minutes, player2_ace, player2_df, player2_svpt, player2_1stIn, player2_1stWon, 
        #           player2_2ndWon, player2_SvGms, player2_bpSaved, player2_bpFaced, player1_ace, player1_df, player1_svpt, 
        #           player1_1stIn, player1_1stWon, player1_2ndWon, player1_SvGms, player1_bpSaved, player1_bpFaced

    with open('D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_wta_modified_features.csv', 'w') as features,\
            open('D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_wta_modified_labels.csv', 'w') as labels:
        filewriter_features = csv.writer(features, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_labels = csv.writer(labels, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter_features.writerow(['year', 'tourney_id', 'player2_id', 'player1_id',
                                    'surface', 'draw_size', 'best_of', 'player2_seed', 'player2_hand', 'player2_ht',
                                    'player2_age', 'player2_rank', 'player1_seed', 'player1_hand', 'player1_ht', 
                                    'player1_age', 'player1_rank'])
        filewriter_labels.writerow(['set1_w', 'set1_l', 'set2_w', 'set2_l', 'set3_w', 'set3_l', 'set4_w', 'set4_l', 'set5_w', 'set5_l',
                                    't1', 't2', 't3', 't4', 't5',
                                    'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                                    'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                                    'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                                    'player1_bpFaced'])

        for file in os.listdir(WTA_path):
            if "wta_matches" in file and "futures" not in file:
                read_file = pd.read_csv(os.path.join(WTA_path, file), parse_dates=False, encoding = "ISO-8859-1")
                print("Processing ... " + os.path.join(WTA_path, file))
                if RENAME == 1:
                    read_file.rename(index=str, columns={
                        "winner_seed" : "player2_seed",
                        "winner_name" : "player2_name",
                        "winner_rank" : "player2_rank",
                        "winner_hand" : "player2_hand",
                        "winner_ht" : "player2_ht",
                        "winner_ioc" : "player2_ioc",
                        "winner_rank_points" : "player2_rank_points",
                        "winner_id" : "player2_id",
                        "winner_entry" : "player2_entry",
                        "winner_age" : "player2_age",
                        "w_ace" : "player2_ace",
                        "w_df" : "player2_df",
                        "w_svpt" : "player2_svpt",
                        "w_bpFaced" : "player2_bpFaced",
                        "w_bpSaved" : "player2_bpSaved",
                        "w_SvGms" : "player2_SvGms",
                        "w_2ndWon" : "player2_2ndWon",
                        "w_1stWon" : "player2_1stWon",
                        "w_1stIn" : "player2_1stIn",
                        "loser_seed" : "player1_seed",
                        "loser_name" : "player1_name",
                        "loser_rank" : "player1_rank",
                        "loser_hand" : "player1_hand",
                        "loser_ht" : "player1_ht",
                        "loser_ioc" : "player1_ioc",
                        "loser_rank_points" : "player1_rank_points",
                        "loser_id" : "player1_id",
                        "loser_entry" : "player1_entry",
                        "loser_age" : "player1_age",
                        "l_ace" : "player1_ace",
                        "l_df" : "player1_df",
                        "l_svpt" : "player1_svpt",
                        "l_bpFaced" : "player1_bpFaced",
                        "l_bpSaved" : "player1_bpSaved",
                        "l_SvGms" : "player1_SvGms",
                        "l_2ndWon" : "player1_2ndWon",
                        "l_1stWon" : "player1_1stWon",
                        "l_1stIn" : "player1_1stIn"
                    }, inplace=True)
                    # print("New Header: ")
                    # print(read_file.head(1))

                for index, row in read_file.iterrows():
                    # if 'futures' in file: year = file.split('_')[3].split('.')[0]
                    if 'qual_itf' in file: 
                        year = file.split('_')[4].split('.')[0]
                    else: 
                        year = file.split('_')[2].split('.')[0]
                    filewriter_features.writerow([year,
                                                row['tourney_id'], row['player2_id'], row['player1_id'],
                                                row['surface'], row['draw_size'], row['best_of'],
                                                row['player2_seed'], row['player2_hand'], 
                                                row['player2_ht'], row['player2_age'], 
                                                row['player2_rank'], row['player1_seed'],
                                                row['player1_hand'], row['player1_ht'],
                                                row['player1_age'], row['player1_rank']])
                for index, row in read_file.iterrows():
                    set1_w, set1_l, set2_w, set2_l, set3_w, set3_l, set4_w, set4_l, set5_w, set5_l,\
                        t1, t2, t3, t4, t5 = get_set_values(str(row['score']), file)
                    
                    filewriter_labels.writerow([set1_w, set1_l, set2_w, set2_l, set3_w, set3_l, set4_w, set4_l, set5_w, set5_l,
                                                t1, t2, t3, t4, t5,
                                                row['minutes'], row['player2_ace'], row['player2_df'], 
                                                row['player2_svpt'], row['player2_1stIn'], row['player2_1stWon'], 
                                                row['player2_2ndWon'], row['player2_SvGms'], row['player2_bpSaved'], 
                                                row['player2_bpFaced'], row['player1_ace'], row['player1_df'], 
                                                row['player1_svpt'], row['player1_1stIn'], row['player1_1stWon'], 
                                                row['player1_2ndWon'], row['player1_SvGms'], row['player1_bpSaved'],
                                                row['player1_bpFaced']])


def atp_prepare():
    with open('D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_atp_modified_features.csv', 'w') as features,\
            open('D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\combined_atp_modified_labels.csv', 'w') as labels:
        
        filewriter_features = csv.writer(features, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_labels = csv.writer(labels, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter_features.writerow(['year', 'tourney_id', 'player2_id', 'player1_id',
                                    'surface', 'draw_size', 'player2_seed', 'player2_hand', 'player2_ht',
                                    'player2_age', 'player2_rank', 'player1_seed', 'player1_hand', 'player1_ht', 
                                    'player1_age', 'player1_rank'])
        filewriter_labels.writerow(['score', 'minutes', 'player2_ace', 'player2_df', 'player2_svpt', 'player2_1stIn', 'player2_1stWon',
                                    'player2_2ndWon', 'player2_SvGms', 'player2_bpSaved', 'player2_bpFaced', 'player1_ace', 'player1_df', 
                                    'player1_svpt', 'player1_1stIn', 'player1_1stWon', 'player1_2ndWon', 'player1_SvGms', 'player1_bpSaved',
                                    'player1_bpFaced'])
        # os.chdir(ATP_path)
        
        filenames = os.listdir(ATP_path)
        print(filenames)
        finaldf = pd.concat( [ pd.read_csv(os.path.join(ATP_path, f), dtype={'tourney_id': np.str}) for f in filenames ] )
        print(finaldf.head(0))

        if RENAME == 1:
            finaldf.rename(index=str, columns={
                "winner_seed" : "player2_seed",
                "winner_name" : "player2_name",
                "winner_rank" : "player2_rank",
                "winner_hand" : "player2_hand",
                "winner_ht" : "player2_ht",
                "winner_ioc" : "player2_ioc",
                "winner_rank_points" : "player2_rank_points",
                "winner_id" : "player2_id",
                "winner_entry" : "player2_entry",
                "winner_age" : "player2_age",
                "w_ace" : "player2_ace",
                "w_df" : "player2_df",
                "w_svpt" : "player2_svpt",
                "w_bpFaced" : "player2_bpFaced",
                "w_bpSaved" : "player2_bpSaved",
                "w_SvGms" : "player2_SvGms",
                "w_2ndWon" : "player2_2ndWon",
                "w_1stWon" : "player2_1stWon",
                "w_1stIn" : "player2_1stIn",
                "loser_seed" : "player1_seed",
                "loser_name" : "player1_name",
                "loser_rank" : "player1_rank",
                "loser_hand" : "player1_hand",
                "loser_ht" : "player1_ht",
                "loser_ioc" : "player1_ioc",
                "loser_rank_points" : "player1_rank_points",
                "loser_id" : "player1_id",
                "loser_entry" : "player1_entry",
                "loser_age" : "player1_age",
                "l_ace" : "player1_ace",
                "l_df" : "player1_df",
                "l_svpt" : "player1_svpt",
                "l_bpFaced" : "player1_bpFaced",
                "l_bpSaved" : "player1_bpSaved",
                "l_SvGms" : "player1_SvGms",
                "l_2ndWon" : "player1_2ndWon",
                "l_1stWon" : "player1_1stWon",
                "l_1stIn" : "player1_1stIn"
            }, inplace=True)
            # print("New Header: ")
            # print(read_file.head(1))

        for index, row in finaldf.iterrows():
            filewriter_features.writerow([str(row['tourney_id']).split('-')[0],
                                        row['tourney_id'], row['player2_id'], row['player1_id'],
                                        row['surface'], row['draw_size'], 
                                        row['player2_seed'], row['player2_hand'], 
                                        row['player2_ht'], row['player2_age'], 
                                        row['player2_rank'], row['player1_seed'],
                                        row['player1_hand'], row['player1_ht'],
                                        row['player1_age'], row['player1_rank']])
        for index, row in finaldf.iterrows():
            filewriter_labels.writerow([row['score'], row['minutes'], row['player2_ace'], row['player2_df'], 
                                        row['player2_svpt'], row['player2_1stIn'], row['player2_1stWon'], 
                                        row['player2_2ndWon'], row['player2_SvGms'], row['player2_bpSaved'], 
                                        row['player2_bpFaced'], row['player1_ace'], row['player1_df'], 
                                        row['player1_svpt'], row['player1_1stIn'], row['player1_1stWon'], 
                                        row['player1_2ndWon'], row['player1_SvGms'], row['player1_bpSaved'],
                                        row['player1_bpFaced']])


def get_set_values(score, file):
    set1_w, set1_l, set2_w, set2_l, set3_w, set3_l, set4_w, set4_l, set5_w, set5_l = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    t1, t2, t3, t4, t5 = 0, 0, 0, 0, 0

    nb_sets = score.count('-')

    date = ['3-Jun', '3Jun']
    for d in date: 
        if d in score:
            print("REPLACED THE 3-JUN!!!")
            score = score.replace(d, "6-3 0-0")
    if 'Jun' in score:
        print("REPLACED THE JUN!!!")
        score = score.replace('Jun', "6")
    if 'Feb' in score:
        print("REPLACED THE Feb!!!")
        score = score.replace('Feb', "2")
    if 'Jul' in score:
        print("REPLACED THE Jul!!!")
        score = score.replace('Jul', "7")
    if 'RET' in score:
        # count and set sets and put -1 on last tiebreak -> t5
        t5 = -1
    elif 'W/O' in score:
        # put -1 on first tiebreak -> t1 = -1
        t1 = -1
    elif 'Played and abandoned' in score:
        # count and set sets and put -2 on last tiebreak -> t5
        t5 = -2
    elif 'Played and unfinished' in score:
        # count and set sets and put -3 on last tiebreak -> t5
        t5 = -3
    elif 'Unfinished' in score:
        # count and set sets and put -4 on last tiebreak -> t5
        t5 = -4
    elif 'In Progress' in score:
        # count and set sets and put -5 on last tiebreak -> t5
        t5 = -5
    elif 'DEF' in score:
        # count and set sets and put -6 on last tiebreak -> t5
        t5 = -6
    elif 'Walkover' in score:
        # count and set sets and put -7 on last tiebreak -> t5
        t5 = -7

    nb_sets = score.count('-')
    # set 1
    if nb_sets > 0: # we have at least 1 set
        s1 = score.split(' ')[0] # ex. 6-3
        if '(' in s1: # ex. 7-6(4), 6-7(3)
            t1 = s1.split('(')[1].split(')')[0]
        # print(score, file)
        if s1.split('-')[0] > s1.split('-')[1]:
            set1_w = s1.split('-')[0].split('(')[0]
            set1_l = s1.split('-')[1].split('(')[0]
        else:
            set1_l = s1.split('-')[0].split('(')[0]
            set1_w = s1.split('-')[1].split('(')[0]
    # set 2
    if nb_sets > 1: # we have a second set
        s2 = score.split(' ')[1].split(')')[0]
        if '(' in s2: # ex. 7-6(4), 6-7(3)
            t2 = s2.split('(')[1]
        if s2.split('-')[0] > s2.split('-')[1]:
            set2_w = s2.split('-')[0].split('(')[0]
            set2_l = s2.split('-')[1].split('(')[0]
        else:
            set2_l = s2.split('-')[0].split('(')[0]
            set2_w = s2.split('-')[1].split('(')[0]
    # set 3
    if nb_sets > 2: # winner might have won in 2 sets
        s3 = score.split(' ')[2]
        if '(' in s3: # ex. 7-6(4), 6-7(3)
            t3 = s3.split('(')[1].split(')')[0]
        if s3.split('-')[0] > s3.split('-')[1]:
            set3_w = s3.split('-')[0].split('(')[0]
            set3_l = s3.split('-')[1].split('(')[0]
        else:
            set3_l = s3.split('-')[0].split('(')[0]
            set3_w = s3.split('-')[1].split('(')[0]
    if nb_sets > 3: # we have a forth set (best of 5)
        s4 = score.split(' ')[3]
        if '(' in s4: # ex. 7-6(4), 6-7(3)
            t4 = s4.split('(')[1].split(')')[0]
        if s4.split('-')[0] > s4.split('-')[1]:
            set4_w = s4.split('-')[0].split('(')[0]
            set4_l = s4.split('-')[1].split('(')[0]
        else:
            set4_l = s4.split('-')[0].split('(')[0]
            set4_w = s4.split('-')[1].split('(')[0]
    if nb_sets > 4: # we have a fifth! set
        s5 = score.split(' ')[4]
        if '(' in s5: # ex. 7-6(4), 6-7(3)
            t5 = s5.split('(')[1].split(')')[0]
        if s5.split('-')[0] > s5.split('-')[1]:
            set5_w = s5.split('-')[0].split('(')[0]
            set5_l = s5.split('-')[1].split('(')[0]
        else:
            set5_l = s5.split('-')[0].split('(')[0]
            set5_w = s5.split('-')[1].split('(')[0]
    
    return set1_w, set1_l, set2_w, set2_l, set3_w, set3_l, set4_w, set4_l, set5_w, set5_l, t1, t2, t3, t4, t5


def main():
    # read_datasets_atp()
    read_datasets_wta()
    # atp_prepare()


if __name__ == "__main__":
    main()