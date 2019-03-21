import os
import pandas as pd
import numpy as np
import csv

WTA_path = "D:\\EXPERT\\lab_se\\tennis_wta"
ATP_path = "D:\\EXPERT\\lab_se\\tennis_atp\\matches"

def main():
    with open('D:\\EXPERT\\lab_se\\TennisPrediction\\wta_players_basicinfo.csv', 'w', encoding='utf-8') as map_wta:
        filewriter_map_wta = csv.writer(map_wta, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_map_wta.writerow(['name', 'age', 'height'])

        players_list = []
        for file in os.listdir(WTA_path):
            if "wta_matches" in file and "futures" not in file:
                read_file = pd.read_csv(os.path.join(WTA_path, file), parse_dates=False, encoding = "ISO-8859-1")
                print("Processing ... " + os.path.join(WTA_path, file))
                
                for index, row in read_file.iterrows():
                    p1_name = str(row['winner_name'])
                    p2_name = str(row['loser_name'])
                    
                    if p1_name not in players_list:
                        height = str(row['winner_ht'])
                        age = str(row['winner_age'])
                        if ((height is not None or height is not "") and (age is not None or age is not "")):
                            players_list.append(p1_name)
                            filewriter_map_wta.writerow([p1_name, age, height])
                    
                    if p2_name not in players_list:
                        height = str(row['loser_ht'])
                        age = str(row['loser_age'])
                        if ((height is not None or height is not "") and (age is not None or age is not "")):
                            players_list.append(p2_name)
                            filewriter_map_wta.writerow([p2_name, age, height])
    
    with open('D:\\EXPERT\\lab_se\\TennisPrediction\\atp_players_basicinfo.csv', 'w', encoding='utf-8') as map_atp:
        filewriter_map_atp = csv.writer(map_atp, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_map_atp.writerow(['name', 'age', 'height'])

        players_list = []
        for file in os.listdir(ATP_path):
            if "atp_matches" in file and "futures" not in file:
                read_file = pd.read_csv(os.path.join(ATP_path, file), parse_dates=False, encoding = "ISO-8859-1")
                print("Processing ... " + os.path.join(ATP_path, file))
                
                for index, row in read_file.iterrows():
                    p1_name = str(row['winner_name'])
                    p2_name = str(row['loser_name'])
                    
                    if p1_name not in players_list:
                        height = str(row['winner_ht'])
                        age = str(row['winner_age'])
                        if (height is not None or height is not "") and (age is not None or age is not ""):
                            players_list.append(p1_name)
                            filewriter_map_atp.writerow([p1_name, age, height])
                    
                    if p2_name not in players_list:
                        height = str(row['loser_ht'])
                        age = str(row['loser_age'])
                        if (height is not None or height is not "") and (age is not None or age is not ""):
                            players_list.append(p2_name)
                            filewriter_map_atp.writerow([p2_name, age, height])


if __name__ == "__main__":
    main()