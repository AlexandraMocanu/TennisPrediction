import os
import pandas as pd
import numpy as np
import csv

WTA_players = "D:\\EXPERT\\lab_se\\tennis_wta\\wta_players.csv"
ATP_players = "D:\\EXPERT\\lab_se\\tennis_atp\\atp_players.csv"

def main():
    with open('D:\\EXPERT\\lab_se\\TennisPrediction\\wta_players_map.csv', 'w', encoding='utf-8') as map_wta:
        filewriter_map_wta = csv.writer(map_wta, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_map_wta.writerow(['id', 'full_name'])
        players = pd.read_csv(WTA_players, encoding = "ISO-8859-1")
        for index, row in players.iterrows():
            id_player = str(row['id'])
            name_player = str(row['first']) + " " + str(row['last'])
            filewriter_map_wta.writerow([id_player, name_player])
    
    with open('D:\\EXPERT\\lab_se\\TennisPrediction\\atp_players_map.csv', 'w', encoding='utf-8') as map_atp:
        filewriter_map_atp = csv.writer(map_atp, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_map_atp.writerow(['id', 'full_name'])
        players = pd.read_csv(ATP_players, encoding = "ISO-8859-1")
        for index, row in players.iterrows():
            id_player = str(row['id'])
            name_player = str(row['first']) + " " + str(row['last'])
            filewriter_map_atp.writerow([id_player, name_player])


if __name__ == "__main__":
    main()