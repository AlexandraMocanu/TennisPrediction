import os
import pandas as pd
import numpy as np

ATP_path = "D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\atp-matches-dataset"

def main():
    
    set_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for file in os.listdir(ATP_path):
        read_file = pd.read_csv(os.path.join(ATP_path, file), parse_dates=False, dtype={'score': np.str})
        print("Processing ... " + os.path.join(ATP_path, file))
        for index, row in read_file.iterrows():
            # print(row['score'])
            # for month in set_months:
            #     if row['score'] == 0:
            #         print("Found 0 score: ", row['score'])
            #     elif month in row['score']:
            #         print("Found wrong score: ", row['score'])
            if isinstance(row['score'], float): print("Float? ", row['score'])
            elif row['score'].count('-') == 1 and "RET" not in row['score'] and "DEF" not in row['score']:
                print("One set: ", row['score'])


if __name__ == "__main__":
    main()