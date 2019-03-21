import csv
import pandas as pd
import os

ATP_path = "D:\\EXPERT\\lab_se\\tennis_atp\\matches"
WTA_path = "D:\\EXPERT\\lab_se\\tennis_wta"

def create_unique_tourneys():
    # ATP
    # all_atp = pd.concat([pd.read_csv(os.path.join(ATP_path, f), engine='python') for f in os.listdir(ATP_path) if 'futures' not in f], sort=True)
    # t = all_atp['tourney_name'].tolist()
    # t_df = pd.DataFrame(data=t, columns=['tourney_name'])
    # t_df = set_new_tourney_names_atp(t_df)
    # u_df = t_df.tourney_name.unique()
    # u_df = pd.DataFrame(data=u_df, columns=['tourney_name'])
    # u_df.to_csv(os.path.join(ATP_path,'unique_tourneys_atp.csv'), columns=['tourney_name'], index=True, index_label='id')
    # WTA
    all_wta = pd.concat([pd.read_csv(os.path.join(WTA_path, f), engine='python') for f in os.listdir(WTA_path) if 'matches' in f], sort=True)
    t = all_wta['tourney_name'].tolist()
    t_df = pd.DataFrame(data=t, columns=['tourney_name'])
    t_df = set_new_tourney_names_wta(t_df)
    u_df = t_df.tourney_name.unique()
    u_df = pd.DataFrame(data=u_df, columns=['tourney_name'])
    u_df.to_csv(os.path.join(WTA_path,'unique_tourneys_wta.csv'), columns=['tourney_name'], index=True, index_label='id')


def set_new_tourney_names_atp(df_t):
    with open('D:\\EXPERT\\lab_se\\TennisPrediction\\atp_tourney_map.csv', 'w') as map_atp:
        filewriter_map_atp = csv.writer(map_atp, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_map_atp.writerow(['old_name', 'new_name'])

        for index, row in df_t.iterrows():
            t_name = str(row['tourney_name'])
            if "Davis" in t_name:
                t_name = "Davis Cup"
            if " CH" in t_name:
                t = t_name.split(' CH')[0]
                t_name = t
            if " Q" in t_name:
                t = t_name.split(' Q')[0]
                t_name = t
            if " WCT" in t_name:
                t = t_name.split(' WCT')[0]
                t_name = t
            if " WTC" in t_name:
                t = t_name.split(' WTC')[0]
                t_name = t
            if "2" in t_name:
                t = t_name.split('2')[0]
                t_name = t
            if "1" in t_name:
                t = t_name.split('1')[0]
                t_name = t
            if "Janeiro" in t_name:
                t_name = "Rio de Janeiro"
            if "Hertogenbosch" in t_name:
                t_name = "\'s-Hertogenbosch"
            if "Petersburg" in t_name:
                t_name = "St. Petersburg"
            if "Lumpur" in t_name:
                t_name = "Kuala Lumpur"
            if "Louis" in t_name:
                t_name = "St. Louis"
            if "Olympics" in t_name:
                t_name = "Olympics"
            if "Del Mar" in t_name or "del Mar" in t_name:
                t_name = "Vina Del Mar"
            if "Sopot" in t_name:
                t_name = "Sopot"
            if "Minh" in t_name:
                t_name = "Ho Chi Minh"
            if "Tour" in t_name:
                t_name = "ATP Finals"
            if "Rijn" in t_name:
                t_name = "Alphen Aan Den Rijn"
            if "Remy" in t_name:
                t_name = "St. Remy"
            if "Tronto" in t_name:
                t_name = "San Benedetto del Tronto"
            if "-" in t_name and t_name.split('-')[1] is '':
                t_name = t_name.split('-')[0]
            
            filewriter_map_atp.writerow([str(row['tourney_name']), t_name])
            df_t.set_value(index, 'tourney_name', t_name)
            
    return df_t   


def set_new_tourney_names_wta(df_t):
    with open('D:\\EXPERT\\lab_se\\TennisPrediction\\wta_tourney_map.csv', 'w') as map_atp:
        filewriter_map_atp = csv.writer(map_atp, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter_map_atp.writerow(['old_name', 'new_name'])

        for index, row in df_t.iterrows():
            t_name = str(row['tourney_name'])
            if "Fed Cup" in t_name:
                t_name = "Fed Cup"
            if "Hertogenbosch" in t_name:
                t_name = "\'s-Hertogenbosch"
            if "Indian" in t_name:
                t_name = "Indian Wells"
            if "Olympics" in t_name or "Olympic" in t_name:
                t_name = "Olympics"
            if "French" in t_name  or "Paris" in t_name:
                t_name = "Roland Garros"
            if "Australia Circuit" in t_name:
                t_name = "Australia Circuit"
            if "Mexico Circuit" in t_name:
                t_name = "Mexico Circuit"
            if "Taipei" in t_name:
                t_name = "Taipei"
            if "WTA" in t_name:
                t_name = "WTA Finals"
            if "Dubai" in t_name:
                t_name = "Dubai"
            if "Us Open" in t_name:
                t_name = "US Open"

            filewriter_map_atp.writerow([str(row['tourney_name']), t_name])
            df_t.set_value(index, 'tourney_name', t_name)
            
    return df_t   

def main():
    create_unique_tourneys()


if __name__ == "__main__":
    main()