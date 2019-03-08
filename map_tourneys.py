import csv
import pandas as pd
import os

ATP_path = "D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\tennis_atp\\matches"
WTA_path = "D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\tennis_wta"

def create_unique_tourneys():
    # ATP
    os.chdir(ATP_path)
    all_atp = pd.concat([pd.read_csv(f, engine='python') for f in os.listdir(os.curdir) if 'futures' not in f], sort=True)
    t = all_atp['tourney_name'].tolist()
    t_df = pd.DataFrame(t)
    u_df = pd.DataFrame(t_df[0].unique())
    u_df.rename(columns = {'': 'id', '0': 'tourney_name'}, inplace = True)
    u_df.to_csv(os.path.join(os.curdir,'unique_tourneys_atp.csv'))
    # WTA
    os.chdir(WTA_path)
    all_wta = pd.concat([pd.read_csv(f, engine='python') for f in os.listdir(os.curdir) if 'futures' not in f], sort=True)
    t = all_wta['tourney_name'].tolist()
    t_df = pd.DataFrame(t)
    u_df = pd.DataFrame(t_df[0].unique())
    u_df.rename(columns = {'': 'id', '0': 'tourney_name'}, inplace = True)
    u_df.to_csv(os.path.join(os.curdir,'unique_tourneys_wta.csv'))


def map_tourneys_atp():

    with open('D:\\college\\an4CTI\\SEMESTER_2\\SE\\lab\\tourney_map_atp.csv', 'w') as map_tourneys_atp:
        filewriter_map_tourneys_atp = csv.writer(map_tourneys_atp, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter_map_tourneys_atp.writerow(['tourney_id_original', 'tourney_id_new', 
                                            'tourney_name_original', 'tourney_name_new'])
        
        map_tourneys_atp_df = pd.DataFrame(columns=['tourney_id_original', 'tourney_id_new', 
                                            'tourney_name_original', 'tourney_name_new'])
        
        k = 0
        for file in os.listdir(ATP_path):
            # if "atp_matches" in file and "2015" not in file and "2016" not in file and "2017" not in file:
            if "atp_matches" in file and "futures" not in file:
                read_file = pd.read_csv(os.path.join(ATP_path, file), parse_dates=False, encoding = "ISO-8859-1")
                print("Processing ... " + os.path.join(ATP_path, file))
                
                for index, row in read_file.iterrows():

                    new_tourney_name = get_new_tourney_name(row['tourney_name'])


def get_new_tourney_name(old_name):
    return ""


def main():
    create_unique_tourneys()
    # map_tourneys_atp()
    # read_datasets_wta()


if __name__ == "__main__":
    main()