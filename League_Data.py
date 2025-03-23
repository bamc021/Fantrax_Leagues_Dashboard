import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title = 'Fantrax League Stats',layout='wide')

def safe_request(url, method='get', **kwargs):
    try:
        if method == 'get':
            response = requests.get(url, **kwargs)
        elif method == 'post':
            response = requests.post(url, **kwargs)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {url}: {e}")
        return None

def MLB_stats_import():
    leadingcols = ['PlayerName','TeamNameAbb','playerid','Fantrax_ID']

    MLB_hitting_request = safe_request(url="https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=bat&lg=all&qual=0&season=2024&season1=2024&startdate=2024-03-01&enddate=2024-11-01&month=0&hand=&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR")
    MLB_hitting_response = MLB_hitting_request
    MLB_hitting_df = pd.json_normalize(MLB_hitting_response['data'])
    MLB_hitting_df.drop(['Name','Team','TeamName','PlayerNameRoute','position','teamid'],axis=1,inplace=True)
    MLB_hitting_df = (
        MLB_hitting_df[MLB_hitting_df['playerid'].isin(ss.idkey['IDFANGRAPHS']) & (MLB_hitting_df['PA'] > 0)]
        .reset_index(drop=True)
    )
    duplicates = ss.idkey[ss.idkey['IDFANGRAPHS'].duplicated()]
    if not duplicates.empty:
        print("Duplicate entries in 'IDFANGRAPHS':")
        print(duplicates)
    MLB_hitting_df['Fantrax_ID'] = MLB_hitting_df.playerid.map(ss.idkey.set_index('IDFANGRAPHS')['FANTRAXID'])
    MLB_hitting_df['Swings'] = round(MLB_hitting_df['Pitches']*MLB_hitting_df['Swing%'])

    nonleadingcols = [col for col in MLB_hitting_df.columns if col not in leadingcols]
    MLB_hitting_df = MLB_hitting_df[leadingcols + nonleadingcols]
    MLB_hitting_df = MLB_hitting_df.rename(columns={'PlayerName':'Player','TeamNameAbb':'MLBTeam'})

    MLB_pitching_request = safe_request(url="https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&qual=0&season=2024&season1=2024&startdate=2024-03-01&enddate=2024-11-01&month=0&hand=&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR")
    MLB_pitching_response = MLB_pitching_request
    MLB_pitching_df = pd.json_normalize(MLB_pitching_response['data'])
    MLB_pitching_df.drop(['Name','Team','TeamName','PlayerNameRoute','position','teamid'],axis=1,inplace=True)
    MLB_pitching_df = (
        MLB_pitching_df[MLB_pitching_df['playerid'].isin(ss.idkey['IDFANGRAPHS']) & (MLB_pitching_df['IP'] > 0)]
        .reset_index(drop=True)
    )
    MLB_pitching_df['Fantrax_ID'] = MLB_pitching_df.playerid.map(ss.idkey.set_index('IDFANGRAPHS')['FANTRAXID'])
    
    MLB_pitching_df['Swings'] = round(MLB_pitching_df['Pitches']*MLB_pitching_df['Swing%'])

    nonleadingcols = [col for col in MLB_pitching_df.columns if col not in leadingcols]
    MLB_pitching_df = MLB_pitching_df[leadingcols + nonleadingcols]
    MLB_pitching_df = MLB_pitching_df.rename(columns={'PlayerName':'Player','TeamNameAbb':'MLBTeam','sp_stuff':'Stuff+',
                                                      'sp_location':'Location+','sp_pitching':'Pitching+'})

    return [MLB_pitching_df, MLB_hitting_df]

def minors_stats_import():
    leadingcols = ['PlayerName','AffAbbName','aLevel','Age','playerids','Fantrax_ID']

    minors_hitting_request = safe_request(url="https://www.fangraphs.com/api/leaders/minor-league/data?pos=all&level=0&lg=2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32&stats=bat&qual=0&type=0&team=&season=2024&seasonEnd=2024&org=&ind=0&splitTeam=false")
    minors_hitting_response = minors_hitting_request
    minors_hitting_df = pd.DataFrame(minors_hitting_response)
    minors_hitting_df.drop(['Name','Team','TeamName','AffId'],axis=1,inplace=True)
    minors_hitting_df = (
        minors_hitting_df[minors_hitting_df['playerids'].isin(ss.idkey['IDFANGRAPHS']) & (minors_hitting_df['PA'] > 0)]
        .reset_index(drop=True)
    )
    minors_hitting_df['Fantrax_ID'] = minors_hitting_df.playerids.map(ss.idkey.set_index('IDFANGRAPHS')['FANTRAXID'])

    nonleadingcols = [col for col in minors_hitting_df.columns if col not in leadingcols]
    minors_hitting_df = minors_hitting_df[leadingcols + nonleadingcols]
    minors_hitting_df = minors_hitting_df.rename(columns={'PlayerName':'Player','AffAbbName':'Org','aLevel':'Level'})

    minors_pitching_request = safe_request(url="https://www.fangraphs.com/api/leaders/minor-league/data?pos=all&level=0&lg=2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32&stats=pit&qual=0&type=0&team=&season=2024&seasonEnd=2024&org=&ind=0&splitTeam=false")
    minors_pitching_response = minors_pitching_request
    minors_pitching_df = pd.DataFrame(minors_pitching_response)
    minors_pitching_df.drop(['Name','Team','TeamName','AffId'],axis=1,inplace=True)
    minors_pitching_df = (
        minors_pitching_df[minors_pitching_df['playerids'].isin(ss.idkey['IDFANGRAPHS']) & (minors_pitching_df['IP'] > 0)]
        .reset_index(drop=True)
    )
    minors_pitching_df['Fantrax_ID'] = minors_pitching_df.playerids.map(ss.idkey.set_index('IDFANGRAPHS')['FANTRAXID'])

    nonleadingcols = [col for col in minors_pitching_df.columns if col not in leadingcols]
    minors_pitching_df = minors_pitching_df[leadingcols + nonleadingcols]
    minors_pitching_df = minors_pitching_df.rename(columns={'PlayerName':'Player','AffAbbName':'Org','aLevel':'Level'})

    return [minors_pitching_df, minors_hitting_df]

def league_list():
    league_list_request = safe_request(url="https://www.fantrax.com/fxea/general/getLeagues?userSecretId=xshrve29lrtpb475")
    league_list_response = league_list_request
    league_list_df = pd.json_normalize(league_list_response['leagues'])
    league_list_df = league_list_df[['leagueName','leagueId']]

    return league_list_df

def app_startup():
    [ss.MLB_pitching_df, ss.MLB_hitting_df] = MLB_stats_import()
    [ss.minors_pitching_df, ss.minors_hitting_df] = minors_stats_import()
    ss.league_list_df = league_list()

    # Ensure that rosters_df is initialized
    if "league_name" in ss and ss.league_name is not None:
        league_id = ss.league_list_df.loc[ss.league_list_df['leagueName'] == ss.league_name, 'leagueId'].iloc[0]
        ss.rosters_df, ss.activepitchers, ss.activehitters = rosters(league_id)
    else:
        # Initialize rosters_df as an empty DataFrame if league_name is not set
        ss.rosters_df = pd.DataFrame()

    return ss.MLB_pitching_df, ss.MLB_hitting_df, ss.minors_pitching_df, ss.minors_hitting_df, ss.league_list_df

# for l in league_list_df['leagueId']:
def rosters(league_id):
    league_url = 'https://www.fantrax.com/fxea/general/getTeamRosters?leagueId=' + league_id
    rosters_request = safe_request(url=league_url)
    rosters_response = rosters_request
    
    # Check if the response contains the expected data
    if 'rosters' not in rosters_response:
        raise ValueError("No rosters found in the response.")

    roster_info = pd.json_normalize(rosters_response['rosters'])
    
    # Check the structure of the roster_info DataFrame
    # print("Roster Info Columns:", roster_info.columns)

    team_names = roster_info.iloc[:, ::2].values.flatten()  # Odd-indexed columns (1, 3, 5, ...)
    roster_items = roster_info.iloc[:, 1::2].values.flatten()  # Even-indexed columns (2, 4, 6, ...)
    roster_info = pd.DataFrame({'teamName': team_names, 'rosterItems': roster_items})

    # Normalize the 'rosterItems' column
    rosters_df = pd.json_normalize(roster_info.explode('rosterItems')['rosterItems'])

    # Repeat the teamName values to match the number of rows in the normalized data
    rosters_df['teamName'] = roster_info.explode('rosterItems')['teamName'].values

    # Check the structure of the rosters_df DataFrame
    # print("Rosters DataFrame Columns:", rosters_df.columns)

    # Ensure 'id' and 'teamName' columns exist before proceeding
    if 'id' not in rosters_df.columns or 'teamName' not in rosters_df.columns:
        raise KeyError("Expected columns 'id' or 'teamName' not found in rosters_df.")

    rosters_df['Name'] = rosters_df.id.map(ss.idkey.set_index('FANTRAXID')['FANTRAXNAME'])
    activepitchers = rosters_df.loc[
        (rosters_df['status'] == "ACTIVE") & (rosters_df['position'].isin(["P", "SP", "RP"])) |
        (rosters_df['id'] == "02yc4")
    ]
    activehitters = rosters_df.loc[
        ((rosters_df['status'] == "ACTIVE") & (~rosters_df['position'].isin(["P", "SP", "RP"]))) |
        (rosters_df['id'] == "02yc4")
    ]

    # print(activehitters)

    return rosters_df, activepitchers, activehitters

def sum_team_stats(df, columns_to_sum, output_df_name, globals_dict):
    # Sum specified columns grouped by 'teamName'
    summed_df = df.groupby('teamName')[columns_to_sum].sum().reset_index()
    
    # Check if output dataframe already exists
    if output_df_name in globals_dict:
        existing_df = globals_dict[output_df_name]
        existing_df = existing_df.set_index('teamName')
        summed_df = summed_df.set_index('teamName')
        existing_df.update(summed_df)
        summed_df = existing_df.reset_index()
    
    # Store in globals dictionary
    globals_dict[output_df_name] = summed_df
    
    return summed_df

def calculate_team_rate_statistics(df, rate_columns, denominator_col, output_df_name, globals_dict):
    """
    Calculate team rate statistics based on active players' statistics.

    Parameters:
    - df: DataFrame containing player statistics.
    - rate_columns: List of columns to calculate rates for.
    - denominator_col: Column used as the denominator for rate calculations.
    - output_df_name: Name of the output DataFrame to store results.
    - globals_dict: Dictionary to store global variables.

    Returns:
    - DataFrame with calculated team rate statistics.
    """
    # Ensure the denominator column exists
    if denominator_col not in df.columns:
        raise ValueError(f"Denominator column '{denominator_col}' not found in DataFrame.")

    # Group the DataFrame by 'teamName'
    grouped = df.groupby('teamName')

    # Initialize a new DataFrame to store the results
    rate_df = pd.DataFrame({'teamName': grouped.groups.keys()})

    # Calculate the denominator sums
    denominator_sums = grouped[denominator_col].sum()

    # Avoid division by zero by replacing zeros with NaN
    denominator_sums.replace(0, float('nan'), inplace=True)

    # Calculate rate statistics for each specified column
    for col in rate_columns:
        if col not in df.columns:
            continue  # Skip if the column is not found
        
        # Calculate the numerator as the sum of (rate column * denominator column) for each team
        numerator_sums = grouped.apply(lambda x: (x[col] * x[denominator_col]).sum())

        # Create a DataFrame to combine numerator and denominator sums
        sums_df = pd.DataFrame({
            'teamName': numerator_sums.index,
            'numerator': numerator_sums.values,
            'denominator': denominator_sums.values
        })

        # Calculate the rate and add it as a new column
        sums_df['rate'] = sums_df['numerator'] / sums_df['denominator']

        # Store the calculated rates in rate_df
        rate_df[col] = sums_df['rate']

    # Merge with existing dataframe if it exists
    if output_df_name in globals_dict:
        existing_df = globals_dict[output_df_name]
        merged_df = existing_df.merge(rate_df, on='teamName', how='left')
    else:
        merged_df = rate_df

    # Store the results in the globals dictionary
    globals_dict[output_df_name] = merged_df

    return merged_df

def league_selected(league_name):
    # Ensure rosters_df is available before proceeding
    if 'rosters_df' not in ss or ss.rosters_df.empty:
        league_id = ss.league_list_df.loc[ss.league_list_df['leagueName'] == league_name, 'leagueId'].iloc[0]
        ss.rosters_df, ss.activepitchers, ss.activehitters = rosters(league_id)

    league_id = ss.league_list_df.loc[ss.league_list_df['leagueName'] == league_name, 'leagueId'].iloc[0]

    [ss.rosters_df, ss.activepitchers, ss.activehitters] = rosters(league_id)

    rostered_hitters = ss.MLB_hitting_df.merge(
        ss.rosters_df[['id', 'teamName']], 
        left_on='Fantrax_ID', 
        right_on='id', 
        how='left'
    )

    rostered_hitters.drop(columns=['id'], inplace=True)

    MLBhittingdisplay = (
        rostered_hitters[rostered_hitters['Fantrax_ID'].isin(ss.rosters_df['id'])]
        .reset_index(drop=True)
    )

    ss.MLB_hitting_df = ss.MLB_hitting_df.merge(
        ss.activehitters[['id', 'teamName']], 
        left_on='Fantrax_ID', 
        right_on='id', 
        how='left'
    )

    # Drop 'id' column after merge if it's no longer needed
    ss.MLB_hitting_df.drop(columns=['id'], inplace=True)

    activehitting = (
        ss.MLB_hitting_df[ss.MLB_hitting_df['Fantrax_ID'].isin(ss.activehitters['id'])]
        .reset_index(drop=True)
    )

    ss.MLB_pitching_df = ss.MLB_pitching_df.merge(
        ss.activepitchers[['id', 'teamName']], 
        left_on='Fantrax_ID', 
        right_on='id', 
        how='left'
    )

    # Drop 'id' column after merge if it's no longer needed
    ss.MLB_pitching_df.drop(columns=['id'], inplace=True)

    activepitching = (
        ss.MLB_pitching_df[ss.MLB_pitching_df['Fantrax_ID'].isin(ss.activepitchers['id'])]
        .reset_index(drop=True)
    )

    teamhitting_vars = activehitting.columns.tolist()
    teampitching_vars = activepitching.columns.tolist()

    teamhitting_vars = [e for e in teamhitting_vars if e not in ('playerid','playerids','Fantrax_ID','Bats','xMLBAMID','Season','AgeR','SeasonMin','SeasonMax','GDP','GB','FB','LD','IFFB',
                                                            'Balls','Strikes','1B','IFH','BU','BUH','BB/K','IFH%','BUH%','TTO%','wRAA','wRC','Batting','Fielding','Replacement',
                                                            'Positional','wLeague','CFraming','Defense','Offense','RAR','WAROld','Dollars','BaseRunning','Spd','wBsR','WPA','-WPA',
                                                            '+WPA','RE24','REW','pLI','phLI','PH','WPA/LI','Clutch','FB%1','FBv','SL%','SLv','CT%','CTv','CB%','CBv','CH%','CHv','SF%',
                                                            'SFv','KN%','KNv','XX%','PO%','wFB','wSL','wCT','wCB','wCH','wSF','wKN','wFB/C','wSL/C','wCT/C','wCB/C','wCH/C','wSF/C',
                                                            'wKN/C','F-Strike%','CStr%','Pull','Cent','Oppo','Soft','Med','Hard','bipCount','Soft%','Med%','UBR','GDPRuns','AVG+',
                                                            'BB%+','K%+','OBP+','SLG+','ISO+','BABIP+','LD%+','GB%+','FB%+','HRFB%+','Pull%+','Cent%+','Oppo%+','Soft%+',
                                                            'Med%+','Hard%+','XBR','PPTV','CPTV','BPTV','DSV','DGV','BTV','rPPTV','rCPTV','rBPTV','rDSV','rDGV','rBTV','EBV',
                                                            'ESV','rFTeamV','rBTeamV','rTV','pfxFA%','pfxFT%','pfxFC%','pfxFS%','pfxFO%','pfxSI%','pfxSL%','pfxCU%','pfxKC%',
                                                            'pfxEP%','pfxCH%','pfxSC%','pfxKN%','pfxUN%','pfxvFA','pfxvFT','pfxvFC','pfxvFS','pfxvFO','pfxvSI','pfxvSL',
                                                            'pfxvCU','pfxvKC','pfxvEP','pfxvCH','pfxvSC','pfxvKN','pfxFA-X','pfxFT-X','pfxFC-X','pfxFS-X','pfxFO-X',
                                                            'pfxSI-X','pfxSL-X','pfxCU-X','pfxKC-X','pfxEP-X','pfxCH-X','pfxSC-X','pfxKN-X','pfxFA-Z','pfxFT-Z',
                                                            'pfxFC-Z','pfxFS-Z','pfxFO-Z','pfxSI-Z','pfxSL-Z','pfxCU-Z','pfxKC-Z','pfxEP-Z','pfxCH-Z','pfxSC-Z',
                                                            'pfxKN-Z','pfxwFA','pfxwFT','pfxwFC','pfxwFS','pfxwFO','pfxwSI','pfxwSL','pfxwCU','pfxwKC','pfxwEP',
                                                            'pfxwCH','pfxwSC','pfxwKN','pfxwFA/C','pfxwFT/C','pfxwFC/C','pfxwFS/C','pfxwFO/C','pfxwSI/C','pfxwSL/C',
                                                            'pfxwCU/C','pfxwKC/C','pfxwEP/C','pfxwCH/C','pfxwSC/C','pfxwKN/C','pfxO-Swing%','pfxZ-Swing%','pfxSwing%',
                                                            'pfxO-Contact%','pfxZ-Contact%','pfxContact%','pfxZone%','pfxPace','piCH%','piCS%','piCU%','piFA%',
                                                            'piFC%','piFS%','piKN%','piSB%','piSI%','piSL%','piXX%','pivCH','pivCS','pivCU','pivFA',
                                                            'pivFC','pivFS','pivKN','pivSB','pivSI','pivSL','pivXX','piCH-X','piCS-X','piCU-X','piFA-X','piFC-X',
                                                            'piFS-X','piKN-X','piSB-X','piSI-X','piSL-X','piXX-X','piCH-Z','piCS-Z','piCU-Z','piFA-Z',
                                                            'piFC-Z','piFS-Z','piKN-Z','piSB-Z','piSI-Z','piSL-Z','piXX-Z','piwCH','piwCS','piwCU','piwFA','piwFC',
                                                            'piwFS','piwKN','piwSB','piwSI','piwSL','piwXX','piwCH/C','piwCS/C','piwCU/C','piwFA/C','piwFC/C',
                                                            'piwFS/C','piwKN/C','piwSB/C','piwSI/C','piwSL/C','piwXX/C','piO-Swing%','piZ-Swing%','piSwing%',
                                                            'piO-Contact%','piZ-Contact%','piContact%','piZone%','piPace','Barrels','HardHit','Q','TG')]
    teampitching_vars = [g for g in teampitching_vars if g not in ('playerid','playerids','Fantrax_ID','Throws','xMLBAMID','Season','AgeR','SeasonMin','SeasonMax','IBB','BK','GB','FB','LD','IFFB',
                                                               'Balls','Strikes','RS','IFH','BU','BUH','IFH%','BUH%','TTO%','CFraming','Starting','Start-IP','Relieving',
                                                               'Relief-IP','RAR','Dollars','RA9-Wins','LOB-Wins','BIP-Wins','BS-Wins','tERA','WPA','-WPA','+WPA','RE24','REW',
                                                               'pLI','inLI','gmLI','exLI','Pulls','Games','WPA/LI','Clutch','SL%','SLv','CT%','CTv','CB%','CBv','CH%','CHv',
                                                               'SF%','SFv','KN%','KNv','XX%','PO%','wFB','wSL','wCT','wCB','wCH','wSF','wKN','wFB/C','wSL/C','wCT/C',
                                                               'wCB/C','wCH/C','wSF/C','wKN/C','F-Strike%','CStr%','SD','MD','ERA-','FIP-','xFIP-','kwERA','RS/9',
                                                               'E-F','Pull','Cent','Oppo','Soft','Med','Hard','bipCount','Pull%','Cent%','Oppo%','Soft%','Med%','Hard%',
                                                               'K/9+','BB/9+','K/BB+','H/9+','HR/9+','AVG+','WHIP+','BABIP+','LOB%+','K%+','BB%+','LD%+','GB%+','FB%+',
                                                               'HRFB%+','Pull%+','Cent%+','Oppo%+','Soft%+','Med%+','Hard%+','pb_o_CH','pb_s_CH','pb_c_CH','pb_o_CU',
                                                               'pb_s_CU','pb_c_CU','pb_o_FF','pb_s_FF','pb_c_FF','pb_o_SI','pb_s_SI','pb_c_SI','pb_o_SL','pb_s_SL',
                                                               'pb_c_SL','pb_o_KC','pb_s_KC','pb_c_KC','pb_o_FC','pb_s_FC','pb_c_FC','pb_o_FS','pb_s_FS',
                                                               'pb_c_FS','pb_overall','pb_stuff','pb_command','pb_xRV100','pb_ERA','sp_s_CH','sp_l_CH',
                                                               'sp_p_CH','sp_s_CU','sp_l_CU','sp_p_CU','sp_s_FF','sp_l_FF','sp_p_FF','sp_s_SI',
                                                               'sp_l_SI','sp_p_SI','sp_s_SL','sp_l_SL','sp_p_SL','sp_s_KC','sp_l_KC','sp_p_KC','sp_s_FC',
                                                               'sp_l_FC','sp_p_FC','sp_s_FS','sp_l_FS','sp_p_FS','sp_s_FO','sp_l_FO','sp_p_FO','PPTV',
                                                               'CPTV','BPTV','DSV','DGV','BTV','rPPTV','rCPTV','rBPTV','rDSV','rDGV','rBTV','EBV','ESV',
                                                               'rFTeamV','rBTeamV','rTV','pfxFA%','pfxFT%','pfxFC%','pfxFS%','pfxFO%','pfxSI%','pfxSL%',
                                                               'pfxCU%','pfxKC%','pfxEP%','pfxCH%','pfxSC%','pfxKN%','pfxUN%','pfxvFA','pfxvFT','pfxvFC',
                                                               'pfxvFS','pfxvFO','pfxvSI','pfxvSL','pfxvCU','pfxvKC','pfxvEP','pfxvCH','pfxvSC','pfxvKN',
                                                               'pfxFA-X','pfxFT-X','pfxFC-X','pfxFS-X','pfxFO-X','pfxSI-X','pfxSL-X','pfxCU-X','pfxKC-X',
                                                               'pfxEP-X','pfxCH-X','pfxSC-X','pfxKN-X','pfxFA-Z','pfxFT-Z','pfxFC-Z','pfxFS-Z','pfxFO-Z',
                                                               'pfxSI-Z','pfxSL-Z','pfxCU-Z','pfxKC-Z','pfxEP-Z','pfxCH-Z','pfxSC-Z','pfxKN-Z','pfxwFA',
                                                               'pfxwFT','pfxwFC','pfxwFS','pfxwFO','pfxwSI','pfxwSL','pfxwCU','pfxwKC','pfxwEP',
                                                               'pfxwCH','pfxwSC','pfxwKN','pfxwFA/C','pfxwFT/C','pfxwFC/C','pfxwFS/C','pfxwFO/C',
                                                               'pfxwSI/C','pfxwSL/C','pfxwCU/C','pfxwKC/C','pfxwEP/C','pfxwCH/C','pfxwSC/C','pfxwKN/C',
                                                               'pfxO-Swing%','pfxZ-Swing%','pfxSwing%','pfxO-Contact%','pfxZ-Contact%','pfxContact%','pfxZone%',
                                                               'pfxPace','piCH%','piCS%','piCU%','piFA%','piFC%','piFS%','piKN%','piSB%','piSI%',
                                                               'piSL%','piXX%','pivCH','pivCS','pivCU','pivFA','pivFC','pivFS','pivKN','pivSB',
                                                               'pivSI','pivSL','pivXX','piCH-X','piCS-X','piCU-X','piFA-X','piFC-X','piFS-X','piKN-X',
                                                               'piSB-X','piSI-X','piSL-X','piXX-X','piCH-Z','piCS-Z','piCU-Z','piFA-Z','piFC-Z',
                                                               'piFS-Z','piKN-Z','piSB-Z','piSI-Z','piSL-Z','piXX-Z','piwCH','piwCS','piwCU','piwFA',
                                                               'piwFC','piwFS','piwKN','piwSB','piwSI','piwSL','piwXX','piwCH/C','piwCS/C',
                                                               'piwCU/C','piwFA/C','piwFS/C','piwKN/C','piwSB/C','piwSI/C','piwSL/C','piwXX/C',
                                                               'piO-Swing%','piZ-Swing%','piSwing%','piO-Contact%','piZ-Contact%','piContact%','piZone%',
                                                               'piPace','Barrels','HardHit','Q','TG')]

    activehitting = activehitting.drop(columns=[col for col in activehitting.columns if col not in teamhitting_vars])
    activepitching = activepitching.drop(columns=[col for col in activepitching.columns if col not in teampitching_vars])

    MLBhittingdisplay = MLBhittingdisplay.drop(columns=[col for col in MLBhittingdisplay.columns if col not in teamhitting_vars])

    hitplotvars = [h for h in teamhitting_vars if h not in ('Player','MLBTeam','teamName')]
    pitchplotvars = [p for p in teampitching_vars if p not in ('Player','MLBTeam','teamName')]

    ss.globals_dict = {}

    sum_team_stats(activehitting, ['PA','H','2B','3B','HR','R','RBI','BB','SO','SB','CS'], 'teamhitting', ss.globals_dict)
    calculate_team_rate_statistics(activehitting, ['AVG','SLG','ISO','xAVG','xSLG'], 'AB', 'teamhitting', ss.globals_dict)
    calculate_team_rate_statistics(activehitting, ['OBP','BB%','K%','wOBA','wRC+','xwOBA'], 'PA', 'teamhitting', ss.globals_dict)
    calculate_team_rate_statistics(activehitting, ['Contact%'], 'Swings', 'teamhitting', ss.globals_dict)
    calculate_team_rate_statistics(activehitting, ['EV','Barrel%','HardHit%'], 'Events', 'teamhitting', ss.globals_dict)
    calculate_team_rate_statistics(activehitting, ['SwStr%','C+SwStr%'], 'Pitches', 'teamhitting', ss.globals_dict)


    sum_team_stats(ss.MLB_pitching_df, ['W','L','QS','CG','SV','BS','HLD','TIP','TBF','H','ER','HR','BB','SO'], 'teampitching', ss.globals_dict)
    calculate_team_rate_statistics(ss.MLB_pitching_df, ['ERA','WHIP','FIP','xERA','xFIP','SIERA'], 'TIP', 'teampitching', ss.globals_dict)
    calculate_team_rate_statistics(ss.MLB_pitching_df, ['GB%','FB%','EV','Barrel%','HardHit%'], 'Events', 'teampitching', ss.globals_dict)
    calculate_team_rate_statistics(ss.MLB_pitching_df, ['Zone%','SwStr%','C+SwStr%','Stuff+','Location+','Pitching+'], 'Pitches', 'teampitching', ss.globals_dict)
    calculate_team_rate_statistics(ss.MLB_pitching_df, ['Contact%'], 'Swings', 'teampitching', ss.globals_dict)
    calculate_team_rate_statistics(ss.MLB_pitching_df, ['K%','BB%'], 'TBF', 'teampitching', ss.globals_dict)

    # print(ss.globals_dict['teamhitting'])

    teamnamecolumn = MLBhittingdisplay.pop('teamName')
    MLBhittingdisplay.insert(1,teamnamecolumn.name,teamnamecolumn)

    teamnamecolumn = activepitching.pop('teamName')
    activepitching.insert(1,teamnamecolumn.name,teamnamecolumn)

    hitplotvars = [e for e in teamhitting_vars if e not in ('ID','Pos','Player','MLBTeam','Eligible','Status','Opponent','FANTRAXID','IDFANGRAPHS','MLBID','team_name')]
    pitchplotvars = [g for g in teampitching_vars if g not in ('ID','Pos','Player','MLBTeam','Eligible','Status','Opponent','FANTRAXID','IDFANGRAPHS','MLBID','team_name')]

    # Filter minors_hitting_df and minors_pitching_df to include only players on the roster
    rostered_minors_hitting = ss.minors_hitting_df.merge(
        ss.rosters_df[['id', 'teamName']], 
        left_on='Fantrax_ID', 
        right_on='id', 
        how='inner'  # Use inner join to keep only players on the roster
    )

    rostered_minors_pitching = ss.minors_pitching_df.merge(
        ss.rosters_df[['id', 'teamName']], 
        left_on='Fantrax_ID', 
        right_on='id', 
        how='inner'  # Use inner join to keep only players on the roster
    )

    # Remove all columns that contain 'id' in their name and the 'season' column
    rostered_minors_hitting = rostered_minors_hitting.loc[:, ~rostered_minors_hitting.columns.str.contains('id', case=False)]
    rostered_minors_hitting = rostered_minors_hitting.drop(columns=['season'], errors='ignore')

    rostered_minors_pitching = rostered_minors_pitching.loc[:, ~rostered_minors_pitching.columns.str.contains('id', case=False)]
    rostered_minors_pitching = rostered_minors_pitching.drop(columns=['season'], errors='ignore')

    # Reorder columns to move 'teamName' to the second position
    rostered_minors_hitting = rostered_minors_hitting[['Player', 'teamName'] + [col for col in rostered_minors_hitting.columns if col != 'teamName' and col != 'Player']]
    rostered_minors_pitching = rostered_minors_pitching[['Player', 'teamName'] + [col for col in rostered_minors_pitching.columns if col != 'teamName' and col != 'Player']]

    # # Update the session state with the filtered dataframes
    # ss.minors_hitting_df = rostered_minors_hitting
    # ss.minors_pitching_df = rostered_minors_pitching

    tab1, tab2, tab3, tab4 = st.tabs(['League Data','MLB Stats','MiLB Stats','Troubleshoot'])

    with tab1:
        col1,col2=st.columns(2)

        with col1:
            @st.fragment
            def hitaxes_select():
                hitting_columns = [col for col in ss.globals_dict['teamhitting'].columns if col not in ['Name', 'teamName']]
                col1_1,col1_2=st.columns(2)
                with col1_1:
                    ss.teamhit_xaxis = st.selectbox('X Axis',hitting_columns,index=4)
                with col1_2:
                    ss.teamhit_yaxis = st.selectbox('Y Axis',hitting_columns,index=9)
                teamhitfig = px.scatter(ss.globals_dict['teamhitting'],x=ss.teamhit_xaxis,y=ss.teamhit_yaxis,
                                        color='teamName',color_discrete_sequence=px.colors.qualitative.Light24)
                st.plotly_chart(teamhitfig)
            hitaxes_select()
        with col2:
            st.header("Team Hitting Statistics")
            st.dataframe(ss.globals_dict['teamhitting'],hide_index=True)


        col3,col4=st.columns(2)

        with col3:        
            @st.fragment
            def pitchaxes_select():
                pitching_columns = [col for col in ss.globals_dict['teampitching'].columns if col not in ['Name', 'teamName']]
                col3_1,col3_2=st.columns(2)
                with col3_1:
                    ss.teampitch_xaxis = st.selectbox('X Axis',pitching_columns,index=13)
                with col3_2:
                    ss.teampitch_yaxis = st.selectbox('Y Axis',pitching_columns,index=14)
                teampitchfig = px.scatter(ss.globals_dict['teampitching'],x=ss.teampitch_xaxis,y=ss.teampitch_yaxis,
                                          color='teamName',color_discrete_sequence=px.colors.qualitative.Light24)
                st.plotly_chart(teampitchfig)
            pitchaxes_select()
        with col4:
            st.header("Team Pitching Statistics")
            st.dataframe(ss.globals_dict['teampitching'],hide_index=True)
    
    with tab2:
        st.header('Major League Hitting')
        @st.fragment
        def majorhitaxes_select():
            col1, col2 = st.columns(2)
            with col1:
                majorhit_xaxis = st.selectbox('X Axis',hitplotvars,index=7)
            with col2:
                majorhit_yaxis = st.selectbox('Y Axis',hitplotvars,index=16)

            majorhitfig = px.scatter(MLBhittingdisplay,x=majorhit_xaxis,y=majorhit_yaxis,color='teamName',
                                    color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
            st.plotly_chart(majorhitfig)
            st.dataframe(MLBhittingdisplay,hide_index=True)
        
        majorhitaxes_select()

        st.header('Major League Pitching')
        @st.fragment
        def majorpitchaxes_select():
            col3,col4 = st.columns(2)
            with col3:
                majorpitch_xaxis = st.selectbox('X Axis',pitchplotvars,index=20)
            with col4:
                majorpitch_yaxis = st.selectbox('Y Axis',pitchplotvars,index=3)

            majorpitchfig = px.scatter(activepitching,x=majorpitch_xaxis,y=majorpitch_yaxis,color='teamName',
                                    color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
            st.plotly_chart(majorpitchfig)
            st.dataframe(activepitching,hide_index=True)
        
        majorpitchaxes_select()
    
    with tab3:
        st.header('Minor League Hitting')
        @st.fragment
        def minorhitaxes_select():
            col1, col2 = st.columns(2)
            with col1:
                minorhit_xaxis = st.selectbox('X Axis',hitplotvars,index=7,key='minor_hit_x')
            with col2:
                minorhit_yaxis = st.selectbox('Y Axis',hitplotvars,index=16,key='minor_hit_y')

            minorhitfig = px.scatter(rostered_minors_hitting,x=minorhit_xaxis,y=minorhit_yaxis,color='teamName',
                                    color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
            st.plotly_chart(minorhitfig)
            st.dataframe(rostered_minors_hitting,hide_index=True)
        
        minorhitaxes_select()

        st.header('Minor League Pitching')
        @st.fragment
        def minorpitchaxes_select():
            col3,col4 = st.columns(2)
            with col3:
                minorpitch_xaxis = st.selectbox('X Axis',pitchplotvars,index=20,key='minor_pitch_x')
            with col4:
                minorpitch_yaxis = st.selectbox('Y Axis',pitchplotvars,index=3,key='minor_pitch_y')

            minorpitchfig = px.scatter(rostered_minors_pitching,x=minorpitch_xaxis,y=minorpitch_yaxis,color='teamName',
                                    color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
            st.plotly_chart(minorpitchfig)
            st.dataframe(rostered_minors_pitching,hide_index=True)
        
        minorpitchaxes_select()

    with tab4:
        st.header('Missing Player IDs')
        if not ss.rosters_df.empty and not ss.idkey.empty:
            missing_fg_ids = ss.rosters_df[~ss.rosters_df['id'].isin(ss.idkey['FANTRAXID'])]
            # Get player names from Fantrax API for missing IDs
            try:
                # Create a request to get player IDs from Fantrax
                response = requests.get('https://www.fantrax.com/fxea/general/getPlayerIds?sport=MLB')
                if response.status_code == 200:
                    player_data = response.json()
                    
                    # Create mapping of Fantrax ID to player name
                    id_to_name = {id: data['name'] for id, data in player_data.items()}
                    
                    # Update Name column where ID matches
                    missing_fg_ids['Name'] = missing_fg_ids['id'].map(id_to_name)
                else:
                    st.error("Failed to fetch player data from Fantrax API")
            except Exception as e:
                st.error(f"Error fetching player data: {str(e)}")
            st.dataframe(missing_fg_ids, hide_index=True)
        else:
            st.write("No data available to compare IDs")

def normalize_json_response(response, key):
    if response and key in response:
        return pd.json_normalize(response[key])
    else:
        st.error(f"Key '{key}' not found in response.")
        return pd.DataFrame()

if 'idkey' not in ss:
    try:
        ss.idkey = pd.read_csv('C:/Users/Brett/Fantasy Baseball Practice/Apps/Fantrax_Leagues_Dashboard/Player ID Key.csv')
    except FileNotFoundError:
        st.error("Player ID Key file not found.")

[MLB_pitching_df,MLB_hitting_df,minors_pitching_df, minors_hitting_df,league_list_df] = app_startup()

if "league_name" not in ss:
    ss.league_name = None

# l = league_list_df['leagueId'][0]
# [rosters_df, activepitchers, activehitters] = rosters(l)

with st.sidebar:
    @st.fragment
    def leagueselectbox():
        # Determine league index
        if ss.league_name != 'None' and not ss.league_list_df[ss.league_list_df['leagueName'] == ss.league_name].empty:
            league_index = ss.league_list_df.index[ss.league_list_df['leagueName'] == ss.league_name].tolist()[0]
        else:
            league_index = 0

        league_name_select = st.selectbox(
            "League Name",
            ss.league_list_df['leagueName'] if not ss.league_list_df.empty else ["No leagues available"],
            index=league_index
        )

        ss.league_name = league_name_select
    
    leagueselectbox()

    if 'refresh_pressed' not in ss:
        ss.refresh_pressed = False

    st.button(
            "Refresh",
            on_click=lambda: (
                league_selected(ss.league_name)
            )
    )