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

    MLB_hitting_request = safe_request(url="https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=bat&lg=all&qual=0&season=2025&season1=2025&startdate=2025-03-01&enddate=2025-11-01&month=0&hand=&team=0&pageitems=30&pagenum=2&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR")
    MLB_hitting_response = MLB_hitting_request
    MLB_hitting_df = pd.json_normalize(MLB_hitting_response['data'])
    
    # Remove unnecessary columns but keep playerid for mapping
    MLB_hitting_df.drop(['Name','Team','TeamName','PlayerNameRoute','position','teamid'],axis=1,inplace=True)
    
    # Only filter by PA > 0, don't filter by ID yet
    MLB_hitting_df = MLB_hitting_df[MLB_hitting_df['PA'] > 0].reset_index(drop=True)
    
    # Convert playerid to string to ensure consistent data type
    MLB_hitting_df['playerid'] = MLB_hitting_df['playerid'].astype(str)
    
    # Ensure idkey has string FanGraphs IDs
    ss.idkey['IDFANGRAPHS'] = ss.idkey['IDFANGRAPHS'].astype(str)
    
    # Create a mapping dictionary for better error handling - with string keys
    id_mapping = ss.idkey.set_index('IDFANGRAPHS')['FANTRAXID'].to_dict()
    
    # Map Fantrax_ID more safely, handling missing IDs
    MLB_hitting_df['Fantrax_ID'] = MLB_hitting_df['playerid'].apply(
        lambda x: id_mapping.get(x, None)
    )
    
    # Calculate derived fields
    MLB_hitting_df['Swings'] = round(MLB_hitting_df['Pitches']*MLB_hitting_df['Swing%'])

    # Organize columns
    nonleadingcols = [col for col in MLB_hitting_df.columns if col not in leadingcols]
    MLB_hitting_df = MLB_hitting_df[leadingcols + nonleadingcols]
    MLB_hitting_df = MLB_hitting_df.rename(columns={'PlayerName':'Player','TeamNameAbb':'MLBTeam'})

    # Repeat for pitching
    MLB_pitching_request = safe_request(url="https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&qual=0&season=2025&season1=2025&startdate=2025-03-01&enddate=2025-11-01&month=0&hand=&team=0&pageitems=30&pagenum=2&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR")
    MLB_pitching_response = MLB_pitching_request
    MLB_pitching_df = pd.json_normalize(MLB_pitching_response['data'])
    MLB_pitching_df.drop(['Name','Team','TeamName','PlayerNameRoute','position','teamid'],axis=1,inplace=True)
    
    # Only filter by IP > 0, don't filter by ID yet
    MLB_pitching_df = MLB_pitching_df[MLB_pitching_df['IP'] > 0].reset_index(drop=True)
    
    # Convert playerid to string to ensure consistent data type
    MLB_pitching_df['playerid'] = MLB_pitching_df['playerid'].astype(str)
    
    # Map Fantrax_ID more safely for pitchers
    MLB_pitching_df['Fantrax_ID'] = MLB_pitching_df['playerid'].apply(
        lambda x: id_mapping.get(x, None)
    )
    
    # Calculate derived fields
    MLB_pitching_df['Swings'] = round(MLB_pitching_df['Pitches']*MLB_pitching_df['Swing%'])

    # Organize columns
    nonleadingcols = [col for col in MLB_pitching_df.columns if col not in leadingcols]
    MLB_pitching_df = MLB_pitching_df[leadingcols + nonleadingcols]
    MLB_pitching_df = MLB_pitching_df.rename(columns={'PlayerName':'Player','TeamNameAbb':'MLBTeam','sp_stuff':'Stuff+',
                                                      'sp_location':'Location+','sp_pitching':'Pitching+'})

    return [MLB_pitching_df, MLB_hitting_df]

def minors_stats_import():
    leadingcols = ['PlayerName','AffAbbName','aLevel','Age','playerids','Fantrax_ID']

    minors_hitting_request = safe_request(url="https://www.fangraphs.com/api/leaders/minor-league/data?pos=all&level=0&lg=2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32&stats=bba&qual=0&type=0&team=&season=2025&seasonEnd=2025&org=&ind=0&splitTeam=false")
    minors_hitting_response = minors_hitting_request
    minors_hitting_df = pd.DataFrame(minors_hitting_response)
    minors_hitting_df.drop(['Name','Team','TeamName','AffId'],axis=1,inplace=True)
    
    # Convert playerids to string
    minors_hitting_df['playerids'] = minors_hitting_df['playerids'].astype(str)
    
    minors_hitting_df = (
        minors_hitting_df[minors_hitting_df['playerids'].isin(ss.idkey['IDFANGRAPHS'].astype(str)) & (minors_hitting_df['PA'] > 0)]
        .reset_index(drop=True)
    )
    
    # Create a mapping dictionary with string keys
    id_mapping = ss.idkey.set_index('IDFANGRAPHS')['FANTRAXID'].to_dict()
    
    # Map using string comparison
    minors_hitting_df['Fantrax_ID'] = minors_hitting_df['playerids'].apply(
        lambda x: id_mapping.get(x, None)
    )

    nonleadingcols = [col for col in minors_hitting_df.columns if col not in leadingcols]
    minors_hitting_df = minors_hitting_df[leadingcols + nonleadingcols]
    minors_hitting_df = minors_hitting_df.rename(columns={'PlayerName':'Player','AffAbbName':'Org','aLevel':'Level'})

    minors_pitching_request = safe_request(url="https://www.fangraphs.com/api/leaders/minor-league/data?pos=all&level=0&lg=2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32&stats=pit&qual=0&type=0&team=&season=2025&seasonEnd=2025&org=&ind=0&splitTeam=false")
    minors_pitching_response = minors_pitching_request
    minors_pitching_df = pd.DataFrame(minors_pitching_response)
    minors_pitching_df.drop(['Name','Team','TeamName','AffId'],axis=1,inplace=True)
    
    # Convert playerids to string
    minors_pitching_df['playerids'] = minors_pitching_df['playerids'].astype(str)
    
    minors_pitching_df = (
        minors_pitching_df[minors_pitching_df['playerids'].isin(ss.idkey['IDFANGRAPHS'].astype(str)) & (minors_pitching_df['IP'] > 0)]
        .reset_index(drop=True)
    )
    
    # Map using string comparison
    minors_pitching_df['Fantrax_ID'] = minors_pitching_df['playerids'].apply(
        lambda x: id_mapping.get(x, None)
    )

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
    """
    Sum specified columns grouped by 'teamName'.
    
    Parameters:
    - df: DataFrame containing the data to sum
    - columns_to_sum: List of columns to sum
    - output_df_name: Name to use when storing result in globals_dict
    - globals_dict: Dictionary to store results
    
    Returns:
    - DataFrame with summed statistics
    """
    # Check if dataframe is empty or contains no relevant teams/columns
    if df.empty or 'teamName' not in df.columns:
        if output_df_name in globals_dict:
            return globals_dict[output_df_name]
        else:
            return pd.DataFrame(columns=['teamName'])
            
    # Only sum columns that exist in the dataframe
    valid_columns = [col for col in columns_to_sum if col in df.columns]
    
    if not valid_columns:
        # If no valid columns, return existing data or empty dataframe
        if output_df_name in globals_dict:
            return globals_dict[output_df_name]
        else:
            return pd.DataFrame(columns=['teamName'])
    
    # Sum specified columns grouped by 'teamName'
    summed_df = df.groupby('teamName')[valid_columns].sum().reset_index()
    
    # Check if output dataframe already exists
    if output_df_name in globals_dict:
        existing_df = globals_dict[output_df_name]
        
        if not existing_df.empty and 'teamName' in existing_df.columns:
            # Set teamName as index for easier updating
            existing_df_indexed = existing_df.set_index('teamName')
            summed_df_indexed = summed_df.set_index('teamName')
            
            # Update existing columns and add new ones
            for col in summed_df_indexed.columns:
                existing_df_indexed[col] = summed_df_indexed[col]
                
            # Reset index and update summed_df
            summed_df = existing_df_indexed.reset_index()
    
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

    # Check if dataframe is empty or contains no relevant teams
    if df.empty or 'teamName' not in df.columns or df['teamName'].isna().all():
        if output_df_name in globals_dict:
            return globals_dict[output_df_name]
        else:
            return pd.DataFrame(columns=['teamName'])

    # Initialize results dataframe
    rate_df = pd.DataFrame()
    
    # Handle columns differently based on their nature
    # For percentage stats (like AVG, OBP, etc.), we need to calculate the weighted average
    # For raw rate stats (like wRC+), we can take a simpler weighted average

    # Group by team
    grouped = df.groupby('teamName')
    
    # Get team names as a list to initialize the results DataFrame
    team_names = list(grouped.groups.keys())
    rate_df['teamName'] = team_names
    
    # Get denominator sums for each team
    denominator_sums = grouped[denominator_col].sum().reset_index()
    denominator_dict = dict(zip(denominator_sums['teamName'], denominator_sums[denominator_col]))
    
    # Process each rate column
    for col in rate_columns:
        if col not in df.columns:
            continue
            
        # For each team, calculate weighted average based on the denominator
        team_values = {}
        
        for team in team_names:
            team_data = df[df['teamName'] == team]
            
            if team_data.empty or denominator_dict[team] == 0:
                team_values[team] = float('nan')
                continue
                
            # For stats that are already rates (like AVG, OBP, wRC+, etc.)
            # Calculate weighted average: sum(stat * denominator) / sum(denominator)
            team_values[team] = (team_data[col] * team_data[denominator_col]).sum() / denominator_dict[team]
        
        # Add column to results
        rate_df[col] = [team_values[team] for team in team_names]
    
    # Merge with existing dataframe if it exists
    if output_df_name in globals_dict:
        existing_df = globals_dict[output_df_name]
        # If the existing df has columns we need to preserve, merge on teamName
        if len(existing_df.columns) > 1:  # More than just teamName column
            # Set teamName as index for both dataframes
            existing_df_indexed = existing_df.set_index('teamName')
            rate_df_indexed = rate_df.set_index('teamName')
            
            # Update existing dataframe with new values
            for col in rate_df_indexed.columns:
                if col in existing_df_indexed.columns:
                    existing_df_indexed[col] = rate_df_indexed[col]
                else:
                    existing_df_indexed[col] = rate_df_indexed[col]
            
            # Reset index and return
            result_df = existing_df_indexed.reset_index()
        else:
            result_df = rate_df
    else:
        result_df = rate_df
    
    # Store in globals dictionary
    globals_dict[output_df_name] = result_df
    
    return result_df

def league_selected(league_name):
    # Ensure rosters_df is available before proceeding
    if 'rosters_df' not in ss or ss.rosters_df.empty:
        league_id = ss.league_list_df.loc[ss.league_list_df['leagueName'] == league_name, 'leagueId'].iloc[0]
        ss.rosters_df, ss.activepitchers, ss.activehitters = rosters(league_id)

    league_id = ss.league_list_df.loc[ss.league_list_df['leagueName'] == league_name, 'leagueId'].iloc[0]

    [ss.rosters_df, ss.activepitchers, ss.activehitters] = rosters(league_id)
    
    # Ensure consistent data types for IDs in rosters
    if 'id' in ss.rosters_df.columns:
        ss.rosters_df['id'] = ss.rosters_df['id'].astype(str)
    if 'id' in ss.activepitchers.columns:
        ss.activepitchers['id'] = ss.activepitchers['id'].astype(str)
    if 'id' in ss.activehitters.columns:
        ss.activehitters['id'] = ss.activehitters['id'].astype(str)
    
    # Ensure consistent data types in stored MLB DataFrames
    if 'Fantrax_ID' in ss.MLB_hitting_df.columns:
        ss.MLB_hitting_df['Fantrax_ID'] = ss.MLB_hitting_df['Fantrax_ID'].astype(str)
    if 'Fantrax_ID' in ss.MLB_pitching_df.columns:
        ss.MLB_pitching_df['Fantrax_ID'] = ss.MLB_pitching_df['Fantrax_ID'].astype(str)
    
    # Check for NaN values in Fantrax_ID
    if ss.MLB_hitting_df['Fantrax_ID'].isna().any():
        missing_count = ss.MLB_hitting_df['Fantrax_ID'].isna().sum()
        total_count = len(ss.MLB_hitting_df)
        st.warning(f"Warning: {missing_count} out of {total_count} MLB hitters missing Fantrax ID ({missing_count/total_count:.1%})")
    if ss.MLB_pitching_df['Fantrax_ID'].isna().any():
        missing_count = ss.MLB_pitching_df['Fantrax_ID'].isna().sum()
        total_count = len(ss.MLB_pitching_df)
        st.warning(f"Warning: {missing_count} out of {total_count} MLB pitchers missing Fantrax ID ({missing_count/total_count:.1%})")
    
    # Merge with rostered hitters - handle case when there might be no matches
    activehitting = pd.DataFrame()
    if not ss.MLB_hitting_df.empty and not ss.activehitters.empty:
        # Convert IDs to string during merge to ensure type consistency
        activehitting = ss.MLB_hitting_df.merge(
            ss.activehitters[['id', 'teamName']], 
            left_on='Fantrax_ID', 
            right_on='id', 
            how='inner'
        )
        
        # If we got matches, drop the duplicate ID column
        if not activehitting.empty:
            activehitting.drop('id', axis=1, inplace=True)

    # Merge with rostered pitchers - handle case when there might be no matches
    activepitching = pd.DataFrame()
    if not ss.MLB_pitching_df.empty and not ss.activepitchers.empty:
        # Convert IDs to string during merge to ensure type consistency
        activepitching = ss.MLB_pitching_df.merge(
            ss.activepitchers[['id', 'teamName']], 
            left_on='Fantrax_ID', 
            right_on='id', 
            how='inner'
        )
        
        # If we got matches, drop the duplicate ID column
        if not activepitching.empty:
            activepitching.drop('id', axis=1, inplace=True)

    # Generate column lists based on dataframe contents
    # Using the 'or []' syntax to handle empty dataframes
    teamhitting_vars = activehitting.columns.tolist() if not activehitting.empty else ['Player', 'MLBTeam', 'teamName']
    teampitching_vars = activepitching.columns.tolist() if not activepitching.empty else ['Player', 'MLBTeam', 'teamName']

    # Lists of columns to exclude
    hitting_exclude_columns = ['playerid','playerids','Fantrax_ID','Bats','xMLBAMID','Season','AgeR','SeasonMin','SeasonMax','GDP','GB','FB','LD','IFFB',
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
                                                        'piO-Contact%','piZ-Contact%','piContact%','piZone%','piPace','Barrels','HardHit','Q','TG']
    pitching_exclude_columns = ['playerid','playerids','Fantrax_ID','Throws','xMLBAMID','Season','AgeR','SeasonMin','SeasonMax','IBB','BK','GB','FB','LD','IFFB',
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
                                                           'piPace','Barrels','HardHit','Q','TG']

    # Only filter columns if dataframes are not empty
    if not activehitting.empty:
        activehitting = activehitting.drop(columns=[col for col in activehitting.columns if col in hitting_exclude_columns], errors='ignore')
    if not activepitching.empty:
        activepitching = activepitching.drop(columns=[col for col in activepitching.columns if col in pitching_exclude_columns], errors='ignore')

    # Update vars after filtering
    teamhitting_vars = activehitting.columns.tolist() if not activehitting.empty else ['Player', 'MLBTeam', 'teamName']
    teampitching_vars = activepitching.columns.tolist() if not activepitching.empty else ['Player', 'MLBTeam', 'teamName']

    hitplotvars = [h for h in teamhitting_vars if h not in ('Player','MLBTeam','teamName')]
    pitchplotvars = [p for p in teampitching_vars if p not in ('Player','MLBTeam','teamName')]

    ss.globals_dict = {}

    # Only calculate stats if dataframes are not empty
    if not activehitting.empty and 'teamName' in activehitting.columns:
        try:
            # First calculate all sum statistics
            sum_columns = [col for col in ['PA', 'AB', 'H', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'SO', 'SB', 'CS', 'Pitches', 'Swings', 'Events'] 
                          if col in activehitting.columns]
            if sum_columns:
                sum_team_stats(activehitting, sum_columns, 'teamhitting', ss.globals_dict)
            
            # Calculate rate statistics with proper error handling
            if 'AB' in activehitting.columns:
                ab_rate_columns = [col for col in ['AVG', 'SLG', 'ISO', 'xAVG', 'xSLG'] 
                                 if col in activehitting.columns]
                if ab_rate_columns:
                    calculate_team_rate_statistics(activehitting, ab_rate_columns, 'AB', 'teamhitting', ss.globals_dict)
            
            if 'PA' in activehitting.columns:
                pa_rate_columns = [col for col in ['OBP', 'BB%', 'K%', 'wOBA', 'wRC+', 'xwOBA'] 
                                 if col in activehitting.columns]
                if pa_rate_columns:
                    calculate_team_rate_statistics(activehitting, pa_rate_columns, 'PA', 'teamhitting', ss.globals_dict)
            
            if 'Swings' in activehitting.columns and 'Swings' in sum_columns:
                swing_rate_columns = [col for col in ['Contact%'] 
                                   if col in activehitting.columns]
                if swing_rate_columns:
                    calculate_team_rate_statistics(activehitting, swing_rate_columns, 'Swings', 'teamhitting', ss.globals_dict)
            
            if 'Events' in activehitting.columns and 'Events' in sum_columns:
                event_rate_columns = [col for col in ['EV', 'Barrel%', 'HardHit%'] 
                                   if col in activehitting.columns]
                if event_rate_columns:
                    calculate_team_rate_statistics(activehitting, event_rate_columns, 'Events', 'teamhitting', ss.globals_dict)
            
            if 'Pitches' in activehitting.columns and 'Pitches' in sum_columns:
                pitch_rate_columns = [col for col in ['SwStr%', 'C+SwStr%'] 
                                   if col in activehitting.columns]
                if pitch_rate_columns:
                    calculate_team_rate_statistics(activehitting, pitch_rate_columns, 'Pitches', 'teamhitting', ss.globals_dict)
            
        except Exception as e:
            st.error(f"Error calculating hitting statistics: {str(e)}")
            # Initialize with empty dataframe if it doesn't exist
            if 'teamhitting' not in ss.globals_dict:
                ss.globals_dict['teamhitting'] = pd.DataFrame(columns=['teamName'])
    else:
        # Initialize with empty dataframe
        ss.globals_dict['teamhitting'] = pd.DataFrame(columns=['teamName'])

    if not activepitching.empty and 'teamName' in activepitching.columns:
        try:
            # First calculate all sum statistics
            sum_columns = [col for col in ['W', 'L', 'QS', 'CG', 'SV', 'BS', 'HLD', 'IP', 'TIP', 'TBF', 'H', 'ER', 'HR', 'BB', 'SO', 'Pitches', 'Swings', 'Events'] 
                          if col in activepitching.columns]
            if sum_columns:
                sum_team_stats(activepitching, sum_columns, 'teampitching', ss.globals_dict)
            
            # Calculate rate statistics with proper error handling
            if 'TIP' in activepitching.columns and 'TIP' in sum_columns:
                tip_rate_columns = [col for col in ['ERA', 'WHIP', 'FIP', 'xERA', 'xFIP', 'SIERA'] 
                                 if col in activepitching.columns]
                if tip_rate_columns:
                    calculate_team_rate_statistics(activepitching, tip_rate_columns, 'TIP', 'teampitching', ss.globals_dict)
            
            if 'Events' in activepitching.columns and 'Events' in sum_columns:
                event_rate_columns = [col for col in ['GB%', 'FB%', 'EV', 'Barrel%', 'HardHit%'] 
                                   if col in activepitching.columns]
                if event_rate_columns:
                    calculate_team_rate_statistics(activepitching, event_rate_columns, 'Events', 'teampitching', ss.globals_dict)
            
            if 'Pitches' in activepitching.columns and 'Pitches' in sum_columns:
                pitch_rate_columns = [col for col in ['Zone%', 'SwStr%', 'C+SwStr%', 'Stuff+', 'Location+', 'Pitching+'] 
                                   if col in activepitching.columns]
                if pitch_rate_columns:
                    calculate_team_rate_statistics(activepitching, pitch_rate_columns, 'Pitches', 'teampitching', ss.globals_dict)
            
            if 'Swings' in activepitching.columns and 'Swings' in sum_columns:
                swing_rate_columns = [col for col in ['Contact%'] 
                                   if col in activepitching.columns]
                if swing_rate_columns:
                    calculate_team_rate_statistics(activepitching, swing_rate_columns, 'Swings', 'teampitching', ss.globals_dict)
            
            if 'TBF' in activepitching.columns and 'TBF' in sum_columns:
                tbf_rate_columns = [col for col in ['K%', 'BB%'] 
                                 if col in activepitching.columns]
                if tbf_rate_columns:
                    calculate_team_rate_statistics(activepitching, tbf_rate_columns, 'TBF', 'teampitching', ss.globals_dict)
            
        except Exception as e:
            st.error(f"Error calculating pitching statistics: {str(e)}")
            # Initialize with empty dataframe if it doesn't exist
            if 'teampitching' not in ss.globals_dict:
                ss.globals_dict['teampitching'] = pd.DataFrame(columns=['teamName'])
    else:
        # Initialize with empty dataframe
        ss.globals_dict['teampitching'] = pd.DataFrame(columns=['teamName'])

    # Ensure both dataframes exist in globals_dict
    if 'teamhitting' not in ss.globals_dict:
        ss.globals_dict['teamhitting'] = pd.DataFrame(columns=['teamName'])
    if 'teampitching' not in ss.globals_dict:
        ss.globals_dict['teampitching'] = pd.DataFrame(columns=['teamName'])

    # Only reorder columns if dataframes are not empty
    if not activehitting.empty and 'teamName' in activehitting.columns:
        teamnamecolumn = activehitting.pop('teamName')
        activehitting.insert(1, teamnamecolumn.name, teamnamecolumn)
    if not activepitching.empty and 'teamName' in activepitching.columns:
        teamnamecolumn = activepitching.pop('teamName')
        activepitching.insert(1, teamnamecolumn.name, teamnamecolumn)

    hitplotvars = [e for e in hitplotvars if e not in ('ID','Pos','Player','MLBTeam','Eligible','Status','Opponent','FANTRAXID','IDFANGRAPHS','MLBID','team_name')]
    pitchplotvars = [g for g in pitchplotvars if g not in ('ID','Pos','Player','MLBTeam','Eligible','Status','Opponent','FANTRAXID','IDFANGRAPHS','MLBID','team_name')]

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

    tab1, tab2, tab3, tab4 = st.tabs(['League Data','MLB Stats','MiLB Stats','Troubleshoot'])

    with tab1:
        col1,col2=st.columns(2)

        with col1:
            @st.fragment
            def hitaxes_select():
                hitting_columns = [col for col in ss.globals_dict['teamhitting'].columns if col not in ['Name', 'teamName']]
                col1_1,col1_2=st.columns(2)
                with col1_1:
                    ss.teamhit_xaxis = st.selectbox('X Axis',hitting_columns,index=5)
                with col1_2:
                    ss.teamhit_yaxis = st.selectbox('Y Axis',hitting_columns,index=10)
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

            try:
                if not activehitting.empty:
                    majorhitfig = px.scatter(activehitting,x=majorhit_xaxis,y=majorhit_yaxis,color='teamName',
                                        color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
                    st.plotly_chart(majorhitfig)
                    st.dataframe(activehitting,hide_index=True)
                else:
                    st.info("No MLB hitting data available to plot.")
            except Exception as e:
                st.error("Error creating MLB hitting plot")
        
        majorhitaxes_select()

        st.header('Major League Pitching')
        @st.fragment
        def majorpitchaxes_select():
            col3,col4 = st.columns(2)
            with col3:
                majorpitch_xaxis = st.selectbox('X Axis',pitchplotvars,index=20)
            with col4:
                majorpitch_yaxis = st.selectbox('Y Axis',pitchplotvars,index=3)

            try:
                if not activepitching.empty:
                    majorpitchfig = px.scatter(activepitching,x=majorpitch_xaxis,y=majorpitch_yaxis,color='teamName',
                                        color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
                    st.plotly_chart(majorpitchfig)
                    st.dataframe(activepitching,hide_index=True)
                else:
                    st.info("No MLB pitching data available to plot.")
            except Exception as e:
                st.error("Error creating MLB pitching plot")
        
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

            try:
                if not rostered_minors_hitting.empty:
                    minorhitfig = px.scatter(rostered_minors_hitting,x=minorhit_xaxis,y=minorhit_yaxis,color='teamName',
                                        color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
                    st.plotly_chart(minorhitfig)
                    st.dataframe(rostered_minors_hitting,hide_index=True)
                else:
                    st.info("No minor league hitting data available to plot.")
            except Exception as e:
                st.error("Error creating minor league hitting plot")
        
        minorhitaxes_select()

        st.header('Minor League Pitching')
        @st.fragment
        def minorpitchaxes_select():
            col3,col4 = st.columns(2)
            with col3:
                minorpitch_xaxis = st.selectbox('X Axis',pitchplotvars,index=20,key='minor_pitch_x')
            with col4:
                minorpitch_yaxis = st.selectbox('Y Axis',pitchplotvars,index=3,key='minor_pitch_y')

            try:
                if not rostered_minors_pitching.empty:
                    minorpitchfig = px.scatter(rostered_minors_pitching,x=minorpitch_xaxis,y=minorpitch_yaxis,color='teamName',
                                        color_discrete_sequence=px.colors.qualitative.Light24,hover_name='Player')
                    st.plotly_chart(minorpitchfig)
                    st.dataframe(rostered_minors_pitching,hide_index=True)
                else:
                    st.info("No minor league pitching data available to plot.")
            except Exception as e:
                st.error("Error creating minor league pitching plot")
        
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
        ss.idkey = pd.read_csv('https://raw.githubusercontent.com/bamc021/Fantrax_Leagues_Dashboard/refs/heads/main/Player%20ID%20Key.csv')
        # Convert IDFANGRAPHS to string immediately after loading
        ss.idkey['IDFANGRAPHS'] = ss.idkey['IDFANGRAPHS'].astype(str)
    except Exception as e:
        st.error(f"Error loading Player ID Key file: {str(e)}")
        ss.idkey = pd.DataFrame()  # Initialize empty DataFrame to prevent further errors

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
