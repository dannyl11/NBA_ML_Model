import requests
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

teamDict = {'GSW': 'Warriors', 'CHI': 'Bulls', 'CLE': 'Cavaliers', 
            'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets', 
            'DAL': 'Mavericks', 'DEN': 'Nuggets', 'DET': 'Pistons', 
            'HOU': 'Rockets', 'IND': 'Pacers', 'LAC': 'Clippers', 
            'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat', 'MIL': 'Bucks', 
            'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
            'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
            'POR': 'Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 
            'TOR': 'Raptors','UTA': 'Jazz', 'WAS': 'Wizards'}

def getTeam():
    while True:
        team = str(input('Enter betting team abbreviation: '))
        opponent = str(input('Enter opposing team abbreviation: '))
        venue = str(input('Enter betting team home or away: '))
        if isValid(team) and isValid(opponent) and (venue.lower() == 'home' or 
                                                    venue.lower() == 'away'):
            return team, opponent, venue
        else:
            print('Error, try again')

def getOdds():
    while True:
        odds = str(input('Enter moneyline odds: '))
        if isValidOdds(odds):
            return odds
        else:
            print('Error, try again')

def isValidOdds(odds):
    if '+' not in odds and '-' not in odds:
        return False
    elif '+' in odds and '-' in odds:
        return False
    for c in odds[1:]:
        if c.isdigit() == False:
            return False
    return True

def isValid(team):
    if team in teamDict.keys():
        return True
    for abb in teamDict.keys():
        if team.lower() == abb.lower():
            return True
    return False

def getTeamID(team): #helper for getGameLog
    allTeams = teams.get_teams()
    teamX = [t for t in allTeams if t['abbreviation'] == team.upper()][0]
    teamID = teamX['id']
    return teamID

def getOpponent(str): #helper for getGameLog
    if 'vs.' in str:
        oppIndex = str.find('.')
        opponent = str[oppIndex+2:]
        return opponent, 1
    elif '@' in str:
        oppIndex = str.find('@')
        opponent = str[oppIndex+2:]
        return opponent, 0
                
def getGameLog(team): #last 200 games
    teamID = getTeamID(team)
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=str(teamID),
                                        season_type_nullable='Regular Season')
    games = gamefinder.get_data_frames()[0]
    gameLog = pd.DataFrame()
    opponents = []
    winLoss = []
    HomeAway = []
    for index, row in games.head(200).iterrows():
        matchup = games.loc[index, 'MATCHUP']
        opponent, homeAway = getOpponent(matchup)
        try:
            mascot = teamDict[opponent]
        except:
            continue
        opponents.append(mascot)
        HomeAway.append(homeAway)
        outcome = games.loc[index, 'WL']
        if outcome == 'W':
            winLoss.append(1)
        else:
            winLoss.append(0)
    gameLog['Opponent'] = opponents[::-1]
    gameLog['H1/A0'] = HomeAway[::-1]
    gameLog['W/L'] = winLoss[::-1]
    return gameLog

def addStats():
    team, opponent, venue = getTeam()
    opponent = teamDict[opponent.upper()]
    if venue.lower() == 'home':
        venue = 1
    else:
        venue = 0
    gameLog = getGameLog(team)
    url = 'https://www.basketball-reference.com/leagues/NBA_2025.html'
    response = requests.get(url)
    teamStats = pd.read_html(StringIO(response.text), attrs={'id':'advanced-team'}, 
                         header=1)[0]
    teamStats = teamStats.drop(['Rk', 'Age', 'W', 'L', 'PW', 'PL', 'MOV', 'SOS',
    'ORtg', 'DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'Unnamed: 17', 'FT/FGA', 
    'Unnamed: 22', 'eFG%.1', 'TOV%.1', 'FT/FGA.1', 'Unnamed: 27', 'Arena', 
    'Attend.', 'Attend./G'], axis=1)
    teamStats = teamStats.drop(30, axis=0)
    for team in teamStats['Team']:
        fullName = team.split()
        mascot = fullName[-1]
        teamStats.loc[teamStats['Team'] == team, 'Team'] = mascot
    NRtg = []
    srs = []
    eFG = []
    offRebPct = []
    turnoverPct = []
    defRebPct = []
    for opposition in gameLog['Opponent']:
        temp = teamStats.loc[teamStats['Team'] == opposition]
        row = temp.iloc[0].to_dict()
        NRtg.append(row['NRtg'])
        srs.append(row['SRS'])
        eFG.append(row['eFG%'])
        offRebPct.append(row['ORB%'])
        turnoverPct.append(row['TOV%'])
        defRebPct.append(row['DRB%'])
    gameLog['NetRtg'] = NRtg
    gameLog['SRS'] = srs
    gameLog['eFG%'] = eFG
    gameLog['OREB%'] = offRebPct
    gameLog['TOV%'] = turnoverPct
    gameLog['DREB%'] = defRebPct
    #now add upcoming game against opponent to gameLog df
    oppTemp = teamStats.loc[teamStats['Team'] == opponent]
    oppRow = oppTemp.iloc[0].to_dict()
    oppNRtg = oppRow['NRtg']
    oppSRS = oppRow['SRS']
    oppEFG = oppRow['eFG%']
    oppORB = oppRow['ORB%']
    oppTOV = oppRow['TOV%']
    oppDRB = oppRow['DRB%']
    lastRow = len(gameLog)
    gameLog.loc[lastRow] = [opponent, venue, 0, oppNRtg, oppSRS, oppEFG, oppORB,
                       oppTOV, oppDRB]
    return gameLog

def predictOutcome(data):
    vegasOdds = getOdds()
    probability = getProbability(vegasOdds)
    featureCols = ['H1/A0', 'NetRtg', 'SRS', 'eFG%', 'OREB%', 'TOV%', 'DREB%']
    X = data[featureCols] #features
    y = data['W/L'] #target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                            shuffle=False)
    logReg = LogisticRegression(C=0.5, solver='saga', max_iter=10**5)
    logReg.fit(X_train, y_train)
    yPred = logReg.predict_proba(X_test)
    winProb = yPred[-1, 1]    
    return winProb, probability

def getProbability(odds):
    if '+' in odds:
        impliedProbability = 100 / (int(odds) + 100)
        return round(impliedProbability, 3)
    elif '-' in odds:
        impliedProbability = abs(int(odds)) / (abs(int(odds)) + 100)
        return round(impliedProbability, 3)

def main():
    data = addStats()
    pred, vegas = predictOutcome(data)
    print(f'Calculated win probability: {pred}\tBook probability: {vegas}')
    print(f'Value over book: {pred-vegas}')
main()
