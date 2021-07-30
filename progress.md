
First try: Single random forest model, 69% accuracy and f1, 71% precision, 78% roc auc
    Features: map, scoreCT, scoreT, round_status, round_status_time_left, healthCT, healthT, equipmentCT, equipmentT,



## Data Cleaning

# Data Errors

- In 2 datapoints there were 6 players in one of the teams, which should be, in theory, impossible. Only theory I have is in the picture/reddit post
- In half a dozen screenshots, games were started with players having a lot more money than possible by the typical game rules. I now believe that this was a simple mistake on the data, given that in folowing screenshots (presumable from the same match), the money ammounts all go back to normal. At first I believed that I would have to remove the whole match from the dataset, as the extra money at the beginning would bend the way the game is played, creating different strategies and experiences, and so it should not be allowed to influence the models.
- Detected one datapoint where Terrorists had 8 smokes when only 5 were allowed. Started to filter datapoints where a team had more grenades than allowed.

## Feature encoding

Some features are categorical: round_winner (bool), map (8 maps), round_status (3 maybe 4 (SlackTime not found)), bomb_site (bool+NaN), etc...
These features had to be converted from the strings they were into another readable format.

- First approach: using column .cat.codes (e.g. df2["round_status_codes"] = df2["round_status"].cat.codes)
- Second approach: using one hot encoding, and being careful of the dummy variable trap (cutting one unnecessary column)
- possible third? https://www.youtube.com/watch?v=OTPz5plKb40 - One Hot Encoding with Multiple Categories (apply one hot encoding to just the most used categories, and the rarer ones get something else) or Mean Encoding

- is round_status nominal (like the map), or ordinal(a ordem das fases importa) ?

- After the One Hot Enconding was performed (19/07), when I was testing the results with Variance Inflation Factor, I noticed that the round_status and the bomb_site columns had a infinite VIF (divide by 0). This meant that they shared perfect multicollinearity. The thing is: the bomb sites could only be A, B or None (bomb not planted yet), and the latter was dropped as it can be expressed by a lack of bomb in A or B. The round status on the other hand, could be FreezeTime, Normal or BombPlanted, with the latter being dropped before.
The thing is: BombPlanted would always be true when A or B were present, so it had to be eliminated from consideration, which meant that for either FreezeTime or Normal, one column could be eliminated as one feature could represent both values. In the end, only the FreezeTime category, along with both bomb_A and bomb_B features survived. These 3 features can represent all possible combinations of round_status and bomb_site while sharing very little in common.

One issue that I'm finding is that the teams' total health and number of alive players have very high VIF (14 to 24), which makes sense, since most of the time the players will be at full health, making one almost redundant. However, there might be cases where the team is damaged. For example, 2 players with half health play very differently to one player with full health. While this might be a rare occurance in the dataset due to the time gaps in information and lack of events, it might, in a more real scenario, be an essential feature to have. The data will always be highly correlated, but it might be important especially when it tells a different story from normal.