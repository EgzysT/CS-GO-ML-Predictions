
First try: Single random forest model, 69% accuracy and f1, 71% precision, 78% roc auc
    Features: map, scoreCT, scoreT, round_status, round_status_time_left, healthCT, healthT, equipmentCT, equipmentT,


## Data Cleaning

### Data Errors

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

 - One issue that I'm finding is that the teams' total health and number of alive players have very high VIF (14 to 24), which makes sense, since most of the time the players will be at full health, making one almost redundant. However, there might be cases where the team is damaged. For example, 2 players with half health play very differently to one player with full health. While this might be a rare occurance in the dataset due to the time gaps in information and lack of events, it might, in a more real scenario, be an essential feature to have. The data will always be highly correlated, but it might be important especially when it tells a different story from normal.

 - I noticed that the maps had very high multicollinearity values, even while I dropped one of the columns/maps. When i looked more into it, I noticed that one of the possible maps - de_cache - was severely lacking in datapoints, consisting of only 145 (or 0.12%) of datapoints, where all the other maps had at least eleven thousand (or 9%). This means that even though the de_cache map column in one hot encoding was dropped, the others still had an extremely high chance of predicting themselves using the available maps. Also, the extremely small ammount of datapoints for this specific map has lead me to believe that maybe it would be better to not consider those datapoints, as there isn't a substantial enough ammount of data to help.


## Data Leakage
Data Leakage could occur when statistics from the test data are being used to fit the model or part of it. For example, if the scaler uses the data in the test set.
Pipelines can take care of this issue in a safer way, even when using cross-validation.


## Feature Selection
https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
https://machinelearningmastery.com/an-introduction-to-feature-selection/
https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/

Feature selection methods are intended to reduce the number of input variables, by selecting the ones believed to be the most useful to a model, in order to predict the target variable.
Besides the speed up gained by having to consider as many features when developing and training models, feature selection may improve the accuracy of the model by making it more resilient to noise in certain features, as well as removing non-informative variables that could add uncertainty to the predictions.
From what I've read, there is two main types of feature selection methods: Filter and Wrapper.
There are however, algorithms that intrinsically perform feature selection, such as decision trees (and random-forest) or rule-based models.

### Filter methods
Filter methods use statistical data about the different features to choose the most relevant features. Methods like SelectKBest (top k variables) or SelectPercentile (top percentile variables) choose the most well scored features using some metrics, such as ANOVA F-value (f_classif) or information gain (mutual_info_classif)

### Wrapper methods
Wrapper methods consider the selection of a set of features as a search problem, where multiple combinations of features are evaluated and compared against eachother. A predictive model is used to make this evaluation and assign scores based on the resulting accuracy of the model. 
A popular example of a wrapper method is the recursive feature elimination algorithm (RFE in scikit-learn).



## Reasons to choose this dataset
The dataset does not identify individual players, just shows screenshots of a round and who is alive, their inventory, and who was killed with what weapon.
This comes with both advantages and disadvantages. 
- As this work in intended to showcase a process of analyzing competitive games for predicting results and also detect possible inbalances of a "generic" competitive game, this allows for the five players in each side to be almost "reduced" to one team vs the other, something that could be considered closer to, for example, 1v1 games. This can therefore be used as a decent "ground-level" that can be expanded upon in the future.
- By not including information about particular players, or even not having any sort of guaranteed connection between screenshots, we lose a lot of potential information that would allow for a better prediction of outcomes, such as player "skill levels" - a factor considered to be crucial for who wins the game.

## Possible problems with the model (Future Work)
The model only tries to predict who will win the given round. This has some advantages that were "previously mentioned", but it brings with it some disadvantages.
We have to be careful analysing the data for statistics and the like, since the full objective of the players is to win the match, not necessarily just the round. Some strategies might sacrifice a round's win chance but increase the chances of winning following rounds. A "Eco" round, as it is known in the CS:GO community, calls for the team to purchase the minimum equipment, sacrificing the chances of winning this round, in order to save funds that can be used to increase the chances of winning the folowing rounds. This strategy might be perfect for winning the game, but not for winning the round - which is our target variable.
A future improvement on this work could include more general objectives as target variables if it is going to use CS:GO as an example, especially using who won the match, besides the already used who won the round.


The dummy variable trap is dangerous and essential to deal with it when we have a regression problem. In a classification problem however, we might introduce some bias by removing the extra data, so we need to be aware of this.