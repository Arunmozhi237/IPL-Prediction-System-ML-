import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

deliveries=pd.read_csv("deliveries.csv")
matches=pd.read_csv("matches.csv")
pd.set_option("display.max_columns",None)


total_runs=deliveries.groupby(["match_id","inning"]).sum()["total_runs"].reset_index()
total_runs=total_runs[total_runs["inning"]==1]

matches_df=matches.merge(total_runs[["match_id","total_runs"]],left_on="id",right_on="match_id")
team=['Kolkata Knight Riders','Royal Challengers Bengaluru','Chennai Super Kings',
      'Punjab Kings' ,'Rajasthan Royals', 'Delhi Capitals',
 'Mumbai Indians' ,'Sunrisers Hyderabad' ,
 'Lucknow Super Giants' ,'Gujarat Titans']

matches_df['team1']=matches_df['team1'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
matches_df['team2']=matches_df['team2'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
matches_df['winner']=matches_df['winner'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
matches_df['team1']=matches_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
matches_df['team2']=matches_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
matches_df['winner']=matches_df['winner'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
matches_df['team1']=matches_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
matches_df['team2']=matches_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
matches_df['winner']=matches_df['winner'].str.replace('Delhi Daredevils','Delhi Capitals')
matches_df['team1']=matches_df['team1'].str.replace('Kings XI Punjab','Punjab Kings')
matches_df["team2"]=matches_df['team2'].str.replace('Kings XI Punjab','Punjab Kings')
matches_df["winner"]=matches_df['winner'].str.replace('Kings XI Punjab','Punjab Kings')

team_win_count=matches_df["winner"].value_counts().head(10)
"""
sns.barplot(x=team_win_count,y=team_win_count.index)
plt.show()"""

matches_df=matches_df[["match_id","city","winner","total_runs"]].reset_index()
matches_df=matches_df.merge(deliveries,on="match_id")


matches_df=matches_df[matches_df["inning"]==2]

matches_df=matches_df[["match_id","city","winner","total_runs_x","inning","batting_team","bowling_team","over","ball","total_runs_y","player_dismissed"]]
matches_df=matches_df.rename(columns={"total_runs_x":"total_runs","total_runs_y":"run_on_this"})

matches_df=matches_df[matches_df['batting_team'].isin(team)]
matches_df=matches_df[matches_df['bowling_team'].isin(team)]
matches_df=matches_df[matches_df['winner'].isin(team)]
matches_df=matches_df[matches_df["ball"]<7]


matches_df["current_score"]=matches_df.groupby("match_id")["run_on_this"].cumsum()

matches_df["runs_left"]=matches_df["total_runs"]-matches_df["current_score"]
matches_df["ball_left"]=120-(matches_df["over"]*6+matches_df["ball"])

matches_df["wicket"]=matches_df["player_dismissed"].fillna("0")
matches_df["wicket"] = matches_df["player_dismissed"].notna().astype(int)
del matches_df["player_dismissed"]


wickets=matches_df.groupby("match_id")["wicket"].cumsum().values
matches_df["wicket_left"]=10-wickets

matches_df["required_r_r"]=(matches_df["runs_left"]/matches_df["ball_left"])*6
matches_df["current_r_r"]=(matches_df["current_score"]/matches_df["ball"])*6

def result(row):
    return 1 if row["batting_team"]==row["winner"] else 0

matches_df["result"]=matches_df.apply(result,axis="columns")



final_df=matches_df[["match_id","total_runs","batting_team","bowling_team","current_score","runs_left","ball_left","wicket_left","required_r_r","current_r_r","result"]]
final_df=final_df[final_df["ball_left"]!=0]
final_df.dropna(axis="columns",inplace=True)
final_df=final_df.reset_index()


batting_team=OneHotEncoder(sparse_output=False)
bowling_team=OneHotEncoder(sparse_output=False)

encoded1=batting_team.fit_transform(final_df[['batting_team']])
encoded1_df=pd.DataFrame(encoded1,columns=batting_team.get_feature_names_out(),index=final_df.index)

encoded2=bowling_team.fit_transform(final_df[['bowling_team']])
encoded2_df=pd.DataFrame(encoded2,columns=bowling_team.get_feature_names_out(),index=final_df.index)

final_df=pd.concat([final_df,encoded1_df.astype(int),encoded2_df.astype(int)],axis=1)

unique_match_id=final_df["match_id"].unique()
train,test=train_test_split(unique_match_id,test_size=0.2,random_state=42)

train=final_df[final_df["match_id"].isin(train)]
test=final_df[final_df["match_id"].isin(test)]

x_train=train.drop(columns=['index', 'match_id',"result",'batting_team', 'bowling_team'])
y_train=train["result"]

x_test=test.drop(columns=['index', 'match_id',"result",'batting_team', 'bowling_team'])
y_test=test["result"]

model=XGBClassifier(n_estimators=200,max_depth=10)
"""
params = {
    "n_estimators": [100,200,250],
    "max_depth":[10,20,30]}

tuning=GridSearchCV(estimator=model,param_grid=params,cv=5,scoring="accuracy")
tuning.fit(x_train,y_train)

print(tuning.best_params_,tuning.best_score_)  #{'max_depth': 10, 'n_estimators': 200} 0.7607616185415953
"""

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(accuracy_score(y_test,y_pred))

importance = pd.Series(
    model.feature_importances_,
    index=x_train.columns)

print(importance.sort_values(ascending=False))