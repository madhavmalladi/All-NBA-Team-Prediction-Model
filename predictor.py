import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df_train = pd.read_csv(r"D:\Downloads\nba_stats_final.csv")

# Removing unnecessary values
df_train = df_train.drop(['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GS', 'MIN', 'FGA', 'FTM', 'FG3A','FTA', 'FT_PCT', 'OREB', 'DREB', 'PF'], axis = 1)

# Standardizing Values
gp_mean = df_train['GP'].mean()
gp_std = df_train['GP'].std()
df_train['GP'] = (df_train['GP'] - gp_mean)/gp_std
fgm_mean = df_train['FGM'].mean()
fgm_std = df_train['FGM'].std()
df_train['FGM'] = (df_train['FGM'] - fgm_mean) / fgm_std
fgp_mean = df_train['FG_PCT'].mean()
fgp_std = df_train['FG_PCT'].std()
df_train['FG_PCT'] = (df_train['FG_PCT'] - fgp_mean)/fgp_std
fg3m_mean = df_train['FG3M'].mean()
fg3m_std = df_train['FG3M'].std()
df_train['FG3M'] = (df_train['FG3M'] - fg3m_mean)/fg3m_std
fg3p_mean = df_train['FG3_PCT'].mean()
fg3p_std = df_train['FG3_PCT'].std()
df_train['FG3_PCT'] = (df_train['FG3_PCT'] - fg3p_mean)/fg3p_std
reb_mean = df_train['REB'].mean()
reb_std = df_train['REB'].std()
df_train['REB'] = (df_train['REB'] - reb_mean)/reb_std
ast_mean = df_train['AST'].mean()
ast_std = df_train['AST'].std()
df_train['AST'] = (df_train['AST'] - ast_mean)/ast_std
stl_mean = df_train['STL'].mean()
stl_std = df_train['STL'].std()
df_train['STL'] = (df_train['STL'] - stl_mean)/stl_std
blk_mean = df_train['BLK'].mean()
blk_std = df_train['BLK'].std()
df_train['BLK'] = (df_train['BLK'] - blk_mean)/blk_std
tov_mean = df_train['TOV'].mean()
tov_std = df_train['TOV'].std()
df_train['TOV'] = (df_train['TOV'] - tov_mean)/tov_std
pts_mean = df_train['PTS'].mean()
pts_std = df_train['PTS'].std()
df_train['PTS'] = (df_train['PTS'] - pts_mean)/pts_std

# Remove name, Tm, variable, Pos
columns_to_remove = ['Unnamed: 0', 'name', 'TEAM_ID', 'Tm', 'variable', 'Pos']
# Drop the specified columns
df_train.drop(columns_to_remove, axis=1, inplace=True)
#one hot encode the position column
# Convert 'position' column to strings
df_train['position'] = df_train['position'].astype(str)
df_train = pd.get_dummies(df_train, columns=['position'], prefix='position', dtype = 'int')

# Creating train and test sets
X_train = df_train.drop('AllStar', axis=1).to_numpy()
Y_train = df_train['AllStar'].astype('float32').to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state = 42)

def build_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  
    ])
    return model

input_shape = X_train[0].shape
model = build_model(input_shape)

# Compile the model
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Extracting training and validation accuracy from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotting training and validation accuracy
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'ro', label='Training accuracy')
plt.plot(epochs, val_acc, 'bo', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
