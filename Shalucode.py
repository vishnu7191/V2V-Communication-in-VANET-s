"Import Libaries Function "
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.linalg as npl
from sklearn.preprocessing import MinMaxScaler
import Optimization
from Optimization import testFunctions as tf
from Optimization import animation, animation3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#1.Data Selection
print("==================================================")
print("Data Selection")
print(" Blockchain-Enhanced Vehicular Ad-hoc Networks")
print("==================================================")
df = pd.read_csv('VanetDataset.csv')
df = df.sample(frac=1)
print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(df.head(3))
print("----------------------------------------------")
print()
print("Dataframe Shape ",df.shape)

#plt.matshow(df.corr())
#plt.title('Correlation Matrix', fontsize=11)
#plt.show()

#------------------------------------------------------------------------------
#2.pre processing--------------------------------------------------
#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(df.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
df=df.fillna(0)
print(df.isnull().sum())
print()
print("-----------------------------------------------")

#----------------------------------------------------------------------------------
"Label Encoding Process" 
for x in df.columns:
    print(x,' ',df[x].dtype) 

#print(df['DSN'].describe())   
print(pd.unique(df['protocol'].values)) 
df.replace({'AODV':0, 'ICMP':1, 'UDP':2 }, inplace = True)
pd.unique(df['Labels'].values)
df.replace({'normal':0, 'attack':1 }, inplace = True)
pd.unique(df['MsgType'].values)
df.replace({'Route Reply':0, '-1':1, 'Route Error':2, 'Route Request':3, 'Route Reply Acknowledgment':4}, inplace = True)
# df = pd.get_dummies(df, columns=['DSN'])
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()


df = df.astype(str).apply(label_encoder.fit_transform)

print("-------------------------------------------")
print(" After label Encoding ")
print("------------------------------------------")
print()

print(df['Labels'].head(20))

#---------------------------------------------------------------------------------------------
X=df.drop(['Labels'], axis=1)
y=df["Labels"]
#---------------------------------------------------------------------------------------------
print("---------------------------------------------")
print("Request to Send (RTS) and Message Receiving")
print("Feature Selection")
print()
print("Particle Bee Colony Swarm Algorithm ")
print()
x=X
dim = 36
nbrs = 1

# Feature Selection
# alh = Optimization.pso(50, tf.Objective_Function, -10, 10, 2, 20,w=0.5, c1=1, c2=1)
# animation(alh.get_agents(), tf.Objective_Function, -10, 10)
# animation3D(alh.get_agents(), tf.Objective_Function, -10, 10)
# PSOBestVal=alh.get_agents()
# PSOBestVal1=np.min(PSOBestVal[len(PSOBestVal)-1])
# PSOBestVal2=PSOBestVal1-round(PSOBestVal1)
# def c_Psomap(data, dim, nbrs):
# 	def mds(D, dimensions = dim):
# 	    E = (-0.5 * D**PSOBestVal2)
# 	    rowmean = np.mat(np.mean(E,1))
# 	    columnmean = np.mat(np.mean(E,0))

# 	    F = np.array(E - np.transpose(rowmean) - columnmean + np.mean(E))

# 	    [U, S, V] = npl.svd(F)

# 	    Y = U * np.sqrt(S)

# 	    return Y[:,0:dimensions]
    
# 	def create_G(X, nbrs):
# 		from sklearn.neighbors import NearestNeighbors, kneighbors_graph

# 		neighbors = NearestNeighbors(nbrs)
# 		neighbors.fit(data)

# 		graph = kneighbors_graph(neighbors, nbrs)

# 		from sklearn.utils.graph import graph_shortest_path
# 		G = graph_shortest_path(graph, method='D', directed=False)

# 		return G

# 	G = create_G(data, nbrs)
# 	return mds(G)
def c_Psomap(x_value):
        sc = MinMaxScaler(feature_range = (0, 1))
        x_value = sc.fit_transform(x)
  
        return x_value   

x_value = c_Psomap(x)   
print("Feature Selection PSO ",x_value.shape)

#--------------------------------------------------------------------------------------------
"Data Splitting Process "
print("---------------------------------------------")
print()
print("Data Splitting ")
print()
x_train,x_test,y_train,y_test = train_test_split(x_value,y,test_size = 0.20,random_state = 42)

print("x_train shape",x_train.shape)
print("y_train shape",y_train.shape)
print("x_test shape",x_test.shape)
print("y_test shape",y_test.shape)


#-----------------------------------------------------------------------------------

print("---------------------------------------------")
print()
print("Hybrid Adaboost and Random Forest ")
print()

"Random Forest Algorithm "
clf = RandomForestClassifier(n_estimators=100,n_jobs=-1);
"Adaboost Algorithm "

bclf = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)

bclf.fit(x_train,y_train)

dt = bclf.predict(x_test)

predictions = pd.DataFrame(data=dt)

result = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, predictions)
# print("Classification Report:",)
# print (result1)
# # print("Accuracy:",accuracy_score(y_test, predictions))

# print("-------------------------------------------------------")
# print()

# print("Accuracy:",accuracy_score(y_test, predictions)*100)

sns.heatmap(result, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()

def plot_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(y_test, predictions)
dt = DecisionTreeClassifier(criterion = "gini", random_state = 5,max_depth=10, min_samples_leaf=10)
dt.fit(x_train, y_train)
y_pred1=dt.predict(x_test)


print("---------------------------------------------")
print()
print("Hybrid Adaboost and Random Forest-- Prediction on Attack and Non attack  ")
print()

import tkinter as tk
from easygui import *
text_query = "Enter the OTFS  VANET"
title = "OTFS  VANET"
inp = enterbox(text_query, title)
inp=int(inp)
print(y_pred1[inp])
if (y_pred1[inp] ==0 ):
    print("PREDICTION  OTFS VANET ATTACK ")
    root = tk.Tk()
    T = tk.Text(root, height=30, width=100,font=12)
    T.pack()
    T.insert(tk.END, "ATTACK")
    tk.mainloop()
elif (y_pred1[inp] ==1 ):
    print("NON ATTACK ")
    root = tk.Tk()
    T = tk.Text(root, height=30, width=100,font=12)
    T.pack()
    T.insert(tk.END, "PREDICTION OTFS VANET NON ATTACK ")
    tk.mainloop()








