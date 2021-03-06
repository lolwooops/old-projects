#%%
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# %%
df = pd.read_csv("C:/Users/Tim/Desktop/2012.csv")

#np.any(np.isnan(df))
#np.all(np.isfinite(df))

#%%
#def clean_dataset(df):
#    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#    df.dropna(inplace=True)
#    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#    return df[indices_to_keep].astype(np.float64)

#clean_dataset(df)

#df.replace([np.inf, -np.inf], np.nan, inplace=True)
#df.fillna(0, inplace=True)


#%%
df.describe()

#%%
df = df.dropna()

over100=df["age"]>100
less10=df["age"]<10
df=df[~(over100|less10)]

over350 = df["weight"] > 350
less50 = df["weight"] < 50
df = df[~(over350|less50)]

bl = {'Y': 1, 'N': 0}
df.arstmade = [bl[item] for item in df.arstmade]

df = df[df["sex"] != "Z"]
sx = {'M': 1, 'F': 0}
df.sex = [sx[item] for item in df.sex]

# %%
cols=[
	"perobs",
	"perstop",
	"age",
	"weight",
	"ht_feet",
    "arstmade",
    "sex",
]

for col in cols:
	df[col]=pd.to_numeric(df[col],errors="coerce")

df.describe()

# %%
v1x = df["race"].value_counts()
sns.barplot(x=v1x.index, y=v1x.values, palette="rocket", data=df)
plt.xlabel("Race")
plt.ylabel("Amount")
plt.legend(["B: Black", "Q: Hispanic", "W: White", "P: Black Hispanic", "A: Asian", "U,Z,I: Other"])

df = df[df["age"] < 101]
df = df[df["age"] > 9]
sns.distplot(df["age"])

v3x = df["sex"].value_counts()
sns.countplot(x="race",hue="sex",data=df)
plt.legend(["Female", "Male"])

df=df[df["perobs"]<200]
sns.distplot(df["perobs"])

armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]

df["armed"] = df[df[armed] == "Y"]
sns.countplot(df["race"],hue="armed")

# %%
sns.scatterplot(x="perobs",y="arstmade",data=df)

sns.scatterplot(y="perobs",x="race",hue="arstmade",data=df)

df=df[df["perobs"]<11]
sns.boxplot(y="perobs",x="race",hue="arstmade",data=df)
g = sns.FacetGrid(df, col="race", row="sex")
g = g.map(plt.hist, "perobs")

df['age_bins'] = pd.cut(x=df['age'], bins=[10,19, 29, 39, 49,59,69,79,89,99])
sns.boxplot(x="perobs",y="age_bins",hue="arstmade",data=df)

# %%
col2 = [
	"crimsusp",
	"pf_hands",
	"pf_wall",
	"pf_grnd",
	"pf_drwep",
	"pf_ptwep",
	"pf_baton",
	"pf_hcuff",
	"pf_pepsp",
	"pf_other",
	"rf_vcrim",
	"rf_othsw",
	"rf_attir",
	"cs_objcs",
	"cs_descr",
	"cs_casng",
	"cs_lkout",
	"rf_vcact",
	"cs_cloth",
	"cs_drgtr",
	"cs_furtv",
	"rf_rfcmp",
	"rf_verbl",
	"cs_vcrim",
	"cs_bulge",
	"cs_other",
	"rf_knowl",
	"sb_hdobj",
	"sb_outln",
	"sb_admis",
	"sb_other",
]
col3 = [
	"crimsusp",
	"pf_hands",
	"pf_wall",
	"pf_grnd",
	"pf_drwep",
	"pf_ptwep",
	"pf_baton",
	"pf_hcuff",
	"pf_pepsp",
	"pf_other",
	"detailcm"
]
col4 = [
	"pf_hands",
	"pf_wall",
	"pf_grnd",
	"pf_drwep",
	"pf_ptwep",
	"pf_baton",
	"pf_hcuff",
	"pf_pepsp",
	"pf_other",
	"detailcm"
]

df3=df[col3]

df3=df3.replace(["N","Y"],[0,1])
d3m=df3.groupby("crimsusp").mean()
sns.pairplot(data=d3m)
d3m.describe()

df4 = df[col4]
df4 = df4.replace(["N","Y"],[0,1])
d4m = df4.groupby("detailcm").mean()

plt.subplots(figsize=(10,30))
d4mp = sns.heatmap(d4m)
d4mp.figure.savefig("4.png")


# %%
pfs = [col for col in df.columns if col.startswith("pf_")]
#pfs.rename(columns={
	#"pf_baton":"Baton",
	#"pf_drwep":"WeapDrawn",
	#"pf_grnd":"Ground",
	#"pf_hands":"Hands",
	#"pf_hcuff":"Handcuffs",
	#"pf_other":"Other",
	#"pf_pepsp":"PSpray",
	#"pf_ptwep":"WeapPoint",
	#"pf_wall":"Wall",
#},inplace=True)

rfs = [col for col in df.columns if col.startswith("rf_")]
#rfs.rename(columns={
	#"rf_attir":"InappAttire",
	#"rf_bulg":"Bulge",
	#"rf_furt":"FurtiveMove",
	#"rf_knowl":"PriorKnow",
	#"rf_othsw":"OtherWeapon",
	#"rf_rfcmp":"NoComply",
	#"rf_vcact":"ViolCrime",
	#"rf_vcrim":"ViolCrimesusp",
	#"rf_verbl":"VerbalThreat",
#},inplace=True)

css = [col for col in df.columns if col.startswith("cs_")]
#css.rename(columns={
	#"cs_bulge":"Bulge",
	#"cs_casng":"Casing",
	#"cs_cloth":"CrimeClothe",
	#"cs_descr":"RelevantDesc",
	#"cs_drgtr":"DrugTrans",
	#"cs_furtv":"FurtiveMove",
	#"cs_lkout":"Lookout",
	#"cs_objcs":"SuspObj",
	#"cs_other":"Other",
	#"cs_vcrim":"ViolCrime",
#},inplace=True)

acs = [col for col in df.columns if col.startswith("ac_")]
#acs.rename(columns={
	#"ac_assoc": "association",
	#"ac_cgdir": "ChngDirec",
	#"ac_evasv": "EvasiveResp",
	#"ac_incid": "PlaceIncidence",
	#"ac_inves": "OngInvestigation",
	#"ac_other": "Other",
	#"ac_proxm": "Proximity",
	#"ac_rept": "Reported",
	#"ac_stsnd": "Sighted",
	#"ac_time": "TimeIncidence",
#},inplace=True)

#include sex and race, need to ohe into boolean
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
r1 = label_encoder.fit_transform(df["race"])

onehot_encoder = OneHotEncoder(sparse=False)
r1 = r1.reshape(len(r1), 1)
race1 = onehot_encoder.fit_transform(r1)
race1 = pd.DataFrame(race1)
race1 = race1.drop(race1.columns[[2, 5, 7]], axis=1)
race1.rename(columns={
	0: "Asian",
	1: "Black",
	3: "Black Hispanic",
	4: "Hispanic",
	6: "White",
},inplace=True)

s1 = label_encoder.fit_transform(df["sex"])

s1 = s1.reshape(len(s1), 1)
sex1 = onehot_encoder.fit_transform(s1)
sex1 = pd.DataFrame(sex1)
sex1 = sex1.drop(sex1.columns[[2]],axis=1)
sex1.rename(columns={
	0: "Female",
	1: "Male"
},inplace=True)

q2var = [
	race1,
	sex1
]

q2var2 = [
    "arstmade",
    "sumissue",
    "frisked",
    "searched",
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
]

#combine into 1 df
X = df[pfs + acs + css + rfs + q2var2]
X = pd.concat([X, race1, sex1], axis=1)

X2 = X.replace(["N","Y"],[False,True])
X2 = X2.replace([0,1],[False,True])

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

#te = TransactionEncoder()
#te_ary = te.fit(X).transform(X)
#xte = pd.DataFrame(te_ary, columns = te.columns_)

freqitem = apriori(
	X2, 
	min_support=0.25, 
	use_colnames=True,
	verbose=1
)
freqitem
freqitem.to_csv("freqitem.csv")

from mlxtend.frequent_patterns import association_rules
rules = association_rules(
	freqitem,
	metric="confidence",
	min_threshold=0.1,
)
rules.head()

sorted_rules = rules.sort_values("confidence", ascending=False)
sorted_rules.head(10)

# %%
#clustering
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score

df = pd.read_csv("C:/Users/Tim/Desktop/2012.csv")

cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "xcoord",
    "ycoord",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

import numpy as np
from tqdm import tqdm

value_label = pd.read_excel(
    "2012 SQF File Spec.xlsx",
    sheet_name="Value Labels",
    skiprows=range(4)
)
value_label["Field Name"] = value_label["Field Name"].fillna(
    method="ffill"
)
value_label["Field Name"] = value_label["Field Name"].str.lower()
value_label["Value"] = value_label["Value"].fillna(" ")
value_label = value_label.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in value_label]

for col in tqdm(cols):
    df[col] = df[col].apply(
        lambda val: value_label[col].get(val, np.nan)
    )

df["trhsloc"] = df["trhsloc"].fillna("P (unknown)")
df = df.dropna()

import pyproj
srs = "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
p = pyproj.Proj(srs)

df["coord"] = df.apply(
    lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1
)

df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54
df = df[(df["age"] <= 100) & (df["age"] >= 10)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]

df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",        
        
        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",

        # location of stop 
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)

df_murder = df[df["detailcm"] == "MURDER"]

df_murder["lat"] = df["coord"].apply(lambda val: val[1])
df_murder["lon"] = df["coord"].apply(lambda val: val[0])

#%%
#location
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10
for k in tqdm(range(num_city, num_pct, step)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_murder[["lat", "lon"]])
    silhouette_scores[k] = silhouette_score(df_murder[["lat", "lon"]], y)
    labels[k] = y


from sklearn.cluster import KMeans
silhouette_scores, labels = {}, {}
inertia, homogeneity= {}, {}
for k in range(1, 10):
	km = KMeans(n_clusters=k)
	pred = km.fit_predict(df_murder[["lat","lon"]])
	inertia[k] = km.inertia_
	labels[k] = pred

import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.lineplot(
	x=list(inertia.keys()),
	y=list(inertia.values()),
	color="blue",
	label="inertia",
	legend=None,
)
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
pred = km.fit_predict(df_murder[["lat","lon"]])
## metrics
from sklearn.metrics import homogeneity_score
print(f"homogeneity: {homogeneity_score(df_murder[["lat","lon"]], pred)}")
print(f"inertia: {km.inertia_}")


import seaborn as sns
ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)
ax.get_figure().savefig("trend.png", bbox_inches="tight", dpi=400)

sns.scatterplot(x="lat",y="lon", data=df_murder)


import folium
nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_murder["label"] = labels[best_k]
colors = sns.color_palette("hls").as_hex()
for row in tqdm(df_murder[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)
m

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_murder["label"] = y
colors = sns.color_palette("hls", len(np.unique(y))).as_hex()
for row in tqdm(
    df_murder.loc[df_murder["label"] == 63, ["lat", "lon", "label"]].to_dict(
        "records"
    )
):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m

#%%
#reason for stop
dfa = df[df["arstmade"]=="YES"]

dfa["lat"] = dfa["coord"].apply(lambda val: val[1])
dfa["lon"] = dfa["coord"].apply(lambda val: val[0])

css = [col for col in df.columns if col.startswith("cs_")]

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm

c = DBSCAN()
x = dfa[css] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
dfa["label"] = y
colors = sns.color_palette("hls", len(np.unique(y))).as_hex()
for row in tqdm(dfa[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m

#%%
pfs = [col for col in df.columns if col.startswith("pf_")]

dfa = df[df["arstmade"]=="YES"]

dfa["lat"] = dfa["coord"].apply(lambda val: val[1])
dfa["lon"] = dfa["coord"].apply(lambda val: val[0])

c = DBSCAN()
x = dfa[pfs] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
dfa["label"] = y
colors = sns.color_palette("hls", len(np.unique(y))).as_hex()
for row in tqdm(dfa[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m

dfa2 = dfa
del dfa2["pf_hands"]
del dfa2["pf_hcuff"]

pfs = [col for col in dfa2.columns if col.startswith("pf_")]

c = DBSCAN()
x = dfa2[pfs] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))

#sns.scatterplot(x=df_black["lat"],y=df_black["lon"])
#sns.scatterplot(x=df_white["lat"],y=df_white["lon"])
#plt.show

#%%
#3 classification models; 1task
cldf = pd.read_csv("C:/Users/Tim/Desktop/2012.csv")
#cldf.dropna()
#cldf = cldf.replace(r'^\s*$', 0, regex=True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
r1 = label_encoder.fit_transform(df["race"])

onehot_encoder = OneHotEncoder(sparse=False)
r1 = r1.reshape(len(r1), 1)
race1 = onehot_encoder.fit_transform(r1)
race1 = pd.DataFrame(race1)
race1 = race1.drop(race1.columns[[2, 5, 7]], axis=1)
race1.rename(columns={
	0: "Asian",
	1: "Black",
	3: "Black Hispanic",
	4: "Hispanic",
	6: "White",
},inplace=True)

col = [
	"arstmade",
	"perobs",
	"contrabn",
	"pistol",
	"riflshot",
	"asltweap",
	"knifcuti",
	"machgun",
	"othrweap",
	"age",
	"sex",
	"race",
	"ht_feet",
	"weight",
	"detailcm",
]

cldf=cldf[col]

cldf["arstmade"] = cldf["arstmade"].replace(
	["N", "Y"], [0,1]
)
cldf = cldf.replace(
	["N", "Y"], [0,1]
)

cldf = cldf[cldf["race"] != "U"]
cldf = cldf[cldf["race"] != "Z"]
cldf = cldf[cldf["race"] != "I"]
cldf["race"] = cldf["race"].replace(
	["B", "Q", "W", "P", "A"], [0,1,2,3,4]
)
cldf = cldf[cldf["sex"] != "Z"]
cldf["sex"] = cldf["sex"].replace(
	["F","M"],[0,1]
)

over100=cldf["age"]>100
less10=cldf["age"]<10
cldf=cldf[~(over100|less10)]

over350 = cldf["weight"] > 350
less50 = cldf["weight"] < 50
cldf = cldf[~(over350|less50)]

for col in col:
    cldf[col]= pd.to_numeric(cldf[col], errors="coerce")

cldf = cldf.dropna()

cldf = cldf.reset_index()
cldf.replace([np.inf, -np.inf], np.nan, inplace=True)
cldf.fillna(0, inplace=True)

cldf2=cldf

#%%
#OBSOLETE CELL, IGNORE
y = cldf["arstmade"]
cldf.drop("arstmade",axis=1,inplace=True)
x = cldf
#x = x.replace(
#	["N","Y"],[0,1]
#)

#x = x.values.astype(np.float)
#y = y.values.astype(np.float)
#x = x.replace(r'^\s*$', 0, regex=True)

x.drop(columns=["index"])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	x, y, random_state=111
)


#%%
#x=pd.concat([x,race1],axis=1)
cldf2=pd.concat([cldf2,race1],axis=1)
cldf2.drop(["race"],axis=1)

cldf2 = cldf2.dropna()

cldf2 = cldf2.reset_index()
cldf2.replace([np.inf, -np.inf], np.nan, inplace=True)
cldf2.fillna(0, inplace=True)

y2 = cldf2["arstmade"]
cldf2.drop("arstmade",axis=1,inplace=True)
x2 = cldf2
x2.drop(columns=[
	"index",
	"race",
])
x2.dropna()

del x2['level_0']
del x2['index']

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
	x2, y2, random_state=111
)


#%%
#model 1: knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x2_train, y2_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
y2_pred = knn.predict(x2_test)
print(f"accuracy: {accuracy_score(y2_test, y2_pred)}")
f1_score(y2_test,y2_pred,average="weighted")
f1_score(y2_test,y2_pred,average="micro")
f1_score(y2_test,y2_pred,average="macro")

from sklearn import metrics
plt.figure(figsize=(9,9))
cm = metrics.confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#decision region
#from mlxtend.plotting import plot_decision_regions
#plot_decision_regions(x=x_train, y=y_train, clf=knn)
#plt.show()
#doesn't work for 2+ # of features

#%%
#model 2: lreg
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x2_train, y2_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
y2_pred = lr.predict(x2_test)
print(f"accuracy: {accuracy_score(y2_test, y2_pred)}")
f1_score(y2_test,y2_pred,average="weighted")

from sklearn.metrics import classification_report 
print(classification_report(y2_test, y2_pred))
#m = classification_report(y_test, y_pred)
#m.to_csv("lregcrep.csv")

#feature importance
importance = lr.coef_[0]
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()

from sklearn import metrics
plt.figure(figsize=(9,9))
cm = metrics.confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#%%
#model 3: dtrees
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x2_train, y2_train)

y2_pred = dt.predict(x2_test)
print(f"accuracy: {accuracy_score(y2_test, y2_pred)}")
f1_score(y2_test,y2_pred,average="weighted")
print(classification_report(y2_test, y2_pred))

from sklearn import metrics
plt.figure(figsize=(9,9))
cm = metrics.confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#feature importance
importance = dt.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()



# %%
