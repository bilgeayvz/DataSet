#Hangi tür müşterilerimiz hangi oranda ürün almıs anlam çıkar.
import pandas as pd

data=pd.read_csv("dataSet/kiyafet.csv",sep=';')
print(data)

#print(data.shape)
print(data.describe())


data["SatinAldiMi"]=[1 if i== "Evet" else 0 for i in data ["SatinAldiMi"]]
data["Cinsiyet"]=[1 if i== "Kadin" else 0 for i in data["Cinsiyet"]]

#axis in 1 olması sütun 0 olması satır demek.
data.drop(data.columns[0],axis=1,inplace=True)
print(data)

print(data.groupby(["SatinAldiMi"]).count())# groupby=verilerinizi ayrı gruplara ayırmanıza olanak tanır.
data.dropna(inplace=True)
print(data.info())

import matplotlib.pyplot as plt
data.hist(bins=50,figsize=(20,15)) #x eksenini kaç parçaya bölceğini gösterir =bins hist=histogram bir veri kümesinin frekans dağılımını görmemizi sağlar.
plt.show()

corr=data.corr()
print(corr["SatinAldiMi"].sort_values(ascending=False))

data["Gun"]=[0 if i<5 else 1 for i in data["Gun"]]
corr=data.corr()
print(corr["SatinAldiMi"].sort_values(ascending=False))
#y değeri sınıfı temsil eder(0=almadı 1 =aldı)
y=data["SatinAldiMi"].values
#x değeri öznitelikleri temsil eder
x=data.drop(["SatinAldiMi"],axis=1)
print("-----------------------------------------")
print(y)
#Karar ağacı öğrenme modelini uyguluycaz.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)#Her defasında aynı veri üstünden işlem yapması için=random_state=1 dedik)

#Karar ağacı eğitimi
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
clf=dt.fit(x_train,y_train)

#Karar ağacı testi
print("Karar Ağacı Modeli Test Doğruluğu:{}".format(dt.score(x_test,y_test)))

from sklearn.metrics import confusion_matrix
y_pred=dt.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


from sklearn import tree
text_representation=tree.export_text(clf)
print(text_representation)