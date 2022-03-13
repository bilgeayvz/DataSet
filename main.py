import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("dataSet/EvFiyat.csv")

#print(veriler.columns)
#print(veriler)

#İlk 5 değeri getirir
print(veriler.head())

#Son 5 değeri getirir #parametre değeri verirsen mesela 2 son 2 yi getirir.
print(veriler.tail())

#Satır ve Sütun sayısını gösterir.
print(veriler.shape)



#Sütünlara ilişkin biraz detay bilgi verir.(boş olmayan satır sayısını ve veri türünü)
print(veriler.info())

#özniteliklere iliştik istatistiksel bilgi verir.
print(veriler.describe())

#Grafik çizelim
plt.scatter(veriler.Alan,veriler.Fiyat)
plt.xlabel("Alan m^2")
plt.ylabel("Fiyat*1000 TL")
plt.title("Alan-Fiyat ilişkisi")
plt.grid(True)
plt.show()

korelasyon_katsayisi=veriler.corr()
print(korelasyon_katsayisi["Fiyat"].sort_values(ascending=False))

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()

veriler.dropna(inplace=True)
print(veriler)
x=veriler.Alan.values.reshape(-1,1) #satır önemseme bütün değerleri al değişkene çek.
y=veriler.Fiyat.values.reshape(-1,1)
print(linear_reg.fit(x,y))

y_ekseni_kesisim= np.array([0]).reshape(-1,1)
b0=linear_reg.predict(y_ekseni_kesisim)
print("b0:",b0)

#eğimin bulunması
b1=linear_reg.coef_
print("b1",b1)

###
array=np.array([100,110,120,130,140,150,160,200,174,170,180,190,200,80,90,105,115,125]).reshape(-1,1) #Diziye
                                                                                    #dönüştürmeye yarar

plt.figure()
plt.scatter(x,y)
y_head=linear_reg.predict(array)
plt.plot(array,y_head,color="purple")
plt.xlabel("Alan (m2)")
plt.ylabel("Fiyat (TL)")
plt.title("Alan-Fiyah İlişkisi")
plt.grid(True)
plt.show()
