#Görev 2
text = "The goal is to turn data into information, and information into insight."
str = text.upper().replace(",","").replace(".","").split(" ")
print(str)

#Görev 3
list = ["D","A","T","A","S","C","I","E","N","C","E"]

print(len(list))      #Adım1: Verilenlisteninelemansayısınabakınız.
list[0]               #Adım2: Sıfırıncıveonuncuindekstekielemanlarıçağırınız.
list[10]              
newlist = list[0:4]   #Adım3: Verilenlisteüzerinden["D", "A", "T", "A"] listesioluşturunuz.
list.pop(8)           #Adım4: Sekizinciindekstekielemanısiliniz.
list.append("S")      #Adım5: Yeni birelemanekleyiniz.
list.insert(8,"N")    #Adım6: Sekizinciindekse"N" elemanınıtekrarekleyiniz.

#Görev 4
dict.keys()                           #Adım1: Key değerlerine erişiniz.
dict.values()                       #Adım2: Value'lara erişiniz.
dict["Daisy"][1] = 13                #Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Ahmet"] = ["Turkey", 24]      #Adım4: Key değeri Ahmet value değeri[Turkey,24] olan yeni bir değer ekleyiniz.
dict.pop("Antonio")                  #Adım5: Antonio'yu dictionary'den siliniz.



#Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.Liste elemanlarına tek tek erişmeniz gerekmektedir.

def divide(list):
    A = []
    B = []
    for i in list:
        if i % 2 == 0:
            A.append(i)
        else:
            B.append(i)
    return A, B


list = range(10)
divide(list)

#Görev 6: ListComprehension yapısıkullanarakcar_crashesverisindekinumeric değişkenlerinisimlerinibüyükharfeçevirinizvebaşınaNUM ekleyiniz.
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
["NUM_" + i.upper() if i != "abbrev" else i.upper() for i in df.columns]


#Görev 7: ListComprehension yapısıkullanarakcar_crashesverisindeisminde"no" barındırmayandeğişkenlerinisimlerininsonuna"FLAG" yazınız.

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns
[i.upper() + "_FLAG" if ("no" in i) == False else i.upper() for i in df.columns]

#Görev 8: ListComprehension yapısıkullanarakaşağıdaverilendeğişkenisimlerindenFARKLI olandeğişkenlerinisimleriniseçinizveyeni birdataframeoluşturunuz

import seaborn as sns

df = sns.load_dataset("car_crashes")
ogr_list = ["abbrev", "no_previous"]
new_cols = []
[new_cols.append(i) for i in df.columns if (i in ogr_list) != True]
new_df = df[new_cols]
new_df
