#çift sayıların karesini alarak sözlüğe ekleme
dict = {}
num = range(10)
for i in num:
    if i % 2 == 0:
        dict[i] = i ** 2
    else: 
        dict[i] = i
print(dict)

"""
#with dictionary comprehensions 
dict = {i: i ** 2 for i in num if i % 2 == 0}
"""

#exmp 1
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]

#exmp 2
df.columns = ["flag_" + col if "INS" in col else "no_flag_" + col for col in df.columns]
