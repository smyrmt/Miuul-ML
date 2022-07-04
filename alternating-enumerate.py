def change_str(arg):
    temp = ""

    for i in range(len(arg)):
        if i % 2 == 0:
            temp += arg[i].upper()
        else:
            temp += arg[i].lower()
    print(temp)

    #the other way (enumerate)
"""
str = "miuul"
change_str(str)


def change_str(arg):
    temp = ""

    for i, v in enumerate(arg):
        if i % 2 == 0:
            temp += arg[i].upper()
        else:
            temp += arg[i].lower()
    print(temp)


str = "miuul"
change_str(str)
"""
