import pandas as pd

# Logistic:
# school: MS:0, GP:1
# sex: F:0, M:1
# address: R:0, U:1
# famsize: LE3:0, GT3:1
# Pstatus: T:0, A:1
# Mjob&Fjob: servives:1, at_home:2, health:3, teacher:4, other:5
# reason: home:1, reputation:2, course:3, other: 5
# guardian: mother:1, father:2, other: 3
# schoolsup, famsup, paid, activites, nursery, highter, internet, romantic:
# yes: 1, no: 0


DataTrain = pd.read_csv("./student_performance_train.csv")
DataTest = pd.read_csv("./student_performance_test.csv")


print(0)