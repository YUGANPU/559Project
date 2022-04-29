import pandas as pd

# Logistic:
# school: MS:0, GP:1
# sex: F:0, M:1
# address: R:0, U:1
# famsize: LE3:0, GT3:1
# Pstatus: T:0, A:1
# Mjob&Fjob: servives:1, at_home:2, health:3, teacher:4, other:0
# reason: home:1, reputation:2, course:3, other: 0
# guardian: mother:1, father:2, other: 0
# schoolsup, famsup, paid, activites, nursery, highter, internet, romantic:
# yes: 1, no: 0


binaryMap = {"MS":0, "GP":1, "F":0, "M":1, "R":0, "U":1, "LE3":0, "GT3":1, "T":0, "A":1,
             "services":1, "at_home":2, "health":3, "teacher":4, "other":0,
             "home":1, "reputation":2, "course":3, "mother":1, "father":2, "yes": 1, "no":0}

def binary(df):
    ColumnList = ['school','sex','address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                  'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                  'nursery', 'higher', 'internet', 'romantic']
    for col in ColumnList:
        df[col] = df[col].map(binaryMap)
    return df

def Convert2Label(df):
    # Convert Origin G1 G2 G3 col to class format
    def classify(x):
        if 0<=x<=9:
            return 5
        elif 10<=x<=11:
            return 4
        elif 13<=x<=12:
            return 3
        elif 14<=x<=15:
            return 2
        else:
            return 1
    df_Processed = df.copy(deep=True)
    ColumnList = ["G1", "G2", "G3"]
    for col in ColumnList:
        df_Processed[col] = df_Processed[col].apply(classify)
    return df_Processed