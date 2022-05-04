import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Logistic:
# school: MS:0, GP:1
# sex: F:0, M:1
# address: R:0, U:1
# famsize: LE3:0, GT3:1
# Pstatus: T:0, A:1
# schoolsup, famsup, paid, activites, nursery, highter, internet, romantic:
# yes: 1, no: 0


binaryMap = {"MS":0, "GP":1, "F":0, "M":1, "R":0, "U":1, "LE3":0, "GT3":1, "T":0, "A":1,
             "services":1, "at_home":2, "health":3, "teacher":4, "other":0,
             "home":1, "reputation":2, "course":3, "mother":1, "father":2, "yes": 1, "no":0}

def binary(df):
    dfcopy = df.copy(deep=True)
    ColumnList = ['school','sex','address', 'famsize', 'Pstatus',
                  'schoolsup', 'famsup', 'paid', 'activities',
                  'nursery', 'higher', 'internet', 'romantic']
    for col in ColumnList:
        dfcopy[col] = dfcopy[col].map(binaryMap)
    return dfcopy

def convert2onehot(df):
    features_list = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                   'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                   'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                   'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                   'Walc', 'health', 'absences']
    label_list = ['G1', 'G2', 'G3']
    df_train = df[features_list]
    df_label = df[label_list]
    df_one_hot = pd.get_dummies(df_train)
    df_result = pd.concat([df_one_hot, df_label], axis=1)
    return df_result

def Convert2Label(df, head=None):
    # Convert Origin G1 G2 G3 col to class format
    if head is None:
        head = ["G1", "G2", "G3"]

    def classify(x):
        if 0<=x<=9:
            return 4
        elif 10<=x<=11:
            return 3
        elif 12<=x<=13:
            return 2
        elif 14<=x<=15:
            return 1
        else:
            return 0
    df_Processed = df.copy(deep=True)
    ColumnList = head
    for col in ColumnList:
        df_Processed[col] = df_Processed[col].apply(classify)
    return df_Processed

def standardize(data):
    data_processed = data.copy()
    transform = StandardScaler()
    data_processed = transform.fit_transform(data_processed)
    return data_processed

def process(data, onehot=True, labels=False):
    df_binary = binary(data)
    if onehot:
        df_onehot = convert2onehot(df_binary)
    else:
        df_onehot = df_binary
    if labels:
        df_result = Convert2Label(df_onehot, labels)
    else:
        df_result = df_onehot
    return df_result

def confusionMatrix(ture_label, pred_label, classes, title):
    cm = confusion_matrix(ture_label, pred_label)
    cm_plot = ConfusionMatrixDisplay(cm, display_labels=classes).plot()
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    DataTrain = pd.read_csv("./student_performance_train.csv")
    DataTest = pd.read_csv("./student_performance_test.csv")
    result = process(DataTrain)
    array = result.values
    print(result.values)
    print(0)
