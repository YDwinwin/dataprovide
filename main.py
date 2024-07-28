import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN

df_test = pd.read_csv("UNSW_NB15_training-set.csv")
df_train = pd.read_csv("UNSW_NB15_testing-set.csv")
# 获得各种攻击类型
attack_cat_train = df_train["attack_cat"]
attack_cat_test = df_test["attack_cat"]
attack_cat_examples = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
                       "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
# print(attack_cat_train.value_counts()) #输出训练集对应的攻击类型与其数量
# print(attack_cat_test.value_counts())  #输出测试集对应的攻击类型与其数量
# print(df_train["label"].value_counts()) #统计标签类别与数量
df_train_label = df_train["label"]
df_test_label = df_test["label"]
df_train = df_train.drop(["label", "id"], axis=1)  # 预处理（删除id列与label列与攻击类型列）
df_test = df_test.drop(["label", "id"], axis=1)  # 预处理（删除id列与label列与攻击类型列）
list_train = []
list_test = []


def split_category(data, columns):  # 分离字符类型的特征
    cat_data = data[columns]  # 分离出的四个离散变量
    rest_data = data.drop(columns, axis=1)  # 剩余的特征
    return rest_data, cat_data


# 统计每列特征字符串的类型
categorical_mask = (df_train.dtypes == object)
# 统计字符串类型为字符的特征
categorical_columns = df_train.columns[categorical_mask].tolist()


# 分离出来的字符类型的四列特征
# print(df_train[categorical_columns])
# pd.DataFrame(df_train[categorical_columns]).to_csv('.txt', index=False)#保存可以观察
# pd.DataFrame(df_test[categorical_columns]).to_csv('.txt', index=False)
def label_encoder(data):  # 字符编码
    labelencoder = LabelEncoder()
    for col in data.columns:
        data.loc[:, col] = labelencoder.fit_transform(data[col])
    return data


# 将训练集的字符类型变量进行编码
df_train[categorical_columns] = label_encoder(df_train[categorical_columns])
# 将测试集的字符类型变量进行编码
df_test[categorical_columns] = label_encoder(df_test[categorical_columns])

# ---------------------------重采样--------------------------------------
oversample = ADASYN()
df_train, df_train_label = oversample.fit_resample(df_train, df_train_label)

# 将测试集与训练集的字符类型特征与其他特征分离，以便后续单独处理。
df_train, df_train_cat = split_category(df_train, categorical_columns)
df_test, df_test_cat = split_category(df_test, categorical_columns)
test_attack_cat = df_test_cat["attack_cat"]
train_attack_cat = df_train_cat["attack_cat"]


def Onehotencoding(data):  # 使用PCA降维算法 对训练与测试集的三个字符类型的数字编码进行降维与独热
    data = np.array(data)
    data = data.astype(float).astype(int)
    pca = PCA(n_components=2)  # 设置降维后的维度
    reduced_data = pca.fit_transform(data)
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(data).toarray()
    return encoded_data


# 不对攻击类型进行独热处理：在网络入侵检测中，攻击类型是一个非常重要的特征，它可以帮助我们理解和分析入侵行为。因此，保留原始的攻击类型特征可能更有助于后续的分析和建模。
df_train_cat_onehot = Onehotencoding(df_train_cat.iloc[:, :-1])  # loc是基于标签选取数据，iloc是基于位置选取数据
df_test_cat_onehot = Onehotencoding(df_test_cat.iloc[:, :-1])
# pd.DataFrame(df_train_cat_onehot).to_csv('.txt', index=False)
# pd.DataFrame(df_test_cat_onehot).to_csv('.txt', index=False)
datatrain_df = pd.DataFrame(df_train_cat_onehot)
datatest_df = pd.DataFrame(df_test_cat_onehot)
datatrain_df, datatest_df = datatrain_df.align(datatest_df, join='inner', axis=1)
datatrain_df.fillna(0, inplace=True)  # 用NAN填充数据集中的空值
datatest_df.fillna(0, inplace=True)  # 用NAN填充数据集中的空值
datatrain_df.columns = datatrain_df.columns.astype(str)
datatest_df.columns = datatest_df.columns.astype(str)
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)
df_train.columns = df_train.columns.astype(str)
df_test.columns = df_test.columns.astype(str)
df_train = pd.concat([df_train, datatrain_df], axis=1)  # 数据合并
df_test = pd.concat([df_test, datatest_df], axis=1)

min_max_scaler = MinMaxScaler()  # 特征值归一化
df_train = min_max_scaler.fit_transform(df_train)
df_test = min_max_scaler.fit_transform(df_test)
# --------------------把分类标签编码---------------------------------------------
df_train_label_encode = LabelEncoder().fit_transform(df_train_label)
df_test_label_encode = LabelEncoder().fit_transform(df_test_label)
# 保存文件
# pd.DataFrame(df_train_label_encode).to_csv('trainlabel.txt', index=False)
# pd.DataFrame(df_train).to_csv('train（预处理）.txt', index=False)
# pd.DataFrame(df_test_label_encode).to_csv('testlabel.txt', index=False)
# pd.DataFrame(df_test).to_csv('test（预处理）.txt', index=False)#pd.DataFrame(df_test_label_encode).to_csv('testlabel.txt', index=False)
# pd.DataFrame(train_attack_cat).to_csv('train（攻击类型编码）.txt', index=False)
# pd.DataFrame(test_attack_cat).to_csv('test（攻击类型编码）.txt', index=False)
