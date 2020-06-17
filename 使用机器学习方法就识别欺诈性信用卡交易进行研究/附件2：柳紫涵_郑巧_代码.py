#!/usr/bin/env python
# coding: utf-8

# # 研究题目：使用机器学习方法就识别欺诈性信用卡交易进行研究

# # 学号 姓名：201825108 柳紫涵 201825111 郑巧

# In[2]:


import numpy as np
import pandas as pd


# ## 数据预处理

# ### 读取数据集并检查数据集中是否存在缺失值

# In[3]:


df = pd.read_csv("creditcard.csv")
ind = list(df.index.values)
col = list(df.columns.values)
df1 = pd.isna(df)
nid = []
for i in ind:
    for j in range(len(col)):
        if df1.iloc[i,j]==True:
            nind=nid+i
print(nid)


# ### 将V1~V28,Time,Amount,Class所在的列分别取出

# In[4]:


dfv = df.iloc[:,1:29]#取出V1~V28
dft = df.loc[:,'Time']
dfa = df.loc[:,'Amount']
dfc = df.loc[:,'Class']


# ### 分辨交易时间与是否为诈骗交易的相关变量

# In[6]:


# 使用直方图展示时间序列中欺诈和正常事件的频率
import matplotlib.pyplot as plt 
plt.subplot(2, 1, 1)
plt.hist(df.Time[df.Class == 1], bins=100)
plt.title('fraud')
plt.ylabel('transaction numbers')
 
plt.subplot(212)
plt.hist(df.Time[df.Class == 0], bins=100)
plt.title('normal')
plt.subplots_adjust(wspace =0, hspace =0.5)#调整子图间距

#诈骗交易随时间序列分布随机，而正常交易呈现明显的双峰状


# ### 特征衍生：将“时间（秒数）”转为“小时”

# In[7]:


df['Hour'] =df["Time"].apply(lambda x : divmod(x, 3600)[0]) #单位转换


# In[8]:


plt.subplot(211)
plt.hist(df.Hour[df.Class == 1], bins=20)
plt.title('fraud')
plt.ylabel('transaction numbers')
 
plt.subplot(212)
plt.hist(df.Hour[df.Class == 0], bins=20)
plt.title('normal')
plt.subplots_adjust(wspace =0, hspace =0.5)#调整子图间距


# ### 分辨交易金额是否为诈骗交易的相关变量

# In[96]:


#同样使用直方图展示金额与交易性质之间的关系
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,4))

bins = 30

ax1.hist(df["Amount"][df["Class"]== 1], bins = bins)
ax1.set_title('fraud')

ax2.hist(df["Amount"][df["Class"] == 0], bins = bins)
ax2.set_title('normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')#考虑到欺诈交易与正常交易数量的差别，将y轴刻度设置为对数刻度，方便比较
plt.show()


# ### 观察金额在两种交易中的统计数值

# In[11]:


print("fraud")
print(df.Amount[df.Class == 1].describe())

print('\v')

print("normal")
print(df.Amount[df.Class == 0].describe())


# ### 通过散点图观察“金额”和“时间（小时）”与交易性质的关系

# In[12]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,6))

ax1.scatter(df.Hour[df.Class == 1], df.Amount[df.Class == 1])
ax1.set_title('fraud')

ax2.scatter(df.Hour[df.Class == 0], df.Amount[df.Class == 0])
ax2.set_title('normal')

plt.xlabel('Time (in Hours)')
plt.ylabel('Amount')
plt.show()


# ## 特征选择

# ### 直观显示标签和各个特征之间的关系

# In[13]:


import matplotlib.gridspec as gridspec
import seaborn as sns

plt.figure(figsize=(12, 29*4))
# 隐式指定网格行数列数（隐式指定子图行列数）
gs = gridspec.GridSpec(29, 1)

for i, cn in enumerate(dfv.columns.values):
    ax = plt.subplot(gs[i])
    sns.distplot(dfv[cn][df.Class == 1], bins=50, color='blue')
    sns.distplot(dfv[cn][df.Class == 0], bins=50, color='green')
    ax.set_title(str(cn))
plt.subplots_adjust(wspace =0, hspace =0.5)#调整子图间距


# ### 使用卡方检验选取特征

# In[99]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
dfv_st = MinMaxScaler().fit_transform(dfv)#使用MinMax方式对V1~V28进行标准化，因为卡方检验需要传递非负的值
X, y = dfv_st, dfc

X_new = SelectKBest(chi2, k = 15).fit_transform(X, y)#选取15个关联度最高的特征
BestFeature = []
for i in range(np.shape(X_new)[1]):
    for j in range(np.shape(X)[1]) :
        if (X_new[0, i] == X[0, j]):
            BestFeature.append(j)
            
DropList = []
for i in range(np.shape(X)[1]):
    if i in BestFeature:
        pass
    else: DropList.append('V'+str(i+1))

dfv_new = dfv.drop(DropList, axis = 1)#剔除相关度较低的特征
print(dfv_new.shape)


# In[102]:


#标准化“金额”和“时间（小时）”，并作为新的特征添加到dfv_new中
from sklearn.preprocessing import StandardScaler
dfv_new['normAmount'] = StandardScaler().fit_transform(dfa.values.reshape(-1,1))
dfv_new['normHour'] = StandardScaler().fit_transform(df['Hour'].values.reshape(-1,1))
dfv_new.head()
#dfv_new.shape


# ### 数据集的划分

# In[103]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dfv_new, dfc,test_size=0.3, random_state=3)
print(X_train.shape, X_test.shape)


# In[104]:


print(1 in y_train,1 in y_test)
#直接划分训练集和测试集的方法显然是不均衡的，所有诈骗样本将被划分至测试集中


# ### 解决方案一：下采样

# In[106]:


X = dfv_new
y = dfc
fraud_index = np.array(df[df.Class == 1].index)
norm_index = np.array(df[df.Class == 0].index)
random_norm_index = np.random.choice(norm_index,492,replace = False)#从正常交易对应的索引值中随机不放回抽取492个
under_sample_index = np.concatenate((fraud_index,random_norm_index),axis=0)#合并抽取的正常交易的索引值和诈骗交易对应的索引值
under_sample_data = dfv_new.iloc[under_sample_index]#取出对应的样本
print(under_sample_data.shape)


# ### 对下采样后的样本进行划分

# In[107]:


from sklearn.model_selection import train_test_split 
under_sample_X = dfv_new.iloc[under_sample_index]
under_sample_y = dfc.iloc[under_sample_index]
X_train_undersample, X_test_undersample,y_train_undersample, y_test_undersample = train_test_split(                                                           under_sample_X, under_sample_y, test_size = 0.3, random_state = 2)


# ### 尝试使用逻辑回归

# In[111]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(solver='lbfgs')
LR.fit(X_train_undersample, y_train_undersample)


# In[112]:


LR_pred = LR.predict(X_test_undersample)
from sklearn.metrics import classification_report
print(classification_report(y_test_undersample,LR_pred))

print('Accuracy: ', LR.score(X_test_undersample, y_test_undersample))
print('Recall: ', recall_score(y_test_undersample, LR_pred))


# In[113]:


from sklearn.metrics import confusion_matrix
LRcm = confusion_matrix(y_test_undersample, LR_pred)#得到混淆矩阵
print(LRcm)


# In[140]:


#简单绘制混淆矩阵
import matplotlib.pyplot as plt
plt.matshow(LRcm)
plt.title('LogisticRegression Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[115]:


#绘制ROC曲线
from sklearn.metrics import roc_curve, auc 
predictions = LR.predict_proba(X_test_undersample)
FPrate, recall, thresholds = roc_curve(y_test_undersample, predictions[:, 1])
roc_auc = auc(FPrate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(FPrate, recall, 'b', label = 'AUC = %0.2f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'g--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('Fall-out')
plt.ylabel('Recall')


# ### 尝试使用k-近邻算法

# In[116]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score

knn = KNeighborsClassifier()
knn.fit(X_train_undersample, y_train_undersample)
knn_pred = knn.predict(X_test_undersample)
print(classification_report(y_test_undersample, knn_pred))

print('Accuracy: ', knn.score(X_test_undersample, y_test_undersample))
print('Recall: ', recall_score(y_test_undersample, knn_pred))

knncm = confusion_matrix(y_test_undersample, knn_pred)
print(knncm)


# In[117]:


plt.matshow(knncm)
plt.title('KNeighbors Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# ### 尝试使用朴素贝叶斯

# In[118]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()

#由于朴素贝叶斯要求传递非负的值，将特征用Min-Max方式标准化
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(under_sample_X)
X_train_undersample1, X_test_undersample1, y_train_undersample1, y_test_undersample1 = train_test_split(                                                                                                       X_minmax, under_sample_y, test_size = 0.3, random_state = 2)

NB.fit(X_train_undersample1, y_train_undersample1)
NB_pred = NB.predict(X_test_undersample1)
print(classification_report(y_test_undersample1,NB_pred))

print('Accuracy: ', NB.score(X_test_undersample1, y_test_undersample1))
print('Recall: ', recall_score(y_test_undersample1, NB_pred))

NBcm = confusion_matrix(y_test_undersample1,NB_pred)
print(NBcm)
#效果明显不如逻辑回归和k-近邻


# ### 对比朴素贝叶斯和逻辑回归

# In[119]:


lr = LogisticRegression()#使用逻辑回归算法处理数量较大的数据集时，
nb = GaussianNB()

lr_scores = []
nb_scores = []

#展示不同样本大小下朴素贝叶斯和逻辑回归的拟合效果
train_sizes = range(10, len(X_train_undersample1), 25)
for i in train_sizes:
    X_slice1, _, y_slice1, _ = train_test_split(                                              X_train_undersample1, y_train_undersample1, train_size = i, stratify = y_train_undersample1, random_state=1)
    X_slice, _, y_slice, _ = train_test_split(                                              X_train_undersample, y_train_undersample, train_size = i, stratify = y_train_undersample, random_state=1)
    nb.fit(X_slice1, y_slice1)
    nb_scores.append(nb.score(X_test_undersample1, y_test_undersample1))
    lr.fit(X_slice, y_slice)
    lr_scores.append(lr.score(X_test_undersample, y_test_undersample))
    
plt.plot(train_sizes, nb_scores, label = 'Naive Bayes')
plt.plot(train_sizes, lr_scores, label = 'Logistic Regression', linestyle = '--')
plt.title("Naive Bayes and Logistic Regression Accuracies")
plt.xlabel("Number of training instances")
plt.ylabel("Test set accuracy")
plt.legend()


# ### 可以很明显的看到，在训练集数量逐渐增加后，朴素贝叶斯算法的准确率明显低于逻辑回归，符合朴素贝叶斯的特点。因此可以推知，朴素贝叶斯算法针对该数据集整体的拟合效果应该不如逻辑回归

# ### 尝试使用决策树

# In[120]:


from sklearn.tree import DecisionTreeClassifier

treeclf = DecisionTreeClassifier()
treeclf.fit(X_train_undersample, y_train_undersample)
tree_pred = treeclf.predict(X_test_undersample)
print(classification_report(y_test_undersample, tree_pred))

print('Accuracy: ', treeclf.score(X_test_undersample, y_test_undersample))
print('Recall: ', recall_score(y_test_undersample, tree_pred))

treecm = confusion_matrix(y_test_undersample, tree_pred)
print(treecm)


# ### 尝试使用支持向量机

# In[121]:


from sklearn import svm

svmclf = svm.SVC(C = 1.0, kernel = 'linear')
svmclf.fit(X_train_undersample, y_train_undersample)
svm_pred = svmclf.predict(X_test_undersample)
print(classification_report(y_test_undersample, svm_pred))

print('Accuracy: ', svmclf.score(X_test_undersample, y_test_undersample))
print('Recall: ', recall_score(y_test_undersample, svm_pred))

svmcm = confusion_matrix(y_test_undersample, svm_pred)
print(svmcm)


# ### 通过网格搜索优化已有的简单模型（逻辑回归、k-近邻、决策树）并进行比较

# In[122]:


from sklearn.model_selection import GridSearchCV

# 逻辑回归
LR_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_LR = GridSearchCV(LogisticRegression(), LR_params)
grid_LR.fit(X_train_undersample, y_train_undersample)
# 自动获得最佳参数的逻辑回归模型
log_reg = grid_LR.best_estimator_

# k-近邻
knn_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params)
grid_knn.fit(X_train_undersample, y_train_undersample)
knears_neighbors = grid_knn.best_estimator_

# 决策树
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train_undersample, y_train_undersample)
tree_clf = grid_tree.best_estimator_

#支持向量机
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(svm.SVC(), svc_params)
grid_svc.fit(X_train_undersample, y_train_undersample)
svc = grid_svc.best_estimator_


# ### 通过交叉验证观察是否出现过拟合并对四种算法的拟合效果进行比较

# In[123]:


from sklearn.model_selection import cross_val_score

log_reg_score = cross_val_score(log_reg, X_train_undersample, y_train_undersample, cv=10)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_train_undersample, y_train_undersample, cv=10)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train_undersample, y_train_undersample, cv=10)
                          
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train_undersample, y_train_undersample, cv=10)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')


# ### 确定下采样时逻辑回归的最佳参数C

# In[133]:


print(grid_LR.best_params_)


# ### 查看此时逻辑回归的拟合效果

# In[180]:


log_reg
log_reg_pred = log_reg.predict(X_test_undersample)
print(classification_report(y_test_undersample, log_reg_pred))

print('Accuracy: ', log_reg.score(X_test_undersample, y_test_undersample))
print('Recall: ', recall_score(y_test_undersample, log_reg_pred))
log_regcm = confusion_matrix(y_test_undersample, log_reg_pred)
print(log_regcm)


# ### 进一步通过设置阈值优化逻辑回归模型

# In[131]:


#自定义绘制混淆矩阵的函数
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Greens):
    #cm为数据，interpolation='nearest'使用最近邻插值，cmap颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #xticks(刻度下标，刻度标签)
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    #text()命令可以在任意的位置添加文字
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #自动紧凑布局
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[132]:


# 不同的阈值对结果的影响
import itertools
y_pred_undersample_proba = log_reg.predict_proba(X_test_undersample.values)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.figure(figsize=(10, 10))
j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1
    # 计算混淆矩阵
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    # 输出精度为小数点后两位
    np.set_printoptions(precision=2)
    print("Recall :", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    # 画出非标准化的混淆矩阵
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold>=%s' % i)
plt.show()


# ## 解决方案二：过采样

# In[124]:


from imblearn.over_sampling import SMOTE

oversampler = SMOTE(random_state=0)
X_oversample, y_oversample = oversampler.fit_sample(dfv_new, dfc)
print('过采样后，1的样本的个数为：', len(y_oversample[y_oversample == 1]))


# ### 采用逻辑回归模型并用网格搜索进行优化

# In[125]:


X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample = train_test_split(                                                                                                X_oversample, y_oversample, test_size = 0.3, random_state = 2)

grid_search = GridSearchCV(LogisticRegression(),  LR_params, cv=10) 
grid_search.fit(X_train_oversample, y_train_oversample)


# In[89]:


print(grid_search.best_params_)


# In[94]:


LR_os = grid_search.best_estimator_
LR_os_pred = LR_os.predict(X_test_oversample)
print(classification_report(y_test_oversample, LR_os_pred))

print('Accuracy: ', LR_os.score(X_test_oversample, y_test_oversample))
print('Recall: ', recall_score(y_test_oversample, LR_os_pred))
LR_oscm = confusion_matrix(y_test_oversample, LR_os_pred)
print(LR_oscm)


# ### 调整阈值进一步优化

# In[134]:


y_pred_oversample_proba = LR_os.predict_proba(X_test_oversample.values)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.figure(figsize=(10, 10))
j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_oversample_proba[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1
    # 计算混淆矩阵
    cnf_matrix = confusion_matrix(y_test_oversample, y_test_predictions_high_recall)
    # 输出精度为小数点后两位
    np.set_printoptions(precision=2)
    print("Recall :", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    # 画出非标准化的混淆矩阵
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold>=%s' % i)
plt.show()


# In[135]:


from sklearn.metrics import precision_recall_curve

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black'])
plt.figure(figsize=(12,7))

j = 1
for i,color in zip(thresholds,colors):
    y_test_predictions_high_recall = y_pred_oversample_proba[:, 1] > i #预测出来的概率值是否大于阈值  

    precision, recall, thresholds = precision_recall_curve(y_test_oversample, y_test_predictions_high_recall)
    area = auc(recall, precision)
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,
                 label='Threshold: %s, AUC=%0.5f' %(i , area))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")


# ## 结合现实考虑特征选取

# ### 通过对原始数据的直观察，可以发现金额一栏存在0的情况。不论是诈骗交易还是正常交易都出现了这种情况，结合现实考虑，这可以看作是一种异常情况。

# In[164]:


#将金额为0的样本剔除
df1 = df[df.Amount!=0]
df0 = df[df.Amount==0]
df0.head()


# ### 针对新的样本集重新取出特征和标签

# In[181]:


dfv1 = df1.iloc[:,1:29]
dfa1 = df1.loc[:,'Amount']
dfc1 = df1.loc[:,'Class']


# ### 选取15个相关度高的特征

# In[158]:


dfv1_new = dfv1.drop(DropList, axis = 1)
print(dfv1_new.shape)


# ### 将标准化后的金额和小时作为特征加入

# In[159]:


dfv1_new['normAmount'] = StandardScaler().fit_transform(dfa1.values.reshape(-1,1))
dfv1_new['normHour'] = StandardScaler().fit_transform(df1['Hour'].values.reshape(-1,1))
dfv1_new.head()


# ### 对剔除了金额为0的样本的样本集进行过采样

# In[177]:


oversampler1 = SMOTE(random_state=1)
X_oversample1, y_oversample1 = oversampler1.fit_sample(dfv1_new, dfc1)


# ### 划分后使用逻辑回归拟合并找出最佳参数C

# In[178]:


X_train_oversample1, X_test_oversample1, y_train_oversample1, y_test_oversample1 = train_test_split(                                                                                                X_oversample1, y_oversample1, test_size = 0.3, random_state = 4)

grid_search1 = GridSearchCV(LogisticRegression(),  LR_params, cv=10) 
grid_search1.fit(X_train_oversample1, y_train_oversample1)
print(grid_search1.best_params_)


# ### 观察此时的拟合效果

# In[179]:


LR_os1 = grid_search1.best_estimator_
LR_os1_pred = LR_os1.predict(X_test_oversample1)
print(classification_report(y_test_oversample1, LR_os1_pred))

print('Accuracy: ', LR_os1.score(X_test_oversample1, y_test_oversample1))
print('Recall: ', recall_score(y_test_oversample1, LR_os1_pred))
LR_oscm1 = confusion_matrix(y_test_oversample1, LR_os1_pred)
print(LR_oscm1)

