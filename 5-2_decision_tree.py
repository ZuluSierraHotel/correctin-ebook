#-*- coding: utf-8 -*-
#使用ID3决策树算法预测销量高低
import pandas as pd

#参数初始化
inputfile = '../data/sales_data.xls'
data = pd.read_excel(inputfile, index_col = u'序号') #导入数据

#数据是类别标签，要将它转化为数据
#用1来表示“好”，“是”，“高”3个属性，用-1来表示“坏”，“否”，“低”
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
dtc.fit(x, y) #训练模型

#导入相关函数，可视化决策树
#导出的结果是一个dot文件，需要安装Graghviz才能将它转为pdf或者png等格式
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
x = pd.DataFrame(x)
with open("tree2.dot", 'w') as f:
  f = export_graphviz(dtc, feature_names = x.columns, out_file = f)

import graphviz
with open("tree2.dot") as f:
    dot_graph = f.read()
dot=graphviz.Source(dot_graph)
dot.view()


