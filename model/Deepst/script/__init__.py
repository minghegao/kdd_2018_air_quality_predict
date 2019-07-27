'''
启动运行脚本，执行整个项目过程


1. 自动下载数据下载并保存(定时人物)
将新增数据下载到对应文件夹的 csv 表中

2. 数据预处理
基于上述下载的数据，进行数据预处理，并将生成的中间数据保存在 csv 表格中

aq_data_preprocess(city='bj')
print("Finished Beijing aq data preprocess.")
aq_data_preprocess(city='ld')
print("Finished London aq data preprocess.")
meo_data_preprocess(city='bj')
print("Finished Beijing meo data preprocess.")
meo_data_preprocess(city='ld')
print("Finished London meo data preprocess.")

3. 训练集验证集划分
train_dev_set_split(city="bj")
train_dev_set_split(city="ld")

4. 训练模型
5. 模型融合
6. 使用融合的模型预测

'''
