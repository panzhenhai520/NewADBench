##  Papers 
Official code and data repository of [_**ADBench**: Anomaly Detection Benchmark_](https://arxiv.org/abs/2206.09426) (NeurIPS 2022).

Code URL:  https://github.com/Minqi824/ADBench?tab=readme-ov-file#readme

ADBench is designed as a module, with the calling method being：
```python
   from adbench.run import RunPipeline
   pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode=None, noise_type='irrelevant_features')
   results = pipeline.run()
```  
This code is in My_Anomaly.py ,can be run directly.
****


### Algorithms

The Figure below shows the algorithms (14 unsupervised, 7 semi-supervised, and 9 supervised algorithms) in ADBench.

![Algorithms](figs/Algorithms.png)

#### 新增:
#### Pytorch_LSTM, Pytorch_GRU, Keras_LSTM ，SVM_model（不同参数）,RandomForeast_model（不同参数）
----

### 环境:
Python3.7.16, Pytorch 1.13, Pytorch-cuda 11.7 Tensorflow2.11, Keras 2.11.0,Numpy 1.21.5 ,Pandas1.3.5 ,Matplotlib3.53 Tqdm4.66.2

### 修改说明：
newADBench 是PolyU COMP5121 的一个实验项目，我们对ADBench论文代码做了如下改动：

一、为尽快呈现实验结果，只加载dataset_list_classical类的数据，不加载dataset_list_cv, dataset_list_nlp 类型数据，且对classical数据进行筛选，只选择了以下三个数据集进行实验：

| Number | Data | # Samples | # Features | # Anomaly | % Anomaly | Category |
|:--:|:---:|:---------:|:----------:|:---------:|:---------:|:---:|
|3| backdoor|   95329   |    196     |   2329    |   2.44    | Network|
|9|census|  299285   |    500     |   18568   |   6.20    | Sociology|
|25| musk                                 |   3062    |    166     |    97     |   3.17    | Chemistry   |      

只定义一种irrelevant_features噪声类型，加入0.1的噪声，使原数据集增加10%的无关联特征进行干扰。

二、修改baseline包中的Supervised模块：

2.1 新定义一个模型工厂类：Class ModelFactory, 并定义get_model 方法，根据 model_name 分别创建 Pytorch_LSTM, Pytorch_GRU,keras_lstm_model、SVM_model、RandomForest_model 这五种我们自定义的模型，为了和ADBench 中原来的SVM,和RF两种算法对比， 我们自己定义的SVM_model 和 RandomForest_model 选择了不同的参数.

2.2 修改supervised 类：定义pytorch模型需要传入张量, 为了保持原有程序代码的结构，supervised需要新增输入参数Pdata，另外，为了在model_dict 中用ModelFactory创建模型，还需要使用lambda 匿名函数定义ModelFactory闭包，用于动态创建模型实例。如下所示，字典中lambda开头的算法就是新增的算法:
```python
        self.model_dict = {'LR':LogisticRegression,
                           'NB':GaussianNB,
                           'SVM':SVC,
                           'MLP':MLPClassifier,
                           'RF':RandomForestClassifier,
                           'LGB':lgb.LGBMClassifier,
                           'XGB':xgb.XGBClassifier,
                           'CatB':CatBoostClassifier,
                           'Pytorch_LSTM': lambda: ModelFactory(self.model_name, 'LSTM', self.epochs, self.PData).get_model(),
                           'Pytorch_GRU': lambda: ModelFactory(self.model_name, 'GRU', self.epochs, self.PData).get_model(),
                           'keras_lstm_model': lambda: ModelFactory(self.model_name, None, self.epochs, self.PData).get_model(),
                           'SVM_model': lambda: ModelFactory(self.model_name, None, self.epochs, self.PData).get_model(),
                           'RandomForest_model': lambda: ModelFactory(self.model_name, None, self.epochs, self.PData).get_model()
                           }
 ```
2.3 在supervised 类中新增了五种自定义模型对应的训练方法 model_flt() 和模型评估方法 model_performance()

三、修改ADBench包中的 run 模块：

3.1 为Pytorch 模型增加检测GPU设备的方法 get_pdevice() 

3.2 为Pytorch 模型增加张量处理的类：class PytrochData(object)

3.3 修改 RunPipeline 类的初始化方法：当parallel=supervis模式时，在model_dict 列表中新增五种自定义的模型的名称

3.4 修改 RunPipeline 类的run方法，在噪音数据产生之后，实例化PytrochData类 =PData,用于pytorch模型创建时的输入，并且在通过model_dict 字典进行循环时，通过model_name 判断，新增一个分支，用于创建我们自定义的模型，然后调用model_flt方法训练，调用model_performance方法评估。如下所示: # new model added 部分就是新增模型及训练，而# fit and test model 部分是原来的模型：
```python
        
        if self.model_name in ['Pytorch_LSTM','Pytorch_GRU','keras_lstm_model','SVM_model','RandomForest_model']:
            # new model added
            self.clf = self.clf(seed=self.seed, model_name=self.model_name, PData= PData)
            self.clf.model_flt()
            metrics=self.clf.model_performance()
            time_fit=self.clf.time_fit
            time_inference=self.clf.time_inference
        else:
            # fit and test model
            time_fit, time_inference, metrics = self.model_fit()
            results.append([params, model_name, metrics, time_fit, time_inference])
            print(f'Current experiment parameters: {params}, model: {model_name}, metrics: {metrics}, '
            f'fitting time: {time_fit}, inference time: {time_inference}')

```


经过上述修改，newADBench项目在ADBench代码基础上，保持原有结构和输出方式，实现了新增5种算法和原论文算法对比的实验要求。
### 总结：
1. ADBench代码完成机器学习后，会在result文件夹下生成四个csv文件，呈现3个数据集受到10%无关特征干扰的情况下10种算法的性能对比表，newADBench保持了相同的输出方式，把算法增加到15种.

2. 多种算法模型在不同的数据集上的验证结果表明，并不存在能广泛适用于各种数据集的通用异常检测模型。针对不同类型的数据集应选择适应能力好的算法，而newADBench可以作为选择异常检测算法时的工具。

3. SVM和随机森林虽然比较传统，但从几个数据集的测试效果看，其性能并不比采用神经网络构造的模型差.如果考虑训练时间的因素，甚至可以说SVM和随机森林算法在全监督的数据集上表现更好，这充分说明了数学算法的威力。
