from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from adbench.myutils import Utils
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from keras.callbacks import LearningRateScheduler # Tensorflow 调整学习率的回调函数
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, LSTM

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.optimizers import Adam as KerasAdam

from ..datasets import data_generator
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import math

# keras LSTM 模型使用随机搜索来优化超参数，如果=cos,使用余弦退火算法动态重算学习率
tslstm='random'
random_search_result=None

# 为Tensorflow LSTM算法设置余弦退火学习率回调函数
def cosine_lr(epoch, lr_initial, lr_min, lr_max, n_cycles, epochs_per_cycle, total_epochs):
    epoch_in_cycle = epoch % epochs_per_cycle
    lr_range = lr_max - lr_min
    progress = epoch_in_cycle / epochs_per_cycle

    if epoch < total_epochs:
        return lr_min + 0.5 * lr_range * (1 + math.cos(math.pi * progress))
    else:
        return lr_min

class LSTMBinaryClassifier(nn.Module):
    #  LSTM model in Pytorch
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMBinaryClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, 50, batch_first=True)
        self.dense1 = nn.Linear(50, 10)
        self.dense2 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # 注意！该方法是隐式调用的，在nn.Module 的__call__ 中实现
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # 如果x含时间维度,需要 x = self.dense1(x[:, -1, :])
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

    def predict_score(self,x):
        #接收张量x 作为输入,返回正类的概率分数
        self.eval()
        with torch.no_grad():
            score=self(x)
            score=score.squeeze()
        return score

class GRUBinaryClassifier(nn.Module):
    # GRU model in Pytorch
    def __init__(self, input_size, hidden_size):
        super(GRUBinaryClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.fc(output)  # 直接传给全连接层
        output = self.sigmoid(output)
        return output

class TimeHistory(Callback):
    # 在keras_lstm_model模型中每个epoch训练时间的回调函数
    def on_train_begin(self, logs={}):
        self.train_times = []
        self.val_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.train_times.append(time.time() - self.epoch_start_time)
        if 'val_loss' in logs:
           self.val_times.append(time.time() - self.epoch_start_time ) ##- self.train_times[-1])

def remove_prefix(text, prefix="Pytorch_"):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def plot_multiple_roc_curves(data_pairs, titles, overall_title):
   # draw multiple ROC curves
    plt.figure()
    lw = 2

    for (y_data, y_score), title in zip(data_pairs, titles):
        # 计算每一类的ROC曲线和ROC面积
        fpr, tpr, threshold = roc_curve(y_data, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f'{title} ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(overall_title)
    plt.legend(loc="lower right")
    plt.show()

def get_device(self, gpu_specific=False, Showinfo=False):
    if gpu_specific:
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            if Showinfo != False:
                print(f'number of gpu: {n_gpu}')
                print(f'cuda name: {torch.cuda.get_device_name(0)}')
                print('GPU is on')
        else:
            if Showinfo != False:
                print('GPU is off')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device

class ModelFactory:

    def __init__(self, model_name, algorithm_type, epochs, PData):
        self.tslstm='random' #keras LSTM 模型使用随机搜索来优化超参数，如果=cos,使用余弦退火算法动态重算学习率
        self.device = PData.device
        self.model_name = model_name
        self.algorithm_type = algorithm_type
        self.epochs = epochs
        self.PData = PData
        self.y_test_pred = None
        self.y_test_pred_proba = None
        self.time_fit = None
        self.time_inference = None
        self.model = None
        self.y_train_pred_proba = None
        self.overall_title = None
        self.metrics = []
        self.params=[]
        if self.model_name.startswith("Pytorch_"):
            self.algorithm_type = remove_prefix(model_name)
            self.model_name='Pytorch_model'
        else:
            self.algorithm_type = "N"

    def build_Ts_LSTMmodel(self,optimizer='KerasAdam', units=32):
        if optimizer == 'KerasAdam':
            optimizer_instance = KerasAdam()
        elif optimizer == 'rmsprop':
            optimizer_instance = 'rmsprop'
        model = Sequential()
        model.add(
            LSTM(units=units, return_sequences=True, input_shape=(self.PData.X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer_instance, metrics=['accuracy'])
        return model

    def get_model(self):
        if self.model_name == 'Pytorch_model':
            X_train_tensor = self.PData.X_train_tensor.unsqueeze(1)
            if self.algorithm_type == 'LSTM':
                self.model = LSTMBinaryClassifier(X_train_tensor.shape[2], 32)
            elif self.algorithm_type == 'GRU':
                self.model = GRUBinaryClassifier(X_train_tensor.shape[2], 32)
            self.device = get_device(self,gpu_specific=True, Showinfo=False)

        elif self.model_name == 'keras_lstm_model':
            # 采用tensorflow 编写的LSTM 模型
            import tensorflow as tf
            if tf.config.experimental.list_physical_devices('GPU'):
                print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            else:
                print("GPU is not detected")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f'GPU {i}: Name: {gpu.name}, Type: {gpu.device_type}')
            self.device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
            time_callback = TimeHistory()
            with tf.device(self.device):
                if self.tslstm=='cos':
                    # 定义模型
                    lstm = Sequential()
                    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(self.PData.X_train.shape[1], 1)))
                    lstm.add(LSTM(50))
                    lstm.add(Dense(10, activation='relu'))
                    lstm.add(Dense(1, activation='sigmoid'))
                    lstm.compile(loss='binary_crossentropy', optimizer='KerasAdam', metrics=['acc'])
                elif self.tslstm=='random':
                    self.model = KerasClassifier(build_fn=self.build_Ts_LSTMmodel, verbose=1)
                    # 定义超参数网格
                    param_grid = {
                        'units': [32, 64, 128],
                        'optimizer': ['KerasAdam', 'rmsprop'],
                        'batch_size': [32, 64],
                        'epochs': [50, 100]
                    }
                    # 创建并运行随机搜索
                    random_search = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2)
                    global random_search_result
                    random_search_result = random_search.fit(self.PData.X_train, self.PData.y_train)

                    # 打印最佳参数和性能指标
                    print("Best Param: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))
                    # 使用最佳参数重新训练模型
                    model_params = random_search_result.best_params_.copy()
                    print('@@@@@@@@@@@@@@@@@')
                    print(random_search_result)

                    # 最佳参数在random_search_result字典，定义模型的时候不需要epochs和batch_size,训练时才需要
                    if 'epochs' in model_params:
                        del model_params['epochs']
                    if 'batch_size' in model_params:
                        del model_params['batch_size']
                    best_model = self.build_Ts_LSTMmodel(**model_params)
                    self.model=best_model

        elif self.model_name == 'SVM_model':
            # 支持向量机
            from sklearn.svm import SVC
            self.model = SVC(kernel='linear', probability=True)

        elif self.model_name == 'RandomForest_model':
            # 随机森林模型
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10)

        return self.model

class supervised():
    def __init__(self, seed:int=1, model_name:str=None,PData=None):
        self.seed = seed
        self.utils = Utils()
        self.PData = PData
        self.model_name = model_name
        self.device = get_device(self,gpu_specific=True, Showinfo=False)
        self.metrics = []
        self.params = []
        self.epochs= 100
        self.algorithm_type=remove_prefix(model_name, prefix="Pytorch_")

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

    def model_flt(self):
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
        results = []
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'train_time': [], 'val_time': []}
        print(f"以下是我们自定义的模型:{self.model_name} data, params")

        if self.model_name in ['Pytorch_LSTM','Pytorch_GRU']:
            import torch.optim.lr_scheduler as lr_scheduler  # 基于损失动态调整学习率的调度
            # X_train_tensor = self.PData.X_train_tensor.unsqueeze(1)
            self.model = self.model_dict[self.model_name]()
            # 加载训练集数据
            train_loader = DataLoader(dataset=self.PData.train_dataset, batch_size=64, shuffle=True)
            # 加载测试集数据
            test_loader = DataLoader(dataset=self.PData.test_dataset, batch_size=64, shuffle=False)
            self.model.to(self.device)
            # pytorch 通过优化器调参，设置学习率lr的初始值
            optimizer = Adam(self.model.parameters(), lr=0.001)
            # 损失函数
            criterion = nn.BCELoss()
            # 根据损失动态调整学习率
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.05, patience=10)
            # 训练开始之前先生成模型的结构图
            inputs, labels = next(iter(train_loader))
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            # 使用make_dot生成模型结构图
            from torchviz import make_dot
            file_name = f"model_{self.model_name}"  # 动态生成m模型图片文件名
            make_dot(loss, params=dict(self.model.named_parameters())).render(file_name, format="png")

            for epoch in range(self.epochs):
                start_time_epoch = time.time()
                self.model.train()
                train_losses, train_accuracies = [], []
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())
                    predictions = torch.round(outputs)
                    accuracy = torch.sum(predictions == labels).item() / labels.size(0)
                    train_accuracies.append(accuracy)
                end_time_train = time.time()
                time_fit = end_time_train - start_time_epoch

                self.model.eval()
                val_losses, val_accuracies = [], []
                start_time_val = time.time()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_losses.append(loss.item())
                        predictions = torch.round(outputs)
                        accuracy = (predictions == labels).float().mean().item()
                        val_accuracies.append(accuracy)
                end_time_val = time.time()
                time_inference = end_time_val - start_time_val

                val_loss_avg = sum(val_losses) / len(val_losses)
                val_accuracy_avg = sum(val_accuracies) / len(val_accuracies)

                history['loss'].append(np.mean(train_losses))
                history['acc'].append(np.mean(train_accuracies))
                history['val_loss'].append(np.mean(val_losses))
                history['val_acc'].append(np.mean(val_accuracies))
                history['train_time'].append(time_fit)
                history['val_time'].append(time_inference)

                print(f'Epoch {epoch + 1}, Loss: {np.mean(train_losses):.4f}, '
                      f'Acc: {np.mean(train_accuracies):.4f}, '
                      f'Val Loss: {np.mean(val_losses):.4f}, '
                      f'Val Acc: {np.mean(val_accuracies):.4f}, '
                      f'TrainTime: {time_fit:.4f}s, '
                      f'Val Time: {time_inference:.4f}s')

                # 根据损失动态调整学习率
                scheduler.step(val_loss_avg)

            total_train_time = sum(history['train_time'])
            total_val_time = sum(history['val_time'])
            self.time_fit = total_train_time
            self.time_inference = total_val_time

            print(f'Total training time: {total_train_time:.4f}s, Total validation time: {total_val_time:.4f}s')

            # 禁用梯度计算
            with torch.no_grad():
                logits_train = self.model(self.PData.X_train_tensor).squeeze()
                y_train_score = torch.sigmoid(logits_train)
                logits_test = self.model(self.PData.X_test_tensor).squeeze()
                y_test_score = torch.sigmoid(logits_test)
                self.y_train_pred_proba = torch.sigmoid(logits_train).cpu().numpy()
                self.y_test_pred_proba = torch.sigmoid(logits_test).cpu().numpy()
                predictions = self.model(self.PData.X_test_tensor)
                y_pred = torch.round(predictions)

                # 将存储在GPU上的张量转换为NumPy数组之前，先将张量移动回CPU内存
                y_pred_np = y_pred.cpu().numpy().flatten()
                self.y_test_pred = y_pred_np

                self.y_test_pred = y_pred.cpu().numpy().flatten()
                y_train_score_np = y_train_score.cpu().numpy()
                y_test_score_np = y_test_score.cpu().numpy()

            y_test_np = self.PData.y_test_tensor.cpu().numpy().flatten()
            precision = precision_score(y_test_np, self.y_test_pred)
            recall = recall_score(y_test_np, self.y_test_pred)
            f1 = f1_score(y_test_np, self.y_test_pred)

            print("precision: {:.2f}%".format(precision * 100))
            print("recall_score: {:.2f}%".format(recall * 100))
            print("F1 Score: {:.2f}%".format(f1 * 100))
            y_train_score_np = y_train_score.cpu().numpy()
            y_test_score_np = y_test_score.cpu().numpy()
            score_test = y_test_score

            # 确保 y_true、y_score是NumPy数组，并且已经从CUDA移动到CPU
            # y_true_np = data['y_test'].cpu().numpy() if isinstance(data['y_test'], torch.Tensor) else data['y_test']

            y_true_np = self.PData.y_test_tensor.cpu().numpy() if isinstance(self.PData.y_test_tensor,
                                                                             torch.Tensor) else self.PData.y_test
            y_score_np = score_test.cpu().numpy() if isinstance(score_test, torch.Tensor) else score_test

            # 调用metric函数，传入NumPy数组
            self.metrics = data_generator.metric(self,y_true=y_true_np, y_score=y_score_np, pos_label=1)
            results.append([self.params, self.model_name, self.metrics, total_train_time, total_val_time]                                                               8i8i8i8i8i8i8i` )
            #show_history(history)
            print(
                f"\nModel: {self.model_name}, AUC-ROC: {self.metrics['aucroc']}, AUC-PR: {self.metrics['aucpr']}")

        elif self.model_name == 'keras_lstm_model':
            time_callback = TimeHistory()
            self.model = self.model_dict[self.model_name]()
            from keras.utils import plot_model
            if tslstm=='cos':
                total_epochs = self.epochs
                epochs_per_cycle = total_epochs // 5 # 余弦退火学习率调整为设置为 4个周期
                lr_scheduler = LearningRateScheduler(
                       lambda epoch: cosine_lr(epoch, lr_initial=0.001, lr_min=0.001, lr_max=0.01, n_cycles=2,
                                                epochs_per_cycle=epochs_per_cycle, total_epochs=total_epochs),verbose=1)
                hist = self.model.fit(self.PData.X_train, self.PData.y_train,
                                     validation_data=(self.PData.X_test, self.PData.y_test),
                                     epochs=self.epochs, batch_size=32, callbacks=[time_callback,lr_scheduler])
                score = self.model.evaluate(self.PData.X_test, self.PData.y_test, batch_size=32)
            elif tslstm=='random':
                # 现在使用最佳参数和额外的回调函数来训练模型
                print('oooooooooooooooooo')
                print(random_search_result)
                if random_search_result:
                    hist = self.model.fit(self.PData.X_train, self.PData.y_train,
                                          epochs=random_search_result.best_params_['epochs'],
                                          batch_size=random_search_result.best_params_['batch_size'],
                                          verbose=1,
                                          callbacks=[time_callback],
                                          validation_data=(self.PData.X_test, self.PData.y_test))

                    score = self.model.evaluate(self.PData.X_test, self.PData.y_test,
                                               batch_size=random_search_result.best_params_['batch_size'])

            history = {
                'loss': hist.history['loss'],
                'val_loss': hist.history['val_loss'],
                'acc': hist.history['accuracy'],
                'val_acc': hist.history['val_accuracy'],
                'train_time': time_callback.train_times,
                'val_time': time_callback.val_times
            }

            total_train_time = sum(time_callback.train_times)
            total_val_time = sum(time_callback.val_times)

            self.PData.X_train = np.nan_to_num(self.PData.X_train)
            self.PData.X_test = np.nan_to_num(self.PData.X_test)

            y_train_score = self.model.predict(self.PData.X_train).ravel()
            self.y_train_pred_proba = y_train_score
            y_test_score = self.model.predict(self.PData.X_test).ravel()
            self.y_test_pred_proba = y_test_score
            self.y_test_pred = (y_test_score > 0.5).astype(int)

            self.metrics = data_generator.metric(self,self.PData.y_test, y_test_score, pos_label=1)
            results.append([self.params, 'keras_lstm', self.metrics, total_train_time, total_val_time])
            print('----- 生成模型结构图model.png -----')
            self.model.summary()
            plot_model(self.model, to_file='keras_LSTM_model.png')
            # show_history(history)
            print(f"Model: keras_LSTM, AUC-ROC: {self.metrics['aucroc']}, AUC-PR: {self.metrics['aucpr']}")
            self.time_fit = total_train_time
            self.time_inference = total_val_time

        elif self.model_name in ['RandomForest_model','SVM_model']:
            self.model = self.model_dict[self.model_name]()
            start_time = time.time()
            self.model.fit(self.PData.X_train, self.PData.y_train)
            end_time = time.time()
            self.time_fit = end_time - start_time
            start_time = time.time()
            self.y_test_pred = self.model.predict(self.PData.X_test)
            self.y_test_pred_proba = self.model.predict_proba(self.PData.X_test)[:, 1]
            end_time = time.time()
            self.time_inference = end_time - start_time

    def model_performance(self):
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
        from sklearn.metrics import average_precision_score

        test_accuracy = accuracy_score(self.PData.y_test, self.y_test_pred)
        test_precision = precision_score(self.PData.y_test, self.y_test_pred, zero_division=1)

        test_recall = recall_score(self.PData.y_test, self.y_test_pred)
        test_f1 = f1_score(self.PData.y_test, self.y_test_pred)

        if self.model_name == 'RandomForest_model' or self.model_name == 'SVM_model':
            self.y_train_pred_proba = self.model.predict_proba(self.PData.X_train)[:, 1]
            self.y_test_pred_proba = self.model.predict_proba(self.PData.X_test)[:, 1]
            train_fpr, train_tpr, _ = roc_curve(self.PData.y_train, self.y_train_pred_proba)
            train_aucroc = auc(train_fpr, train_tpr)
            train_aucpr = average_precision_score(self.PData.y_train, self.y_train_pred_proba)
        elif self.model_name in ['Pytorch_LSTM','Pytorch_GRU','keras_lstm_model']:
            train_fpr, train_tpr, _ = roc_curve(self.PData.y_train.ravel(), self.y_train_pred_proba)
            train_aucroc = auc(train_fpr, train_tpr)
            train_aucpr = average_precision_score(self.PData.y_train.ravel(), self.y_train_pred_proba)

        print(
            f"Test Accuracy: {test_accuracy:.4%}, Precision: {test_precision:.4%}, Recall: {test_recall:.4%}, F1 Score: {test_f1:.4%}")

        self.overall_title =self.model_name

        self.metrics = {'aucroc': train_aucroc, 'aucpr': train_aucpr}

        # plot_multiple_roc_curves(data_pairs=[(self.PData.y_train.ravel(), self.y_train_pred_proba.ravel()),
        #                                      (self.PData.y_test.ravel(), self.y_test_pred_proba.ravel())],
        #                          titles=['Training', 'Test'],
        #                          overall_title=self.overall_title)
        return self.metrics

    def fit(self, X_train, y_train, ratio=None):
        # 实例化模型
        if self.model_name == 'NB':
            self.model = self.model_dict[self.model_name]()
        elif self.model_name == 'SVM':
            self.model = self.model_dict[self.model_name](probability=True)
        else:
            self.model = self.model_dict[self.model_name](random_state=self.seed)

        # fitting
        self.model.fit(X_train, y_train)

        return self

    def predict_score(self, X):
        score = self.model.predict_proba(X)[:, 1]
        return score

