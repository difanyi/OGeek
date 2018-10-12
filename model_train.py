# encoding: utf-8
import numpy as np
# from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from dataset import *
from nn import *

# def train_and_predict(X_train, y_train, X_val, y_val, X_test, **kwargs):
#     """
#         模型训练和预测
#     Args:
#         X_train:
#         y_train:
#         X_val:
#         y_val:
#         X_test:
#         kwargs:
#     Returns:

#     """
#     bst = XGBClassifier(
#         max_depth=6,
#         learning_rate=0.1,
#         n_estimators=300,
#         slient=False,
#         objective='binary:linear',
#         nthread=-1,
#         seed=42
#     )
#     eval_set = [(X_train, y_train), (X_val, y_val)]
#     print('Training ...\n')
#     bst.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=my_f1_score, eval_set=eval_set, verbose=True)
#     results = bst.evals_result()
#     epochs = len(results['validation_0']['my-f1'])
#     x_axis = range(0, epochs)
#     fig, ax = plt.subplots()
#     ax.plot(x_axis, results['validation_0']['my-f1'], label='Train')
#     ax.plot(x_axis, results['validation_1']['my-f1'], label='Val')
#     ax.legend()
#     plt.ylabel('error')
#     plt.title('XGBoost Model Error')
#     plt.show()
#     print('Predicting ...\n')
#     y_pre = bst.predict(X_val)
#     print('验证集F1：%f' % f1_score(y_val,y_pre))
#     y_test = bst.predict(X_test)
#     # 保存预测结果 待提交
#     print('Saving ...\n')
#     np.savetxt('./output/submit.csv', y_test, delimiter=',')


def my_f1_score(preds, dtrain):
    labels = dtrain.get_label()
    y_bin = [1 if y_cont > 0.5 else 0 for y_cont in preds]
    return 'my-f1', 1 - f1_score(labels, y_bin)


# def train(X_train, y_train, X_val, y_val, X_test, **kwargs):
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,50,10), random_state=1)
#     clf.fit(X_train, y_train)
#     print('Predicting ...\n')
#     y_pre = clf.predict(X_val)
#     print('验证集F1：%f' % f1_score(y_val, y_pre))
#     y_test = clf.predict(X_test)
#     # 保存预测结果 待提交
#     print('Saving ...\n')
#     np.savetxt('./output/submit.csv', y_test, delimiter=',')


def MLP_Training(path):
    dataset = make_ogeek_provider(path,10000)
    data = np.load('./output/data_process.npz')
    model = MLP_Wrapper(10,100,622,2)
    model = model.float()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    
    #定义损失函数和优化器
    model.compile_optimizer('adam', 1e-3, 200, l2_reg = 0,
                          lr_decay=False, lr_decay_rate=0.9, lr_decay_min = None,
                          lr_decay_every = 1000,
                          )
    
    #训练
    print(type(dataset['train']))
    model.fit(dataset['train'], dataset['val'], print_every=100,val_every= 1000)

    #评估

    #预测
    y_pre = predict(data['X_val'])
    print('验证集F1：%f' % f1_score(y_val, y_pre))
    y_test = predict(data['X_test'])
    # 保存预测结果 待提交
    print('Saving ...\n')
    np.savetxt('./output/submit_MLE.csv', y_test, delimiter=',')

    
if __name__ == '__main__':
    path="'./output/data_process.npz'"
    # data = np.load('./output/data_process.npz')
    # print(data)
    MLP_Training(path)
