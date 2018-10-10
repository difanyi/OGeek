import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def train_and_predict(X_train, y_train, X_val, y_val, X_test, **kwargs):
    """
        模型训练和预测
    Args:
        X_train:
        y_train:
        X_val:
        y_val:
        X_test:
        kwargs:
    Returns:

    """
    bst = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        slient=False,
        objective='binary:logistic',
        nthread=-1,
        seed=42
    )
    bst.fit(X_train, y_train)
    y_pre = bst.predict(X_val)
    print('验证集F1：%f' % f1_score(y_val,y_pre))
    y_test = bst.predict(X_test)
    # 保存预测结果 待提交
    np.savetxt('./output/submit.csv', y_test, delimiter=',')



if __name__ == '__main__':
    data = np.load('./output/data_process.npz')
    train_and_predict(**data)
