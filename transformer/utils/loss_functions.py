import numpy as np

def logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def logloss_prime(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_true-y_pred)/y_true.size

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


if __name__ == "__main__":
    Y = np.array([0,0,1,1])
    A = np.array([0,1,0,0])

    loss = logloss_prime(Y, A)
    print(loss.shape)