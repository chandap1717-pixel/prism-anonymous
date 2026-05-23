import numpy as np


def _masked_values(y_true, y_pred, null_val=0.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = y_true != null_val

    mask = mask.astype(np.float32)
    mask = mask / (mask.mean() + 1e-6)
    mask = np.nan_to_num(mask)

    error = y_pred - y_true
    return error, y_true, mask


def MAE(y_true, y_pred, null_val=0.0):
    error, _, mask = _masked_values(y_true, y_pred, null_val)
    return float(np.mean(np.nan_to_num(np.abs(error) * mask)))


def RMSE(y_true, y_pred, null_val=0.0):
    error, _, mask = _masked_values(y_true, y_pred, null_val)
    return float(np.sqrt(np.mean(np.nan_to_num((error ** 2) * mask))))


def MAPE(y_true, y_pred, null_val=0.0):
    error, y_true, mask = _masked_values(y_true, y_pred, null_val)
    denom = np.where(np.abs(y_true) < 1e-5, np.nan, y_true)
    mape = np.abs(error / denom) * mask
    return float(np.mean(np.nan_to_num(mape)))


def RMSE_MAE_MAPE(y_true, y_pred, null_val=0.0):
    rmse = RMSE(y_true, y_pred, null_val)
    mae = MAE(y_true, y_pred, null_val)
    mape = MAPE(y_true, y_pred, null_val)
    return rmse, mae, mape
