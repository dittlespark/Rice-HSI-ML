
import csv
from LightGBM_reg import *
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import openpyxl as op
import warnings

warnings.filterwarnings('ignore')


def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    return rows


if __name__ == '__main__':
    meta_data = read_csv('E:/diaguoxin/20231123/metabolism.csv')
    spect_data = read_csv('E:/diaguoxin/20231123/spectralIndex.csv')
    meta_data = np.array(meta_data)
    spect_data = np.array(spect_data)

    meta_name = meta_data[0, 1:]
    meta_data = meta_data[1:, 1:]
    spect_data = spect_data[1:, 1:]

    meta_data = meta_data.astype(np.float)
    spect_data = spect_data.astype(np.float)

    scaler = StandardScaler()
    spect_data = scaler.fit_transform(spect_data)

    meta_number = range(len(meta_data[0]))
    good_number = 0

    wb = op.Workbook()  # 创建工作簿对象
    ws = wb.create_sheet('r2')

    for i in tqdm(meta_number):
        predict_meta = [x[i] for x in meta_data]
        # X_train, X_test, y_train, y_test = train_test_split(spect_data, predict_meta, test_size=0.2, random_state=2)
        # 设置五折交叉验证

        r2, rmse, cv_pred = lightgbm_reg_kfold(spect_data, predict_meta)

        result_last = []
        result_last.append(meta_name[i])
        result_last.extend(r2)
        result_last.extend(rmse)
        # 将cv_pred转换为list
        cv_pred = cv_pred.tolist()
        result_last.extend(cv_pred)

        # result_last.extend(a)
        ws.append(result_last)
        wb.save('E:/diaguoxin/20231123/lightgbm_search_5flod_31.xlsx')


