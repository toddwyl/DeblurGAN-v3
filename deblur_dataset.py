import os
import shutil

deblur_path = r'../../conv_gru_result_rain/gru_tf_data/3_layer_7_5_3/Test/'
data_path = os.path.join(deblur_path, '112000-qpe-0_23-down')
train_path = r'Deblur/Train'
test_path = r'Deblur/Test'

# train_path = r'Deblur_slim/Train'
# test_path = r'Deblur_slim/Test'


date_dirs = os.listdir(data_path)
print(deblur_path)
train_full_path = os.path.join(deblur_path, train_path)
test_full_path = os.path.join(deblur_path, test_path)

out_train_path = os.path.join(train_full_path, 'sharp')
pred_train_path = os.path.join(train_full_path, 'blur')

out_test_path = os.path.join(test_full_path, 'sharp')
pred_test_path = os.path.join(test_full_path, 'blur')

main_paths = {train_full_path, test_full_path, out_train_path, pred_train_path, out_test_path, pred_test_path}

for path in main_paths:
    if not os.path.exists(path):
        os.makedirs(path)

num = len(date_dirs)
# num = 20
print(0.1 * num)
test_sample_num = int(0.1 * num)
train_num = 0
test_num = 0
for i in range(num):
    date_full_dir = os.path.join(data_path, date_dirs[i])
    out_dir = os.path.join(date_full_dir, 'out')
    out_img_list = os.listdir(out_dir)
    pred_hist_dir = os.path.join(date_full_dir, 'pred_hist')
    # pred_hist_img_list = os.listdir(pred_hist_dir)
    if num - i > test_sample_num:
        for img in out_img_list:
            out_img_path = os.path.join(out_dir, img)
            pred_img_path = os.path.join(pred_hist_dir, img)
            train_img_name = '{}_{}'.format(date_dirs[i], img)
            out_train_img = os.path.join(out_train_path, train_img_name)
            shutil.copy2(out_img_path, out_train_img)
            print("copy from {} to {}".format(out_img_path, out_train_img))
            pred_train_img = os.path.join(pred_train_path, train_img_name)
            shutil.copy2(pred_img_path, pred_train_img)
            print("copy from {} to {}".format(pred_img_path, pred_train_img))
            train_num += 1

    else:
        for img in out_img_list:
            out_img_path = os.path.join(out_dir, img)
            pred_img_path = os.path.join(pred_hist_dir, img)
            test_img_name = '{}_{}'.format(date_dirs[i], img)
            out_test_img = os.path.join(out_test_path, test_img_name)
            shutil.copy2(out_img_path, out_test_img)
            print("copy from {} to {}".format(out_img_path, out_test_img))
            pred_test_img = os.path.join(pred_test_path, test_img_name)
            shutil.copy2(pred_img_path, pred_test_img)
            print("copy from {} to {}".format(pred_img_path, pred_test_img))
            test_num += 1
print("train_num:", train_num)
print("test_num:", test_num)

# train_num: 35940
# test_num: 3990
