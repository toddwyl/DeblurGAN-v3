import os


def config_gru_fms(height, strides):
    gru_fms = [height]
    for i, s in enumerate(strides):
        gru_fms.append(gru_fms[i] // s)
    return gru_fms[1:]


def config_deconv_infer_height(height, strides):
    infer_shape = [height]
    for i, s in enumerate(strides[:-1]):
        infer_shape.append(infer_shape[i] // s)
    return infer_shape


def config_deconv_infer_width(width, strides):
    infer_shape = [width]
    for i, s in enumerate(strides[:-1]):
        infer_shape.append(infer_shape[i] // s)
    return infer_shape


# iterator
root_path = '/home/ices/yl_tmp/SRGAN/'
DATA_BASE_PATH = os.path.join("/extend", "sz17_data")
# REF_PATH = '/extend/sz17_data/radarPNG_expand/'
# REF_PATH = '/extend/2019_png'
REF_PATH = os.path.join(root_path, 'data/2019_png/')
TRAIN_DIR_CLIPS = os.path.join(DATA_BASE_PATH, "15-17_clips")
VALID_DIR_CLIPS = os.path.join(DATA_BASE_PATH, "18_clips")

BASE_PATH = os.path.join(root_path, "conv_gru_result_rain/gru_tf_data")
SAVE_PATH = os.path.join(BASE_PATH, "3_layer_7_5_3")
SAVE_MODEL = os.path.join(SAVE_PATH, "Save")
SAVE_VALID = os.path.join(SAVE_PATH, "Valid")
SAVE_TEST = os.path.join(SAVE_PATH, "Test")
SAVE_TXT = os.path.join(SAVE_PATH, "Txt")
SAVE_SUMMARY = os.path.join(SAVE_PATH, "Summary")
SAVE_METRIC = os.path.join(SAVE_PATH, "Metric")
DISPLAY_PATH = os.path.join(SAVE_PATH, "Display")
if not os.path.exists(SAVE_MODEL):
    os.makedirs(SAVE_MODEL)
if not os.path.exists(SAVE_VALID):
    os.makedirs(SAVE_VALID)

RAINY_TRAIN = ['201501010000', '201801010000']
RAINY_VALID = ['201904112006', '201904120000']
# RAINY_TEST =['201908252336', '201908270100']
RAINY_TEST = [
    # ['201909012336','201909030200'],
    # ['201909032336','201909050200'],
    ['201901010000', '201909150000'],
    # ['201909152336','201909170200'],
    #  ['201905222136','201905240200'],

    # ['201903052136','201903060200'],

    # ['201905312136','201906010300'],
    # ['201905291936','201905310200'],
    # ['201906012136','201906020200'],
    # ['201908312136','201909020200'],
    # ['201907312136','201908010200']
    #         # ['201903042236','201903080200'],
    # ['201904102236','201904130200'],
    # ['201904192236','201904210200'],
    # ['201904262236','201904280200'],
    # ['201905062236','201905080200'],
    # ['201905202236','201905220236'],
    # ['201905222236','201905240000'],
    # ['201905262236','201905300200'],
    # ['201905312236','201906020236'],
    # ['201906102236','201906120200'],
    # ['201906122236','201906140200'],
    # ['201906232236','201906260200'],
    # ['201907022236','201907040200'],
    # ['201907092236','201907110200'],
    # ['201907302236','201908010200'],
    # ['201907312236','201908020200'],
    # ['201908172236','201908190200'],
    # ['201908112236','201908130200'],
    # ['201908242236','201908270200'],
    # ['201908292236','201909010200']

]
# RAINY_TEST = ['201801010006', '201809180000']

# train
MAX_ITER = 5000000
SAVE_ITER = 2000
VALID_ITER = 2000

SUMMARY_ITER = 50

# project
DTYPE = "single"
NORMALIZE = False
FULL_H = 700
FULL_W = 900
MOVEMENT_THRESHOLD = 3000
H = 720
W = 912

BATCH_SIZE = 1
IN_CHANEL = 1

# encoder
# (kernel, kernel, in chanel, out chanel)

CONV_STRIDE = (3, 2, 2)
ENCODER_GRU_FILTER = (96, 192, 192)
ENCODER_GRU_INCHANEL = (8, 96, 192)
ENCODER_FEATURE_MAP_H = [240, 120, 60]
ENCODER_FEATURE_MAP_W = [304, 152, 76]
DECODER_FEATURE_MAP_H = [240, 120, 60]
DECODER_FEATURE_MAP_W = [304, 152, 76]
EN_INC = ENCODER_GRU_INCHANEL
CONV_KERNEL = ((7, 7, 1, EN_INC[0]),
               (5, 5, EN_INC[1], EN_INC[1]),
               (3, 3, EN_INC[2], EN_INC[2]),

               )
# IMAGESIZE_H=[352,176,88,44,22]
# 22
# decoder
# (kernel, kernel, out chanel, in chanel)
FINAL_CONV_CHANEL = 16
DECONV_STRIDE = (3, 2, 2)
DECODER_GRU_FILTER = (96, 192, 192)
DECODER_GRU_INCHANEL = (192, 192, 192)
DE_INC = DECODER_GRU_FILTER
DECONV_KERNEL = (
    # (5, 5, 1, 8),
    (7, 7, FINAL_CONV_CHANEL, DE_INC[0]),
    (5, 5, DE_INC[1], DE_INC[1]),
    (3, 3, DE_INC[2], DE_INC[2]),
)

# Encoder Forecaster

IN_SEQ = 5
# OUT_SEQ = 20
OUT_SEQ = 10
DISPLAY_IN_SEQ = 5

LR = 0.0001

RESIDUAL = False
SEQUENCE_MODE = False

# RNN
I2H_KERNEL = [5, 3, 3]
H2H_KERNEL = [7, 5, 3]

# EVALUATION
ZR_a = 58.53
ZR_b = 1.56

EVALUATION_THRESHOLDS = (0, 15, 25, 35)

USE_BALANCED_LOSS = False
THRESHOLDS = [0.5, 2, 5, 10, 30]
BALANCING_WEIGHTS = [1, 1, 2, 5, 10, 30]

TEMPORAL_WEIGHT_TYPE = "same"
TEMPORAL_WEIGHT_UPPER = 5

# LOSS
L1_LAMBDA = 0.001
L2_LAMBDA = 1
GDL_LAMBDA = 0.001
# SSIM_LAMBDA = 0
SSIM_LAMBDA = 100

# PREDICTION
PREDICT_LENGTH = 10
PREDICTION_H = 900
PREDICTION_W = 900
# DISPLAY_IN_SEQ=5


if __name__ == '__main__':
    print(config_gru_fms(PREDICTION_H, CONV_STRIDE))
    print(config_deconv_infer_height(PREDICTION_H, DECONV_STRIDE))
