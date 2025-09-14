import numpy as np
import pandas as pd

# Adult dataset
adult_res = pd.DataFrame({
    "first_iteration" : [0.7978, 0.7367, 0.7661, 0.6491, 0.7889, 0.7073, 0.8179, 0.6929],
    "second_iteration" : [0.8202, 0.7449, 0.7980, 0.6622, 0.7738, 0.7448, 0.8195, 0.7241],
    "third_iteration": [0.7949, 0.7433, 0.7824, 0.7013, 0.7809, 0.7503, 0.7913, 0.6975],
    "fourth_iteration": [0.7657, 0.7421, 0.7322, 0.6833, 0.7957, 0.7774, 0.8171, 0.7049]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

adult_res["Mean Quality Score"] = adult_res.mean(axis=1)
adult_res["STD Quality Score"] = adult_res.std(axis=1)
print(adult_res[["Mean Quality Score", "STD Quality Score"]])


# Alarm dataset
alarm_res = pd.DataFrame({
    "first_iteration" : [0.4201, 0.3950, 0.4291, 0.3822, 0.4553, 0.4100, 0.4511, 0.3790],
    "second_iteration" : [0.4199, 0.3773, 0.4238, 0.3841, 0.4547, 0.4055, 0.4516, 0.3750],
    "third_iteration": [0.4370, 0.3911, 0.4153, 0.3886, 0.4253, 0.3905, 0.4353, 0.3728],
    "fourth_iteration": [0.4227, 0.3814, 0.4266, 0.3829, 0.4399, 0.4119, 0.4449, 0.3720]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

alarm_res["Mean Quality Score"] = alarm_res.mean(axis=1)
alarm_res["STD Quality Score"] = alarm_res.std(axis=1)
print(alarm_res[["Mean Quality Score", "STD Quality Score"]])

# Census dataset
census_res = pd.DataFrame({
    "first_iteration" : [0.6428, 0.7829, 0.7169, 0.7008, 0.6994, 0.7058, 0.7516, 0.7409],
    "second_iteration" : [0.7008, 0.7884, 0.7238, 0.6821, 0.7204, 0.6856, 0.7119, 0.7036],
    "third_iteration" : [0.6316, 0.7938, 0.7328, 0.6890, 0.7140, 0.6883, 0.6998, 0.7378],
    "fourth_iteration": [0.6816, 0.7833, 0.7203, 0.6570, 0.7186, 0.6800, 0.7396, 0.6785]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

census_res["Mean Quality Score"] = census_res.mean(axis=1)
census_res["STD Quality Score"] = census_res.std(axis=1)
print(census_res[["Mean Quality Score", "STD Quality Score"]])

# Child dataset
child_res = pd.DataFrame({
    "first_iteration" : [0.8728, 0.8576, 0.9157, 0.7954, 0.8652, 0.8746, 0.8971, 0.8243],
    "second_iteration" : [0.8784, 0.8586, 0.8865, 0.8346, 0.8834, 0.8473, 0.9337, 0.8140],
    "third_iteration" : [0.9053, 0.8822, 0.8852, 0.8189, 0.8931, 0.8785, 0.9074, 0.8082],
    "fourth_iteration": [0.8960, 0.8713, 0.8850, 0.8095, 0.8870, 0.7998, 0.9028, 0.8043]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

child_res["Mean Quality Score"] = child_res.mean(axis=1)
child_res["STD Quality Score"] = child_res.std(axis=1)
print(child_res[["Mean Quality Score", "STD Quality Score"]])

# Insurance dataset
insurance_res = pd.DataFrame({
    "first_iteration" : [0.7076, 0.6340, 0.7569, 0.5011, 0.6793, 0.7034, 0.7837, 0.5905],
    "second_iteration" : [0.7410, 0.6328, 0.7093, 0.4786, 0.7180, 0.6774, 0.7052, 0.5960],
    "third_iteration" : [0.7035, 0.6298, 0.7057, 0.4827, 0.7282, 0.6626, 0.7392, 0.5799],
    "fourth_iteration": [0.7103, 0.6236, 0.7325, 0.4888, 0.7368, 0.6828, 0.7730, 0.6137]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

insurance_res["Mean Quality Score"] = insurance_res.mean(axis=1)
insurance_res["STD Quality Score"] = insurance_res.std(axis=1)
print(insurance_res[["Mean Quality Score", "STD Quality Score"]])

# Intrusion dataset
intrusion_res = pd.DataFrame({
    "first_iteration" : [0.6403, 0.6571, 0.6603, 0.6480, 0.7075, 0.6856, 0.6945, 0.6090],
    "second_iteration" : [0.6463, 0.6765, 0.7168, 0.5719, 0.6846, 0.6729, 0.6412, 0.6021],
    "third_iteration" : [0.6487, 0.6594, 0.6717, 0.6492, 0.7223, 0.6450, 0.6692, 0.6009],
    "fourth_iteration": [0.6575, 0.6558, 0.7126, 0.6610, 0.7086, 0.7210, 0.6555, 0.5902]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

intrusion_res["Mean Quality Score"] = intrusion_res.mean(axis=1)
intrusion_res["STD Quality Score"] = intrusion_res.std(axis=1)
print(intrusion_res[["Mean Quality Score", "STD Quality Score"]])

# News dataset
news_res = pd.DataFrame({
    "first_iteration" : [0.8856, 0.8712, 0.8919, 0.8610, 0.8848, 0.8935, 0.8999, 0.8487],
    "second_iteration" : [0.8915, 0.8689, 0.8860, 0.8664, 0.8740, 0.8523, 0.8975, 0.8527],
    "third_iteration" : [0.9141, 0.8658, 0.8874, 0.8662, 0.8831, 0.8892, 0.8852, 0.8658],
    "fourth_iteration": [0.8955, 0.8700, 0.8889, 0.8503, 0.8711, 0.8792, 0.8984, 0.8525]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

news_res["Mean Quality Score"] = news_res.mean(axis=1)
news_res["STD Quality Score"] = news_res.std(axis=1)
print(news_res[["Mean Quality Score", "STD Quality Score"]])

# Covtype dataset
covtype_res = pd.DataFrame({
    "first_iteration" : [0.8292, 0.9093, 0.8501, 0.9125, 0.8295, 0.8054, 0.8412, 0.9020],
    "second_iteration" : [0.7975, 0.9171, 0.8532, 0.8866, 0.8286, 0.8091, 0.8447, 0.8953],
    "third_iteration" : [0.8176, 0.9122, 0.8541, 0.8936, 0.8302, 0.8017, 0.8230, 0.8888],
    "fourth_iteration": [0.8355, 0.9106, 0.8534, 0.8962, 0.8411, 0.8210, 0.8648, 0.8836]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

covtype_res["Mean Quality Score"] = covtype_res.mean(axis=1)
covtype_res["STD Quality Score"] = covtype_res.std(axis=1)
print(covtype_res[["Mean Quality Score", "STD Quality Score"]])