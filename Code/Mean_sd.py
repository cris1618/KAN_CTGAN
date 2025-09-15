import numpy as np
import pandas as pd

# Adult dataset
adult_res = pd.DataFrame({
    "first_iteration" : [0.7978, 0.7367, 0.7661, 0.6491, 0.7889, 0.7073, 0.8179, 0.6929],
    "second_iteration" : [0.8202, 0.7449, 0.7980, 0.6622, 0.7738, 0.7448, 0.8195, 0.7241],
    "third_iteration": [0.7949, 0.7433, 0.7824, 0.7013, 0.7809, 0.7503, 0.7913, 0.6975],
    "fourth_iteration": [0.7657, 0.7421, 0.7322, 0.6833, 0.7957, 0.7774, 0.8171, 0.7049],
    "fifth_iteration": [0.7953, 0.7363, 0.7762, 0.6847, 0.7634, 0.7777, 0.8091, 0.7096],
    "sixth_iteration": [0.7620, 0.7564, 0.7918, 0.6697, 0.7870, 0.7286, 0.7740, 0.7249],
    "seventh_iteration": [0.7509, 0.7499, 0.7767, 0.6853, 0.7828, 0.7378, 0.7685, 0.7012],
    "eight_iteration": [0.8075, 0.7378, 0.7664, 0.6733, 0.7700, 0.6824, 0.8042, 0.7124],
    "ninth_iteration": [0.7917, 0.7510, 0.7866, 0.6528, 0.7857, 0.7470, 0.8259, 0.7151],
    "tenth_iteration": [0.7890, 0.7451, 0.7916, 0.6849, 0.7927, 0.7507, 0.8155, 0.6973]

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
    "fourth_iteration": [0.4227, 0.3814, 0.4266, 0.3829, 0.4399, 0.4119, 0.4449, 0.3720],
    "fifth_iteration": [0.4151, 0.3823, 0.4265, 0.3861, 0.4309, 0.4092, 0.4544, 0.3748],
    "sixth_iteration": [0.4408, 0.3862, 0.4120, 0.3881, 0.4064, 0.4138, 0.4231, 0.3782],
    "seventh_iteration": [0.3886, 0.3938, 0.4205, 0.3845, 0.4324, 0.4057, 0.4253, 0.3810],
    "eight_iteration": [0.4205, 0.3932, 0.4401, 0.3857, 0.4005, 0.4078, 0.4432, 0.3818],
    "ninth_iteration": [0.4413, 0.3792, 0.4209, 0.3810, 0.4364, 0.4058, 0.4441, 0.3793],
    "tenth_iteration": [0.4220, 0.3852, 0.4376, 0.3835, 0.4427, 0.4202, 0.4461, 0.3755]
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
    "fourth_iteration": [0.6816, 0.7833, 0.7203, 0.6570, 0.7186, 0.6800, 0.7396, 0.6785],
    "fifth_iteration": [0.6822, 0.7869, 0.7142, 0.6906, 0.7351, 0.6759, 0.6579, 0.6980],
    "sixth_iteration": [0.6422, 0.7924, 0.6962, 0.6662, 0.6888, 0.6612, 0.7213, 0.7687],
    "seventh_iteration": [0.6287, 0.7794, 0.6985, 0.6690, 0.7267, 0.6772, 0.6868, 0.7496],
    "eight_iteration": [0.6733, 0.7892, 0.7077, 0.7043, 0.6917, 0.6608, 0.6649, 0.7601],
    "ninth_iteration": [0.6289, 0.7886, 0.7125, 0.6856, 0.6890, 0.6900, 0.6749, 0.6794],
    "tenth_iteration": [0.6527, 0.7852, 0.7280, 0.6696, 0.7195, 0.7205, 0.7129, 0.7422]

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
    "fourth_iteration": [0.8960, 0.8713, 0.8850, 0.8095, 0.8870, 0.7998, 0.9028, 0.8043],
    "fifth_iteration": [0.8933, 0.8625, 0.9040, 0.8031, 0.8741, 0.8223, 0.9159, 0.7939],
    "sixth_iteration": [0.9028, 0.8953, 0.8809, 0.8321, 0.8681, 0.8413, 0.8970, 0.8008],
    "seventh_iteration": [0.8524, 0.8728, 0.9018, 0.8373, 0.8694, 0.8529, 0.9114, 0.8239],
    "eight_iteration": [0.8555, 0.8708, 0.8833, 0.7973, 0.8983, 0.8318, 0.8701, 0.8067],
    "ninth_iteration": [0.8674, 0.8607, 0.9150, 0.8112, 0.8561, 0.8289, 0.9007, 0.8001],
    "tenth_iteration": [0.8778, 0.8739, 0.8832, 0.7984, 0.9077, 0.8211, 0.8852, 0.8351]
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
    "fourth_iteration": [0.7103, 0.6236, 0.7325, 0.4888, 0.7368, 0.6828, 0.7730, 0.6137],
    "fifth_iteration": [0.7699, 0.6245, 0.7482, 0.4809, 0.7200, 0.6539, 0.7252, 0.5920],
    "sixth_iteration": [0.7238, 0.6522, 0.7277, 0.4784, 0.7336, 0.6832, 0.7208, 0.6003],
    "seventh_iteration": [0.6932, 0.6278, 0.7395, 0.4855, 0.7127, 0.7360, 0.7423, 0.6130],
    "eight_iteration": [0.7059, 0.6387, 0.7329, 0.5700, 0.7059, 0.6665, 0.7473, 0.5901],
    "ninth_iteration": [0.7007, 0.6385, 0.7326, 0.4717, 0.7300, 0.6606, 0.7427, 0.5965],
    "tenth_iteration": [0.7284, 0.6115, 0.7361, 0.4819, 0.7386, 0.6862, 0.7686, 0.5867]
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
    "fourth_iteration": [0.6575, 0.6558, 0.7126, 0.6610, 0.7086, 0.7210, 0.6555, 0.5902],
    "fifth_iteration": [0.6236, 0.6740, 0.7269, 0.6681, 0.7034, 0.6380, 0.6348, 0.6038],
    "sixth_iteration": [0.6694, 0.6470, 0.6698, 0.6573, 0.7063, 0.7065, 0.6325, 0.5969],
    "seventh_iteration": [0.6526, 0.6748, 0.7065, 0.6242, 0.7076, 0.7222, 0.5832, 0.6039],
    "eight_iteration": [0.6105, 0.6470, 0.7133, 0.6158, 0.6761, 0.7000, 0.6681, 0.6044],
    "ninth_iteration": [0.6153, 0.6418, 0.6762, 0.6018, 0.7067, 0.6223, 0.6188, 0.6103],
    "tenth_iteration": [0.6599, 0.6680, 0.6728, 0.6067, 0.7289, 0.7054, 0.6462, 0.5992]
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
    "fourth_iteration": [0.8955, 0.8700, 0.8889, 0.8503, 0.8711, 0.8792, 0.8984, 0.8525],
    "fifth_iteration": [0.8915, 0.8693, 0.8965, 0.8483, 0.8948, 0.8758, 0.8991, 0.8560],
    "sixth_iteration": [0.9062, 0.8743, 0.9026, 0.8689, 0.8962, 0.9042, 0.8946, 0.8567],
    "seventh_iteration": [0.9010, 0.8715, 0.8921, 0.8537, 0.8886, 0.8964, 0.9242, 0.8468],
    "eight_iteration": [0.8764, 0.8737, 0.8891, 0.8563, 0.8942, 0.8942, 0.9054, 0.8490],
    "ninth_iteration": [0.8786, 0.8816, 0.8932, 0.8605, 0.8990, 0.8899, 0.9090, 0.8465],
    "tenth_iteration": [0.8940, 0.8682, 0.8978, 0.8631, 0.8973, 0.8489, 0.9052, 0.8429]
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
    "fourth_iteration": [0.8355, 0.9106, 0.8534, 0.8962, 0.8411, 0.8210, 0.8648, 0.8836],
    "fifth_iteration": [0.8454, 0.9088, 0.8476, 0.8893, 0.8483, 0.8270, 0.8564, 0.8875],
    "sixth_iteration": [0.8517, 0.9113, 0.8251, 0.9056, 0.8404, 0.8160, 0.8564, 0.8867],
    "seventh_iteration": [0.8098, 0.9090, 0.8271, 0.8955, 0.8608, 0.8179, 0.8616, 0.8988],
    "eight_iteration": [0.8558, 0.9084, 0.8618, 0.9051, 0.8343, 0.8001, 0.8609, 0.8888],
    "ninth_iteration": [0.8307, 0.9070, 0.8000, 0.8893, 0.8143, 0.7891, 0.8486, 0.8963],
    "tenth_iteration": [0.8589, 0.9108, 0.8438, 0.8952, 0.8149, 0.7874, 0.8644, 0.8963]
}, 
    index=["CTGAN", "TVAE", "KAN_CTGAN", "KAN_TVAE", "HYBRID_KAN_CTGAN", "Disc_KAN_CTGAN", "Gen_KAN_CTGAN", "HYBRID_KAN_TVAE"]
)

covtype_res["Mean Quality Score"] = covtype_res.mean(axis=1)
covtype_res["STD Quality Score"] = covtype_res.std(axis=1)
print(covtype_res[["Mean Quality Score", "STD Quality Score"]])