# import numpy as np
#
#
# data_eatv = np.load('data/training/eatvs_npy.npy').reshape(-1, 6)
# data_without_mutation_eatv = np.load('data_without_mutation/training/eatvs_npy.npy').reshape(-1, 6)
# print(np.mean(abs(data_eatv), axis=0))
# # [1.57013596 0.46125459 1.57638876 0.27172551 0.27651924 0.41213462]
# print(np.mean(abs(data_without_mutation_eatv), axis=0))
# # [1.28297623 0.42164127 1.27314744 0.27559419 0.26780788 0.40862213]
# print(np.max(abs(data_eatv), axis=0))
# # [3.14159001 1.57045509 3.14159248 0.94854384 0.94689996 1.03126825]
# print(np.max(abs(data_without_mutation_eatv), axis=0))
# # [3.14154749 1.56238158 3.14158191 0.9425432  0.94796457 1.03054962]
#
# print('\n\n')
#
# data_jas = np.load('data/training/jas_npy.npy').reshape(-1, 6)
# data_without_mutation_jas = np.load('data_without_mutation/training/jas_npy.npy').reshape(-1, 6)
# print(np.mean(abs(data_jas), axis=0))
# # [1.51692191 1.51686736 1.20786796 1.5178439  1.50918103 1.51198143]
# print(np.mean(abs(data_without_mutation_jas), axis=0))
# # [1.5643131  1.55930856 1.22827855 1.557482   1.5671949  1.5568704 ]
# print(np.max(abs(data_jas), axis=0))
# # [3.14159145 3.14159202 2.49999771 3.141576   3.14159264 3.14158191]
# print(np.max(abs(data_without_mutation_jas), axis=0))
# # [3.1415922  3.14158077 2.49997476 3.14159189 3.14154142 3.14156372]
