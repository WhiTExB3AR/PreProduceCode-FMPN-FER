import re

a="--mode",
"test",
"--data_root",
"datasets/JAFFE",
"--test_csv",
"testJA_ids_0.csv",
"--gpu_ids",
"0",
"--model",
"res_cls",
"--solver",
"res_cls",
"--batch_size",
"4",
"--load_model_dir",
"ckpts/JAFFE/res_cls/fold_0/211119_090032",
"--load_epoch",
"100"
a=a.strip()

b= re.split('\s+',a)
for i in b:
    print(f'"{i}",')