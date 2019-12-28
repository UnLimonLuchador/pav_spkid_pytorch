# PAV Speaker Identifier with Deep Neural Networks

Speaker recognition baseline for PAV subject in ETSETB UPC (Telecom BCN)

This program creates a Multilayer perceptron to classify speaker.


Se puede ejecutar directamente si la carpeta pav_spkid_pytorch está en el directorio PAV, y dentro de este directorio también se encuentra la carpeta P4 con todos los archivos

# EJEMPLOS DE COMANDOS USADOS
python3 train.py --save_path model_h30_25fram --hsize 30 --in_frames 25 --db_path ../P4/work/lpcc --tr_list_file cfg/all.train --va_list_file cfg/all.test --ext lpcc

python3 test.py --weights_ckpt model_h30_25fram/bestval_e19_weights.ckpt --train_cfg model_h30_25fram/train.opts --log_file logs/model_h30_25fram.log --db_path ../P4/work/lpcc --te_list_file cfg/all.test --ext lpcc

python3 verify --db_path ../P4/work/lpcc --te_list_file ../P4/lists/verif/all.test --candidates_list ../P4/lists/verif/all.test.candidates --weights_ckpt model_h30_25fram/bestval_e19_weights.ckpt --log_file logs/verif_test_h30_25f.log --train_cfg model_h30_25fram/train.opts --ext lpcc


# PARA EVALUAR CON EL SET DE TEST

python3 class_score.py --r logs/model_h30_25fram.log
python3 verify_score.py --r logs/verif_test_h30_25f.log


# COMANDOS USADOS PARA LA EVALUACION CIEGA FINAL

python3 test.py --weights_ckpt model_h30_25fram/bestval_e19_weights.ckpt --train_cfg model_h30_25fram/train.opts --log_file class_DNN.log --db_path ../P4/work/lpcc/final/spk_cls --te_list_file ..P4/lists/final/class.test --ext lpcc

python3 verify --db_path ../P4/work/lpcc/final --te_list_file ../P4/lists/final/verif.test --candidates_list ../P4/lists/final/verif.test.candidates --weights_ckpt model_h30_25fram/bestval_e19_weights.ckpt --log_file verif_DNN.log --train_cfg model_h30_25fram/train.opts --ext lpcc --blind
