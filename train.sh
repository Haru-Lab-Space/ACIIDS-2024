# !/bin/bash
# pip install -r requirements.txt 
# cd /Thesis/source
# python main_train_up.py --device cuda --model RNN --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model GRU --epochs 100 --batch_size 32 --max_appearances 128 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.001 --patience 5
# python main_train_up.py --device cuda --model LSTM --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.001 --patience 5
# python main_train_up.py --device cuda --model Retain --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.001 --patience 5
# python main_train_up.py --device cuda --model Dipole --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.001 --patience 5
# python main_train_up.py --device cuda --model HiTANet --keys 'reverse_time' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model HiTANet1L --keys 'reverse_time' --epochs 100 --batch_size 32 --max_appearances 128 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model AHiTANet --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model TCNN --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model TCNN1L --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --max_appearances 128 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model TCNN5 --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model TCNNF --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 10
# python main_train_up.py --device cuda --model TCNNG --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 10
# python main_train_up.py --device cuda --model TCNNG5 --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 10
# python main_train_up.py --device cuda --model TCNNGS3 --keys 'reverse_time' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5 --learning_rate 0.0001 --patience 5
# python main_train_up.py --device cuda --model BHiTANet --keys 'reverse_time' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True --with_clip 0.5
# python main_train_up.py --device cuda --model GAHiTANet --keys 'reverse_time' 'gender' 'age' --epochs 100 --batch_size 32 --lossfunction CrossEntropyLoss --shuffle True
# python main_train_up.py --device cuda --model HiTANet --keys 'reverse_time' --epochs 100 --batch_size 32 --lossfunction BCEWithLogitsLoss --shuffle True


pip install -r requirements.txt 
python main_evaluate.py --device cuda --model TTT_Time_Ensemble_Multiscale_Decoder_Only_1 --category_type_map_state True --category_map_state True --keys 'reverse_time' 'age' --metric_name 'classification_report' --batch_size 64 --top_k 15