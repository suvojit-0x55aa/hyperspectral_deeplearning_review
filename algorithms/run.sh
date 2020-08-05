
python svm.py --dataset IP --tr_percent 0.10 --repeat 3
python svm.py --dataset UP --tr_percent 0.10 --repeat 3
python svm.py --dataset KSC --tr_percent 0.10 --repeat 3

python rf.py --dataset IP --tr_percent 0.10 --repeat 3
python rf.py --dataset UP --tr_percent 0.10 --repeat 3
python rf.py --dataset KSC --tr_percent 0.10 --repeat 3

python mlr.py --dataset IP --tr_percent 0.10 --repeat 3
python mlr.py --dataset UP --tr_percent 0.10 --repeat 3
python mlr.py --dataset KSC --tr_percent 0.10 --repeat 3

python recurrent.py --dataset IP --tr_percent 0.10 --repeat 3 --type_recurrent GRU
python recurrent.py --dataset UP --tr_percent 0.10 --repeat 3 --type_recurrent GRU
python recurrent.py --dataset KSC --tr_percent 0.10 --repeat 3 --type_recurrent GRU

python recurrent.py --dataset IP --tr_percent 0.10 --repeat 3 --type_recurrent LSTM
python recurrent.py --dataset UP --tr_percent 0.10 --repeat 3 --type_recurrent LSTM
python recurrent.py --dataset KSC --tr_percent 0.10 --repeat 3 --type_recurrent LSTM