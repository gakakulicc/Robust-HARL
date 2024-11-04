cons=($(seq 0 0.2 1.0))
for con in ${cons[@]}
do
    python train.py --exp state --algo mappo --env pursuit --noise $con --exp_name exp_name &
    python train.py --exp state --algo hasac --env pursuit --noise $con --exp_name exp_name &
    python train.py --exp state --algo hatd3 --env pursuit --noise $con --exp_name exp_name &
    python train.py --exp state --algo happo --env pursuit --noise $con --exp_name exp_name &
    wait
done
