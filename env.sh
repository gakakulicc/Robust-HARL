#test envs
cons=($(seq 0 0.05 1.6))
for con in ${cons[@]}
do
    python train.py --exp env --algo mappo --env pursuit --noise $con --exp_name 6_agents/seed1/"$con"
    python train.py --exp env --algo hasac --env pursuit --noise $con --exp_name 4_agents/seed1/"$con"
    python train.py --exp env --algo hatd3 --env pursuit --noise $con --exp_name 4_agents/seed3/"$con"
    python train.py --exp env --algo happo --env pursuit --noise $con --exp_name 6_agents/seed1/"$con"
done