#!/bin/bash

ARCHS=("mlp")
SEEDS=(20 42 150)
ACTOR_LAY=(9)
CRITIC_LAY=(2 3 4 5 6)
# ACTOR_LAY=(7)
# CRITIC_LAY=(6)
# ACTOR_LR=(0.001 0.0005 0.0001 0.00005)
# CRITIC_LR=(0.001 0.0005 0.0001 0.00005)
ACTOR_LR=(0.001)
CRITIC_LR=(0.001)
EPOCH_MAX=60



for arch in ${ARCHS[@]}
do
    for alay in ${ACTOR_LAY[@]}
    do
        for clay in ${CRITIC_LAY[@]}
        do
            for alr in ${ACTOR_LR[@]}
            do 
                for clr in ${CRITIC_LR[@]}
                do
                    for seed in ${SEEDS[@]}
                    do  
                        txt="$arch $alay $clay $alr $clr ($seed):\t"
                        echo -n -e $txt | tee -a hparamsearch_$arch.out
                        cmd="python main.py --groups_file groups.txt \
                                            --test \
                                            --seed $seed \
                                            --arch $arch \
                                            --epoch_max $EPOCH_MAX \
                                            --actor_lr $alr \
                                            --critic_lr $clr \
                                            --actor_mlp_layers $alay \
                                            --critic_mlp_layers $clay \
                                                2>/dev/null \
                                            | grep 'Average.*flows:\|Number' \
                                            | sed 's/[^0-9.()/]//g' \
                                            | tr '\n' ' ' \
                                            | tee -a hparamsearch_$arch.out"
                        eval "$cmd"
                        echo " " | tee -a hparamsearch_$arch.out
                    done
                done
            done
        done
    done
done