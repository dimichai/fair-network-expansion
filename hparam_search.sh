#!/bin/bash

ARCHS=("rnn")
SEEDS=(20 303 5789)
ACTOR_LAY=(2 3 4 5 6)
CRITIC_LAY=(1 2 3 4 5)
# ACTOR_LR=(0.001 0.0005 0.0001 0.00005)
# CRITIC_LR=(0.001 0.0005 0.0001 0.00005)
ACTOR_LR=(0.0001)
EPOCH_MAX=500

METRO_ENV_L=(diagonal_5x5 dilemma_5x5 xian amsterdam)
METRO_ENV=${METRO_ENV_L[0]}

PRINT_ONLY=0

for arch in ${ARCHS[@]}
do
    if [ $PRINT_ONLY -eq 0 ]; then
        echo -e "\n-------------------" >> hps_"$METRO_ENV"_$arch.txt
    fi
    for alay in ${ACTOR_LAY[@]}
    do
        for clay in ${CRITIC_LAY[@]}
        do
            for alr in ${ACTOR_LR[@]}
            do 
                if [ $alay -gt $clay ]; then
                # diff=$alay-$clay
                # echo $diff
                if [[ $((alay-clay)) -lt 4 ]]; then
                # for clr in ${CRITIC_LR[@]}
                # do
                    for seed in ${SEEDS[@]}
                    do
                        if [ $PRINT_ONLY -eq 0 ]; then
                            d=`date +"%d-%b-%Y %T"`
                            txt="|$d| epochs=$EPOCH_MAX - $arch $alay $clay $alr $alr ($seed):\t"
                            echo -n -e $txt | tee -a hps_"$METRO_ENV"_$arch.txt
                        fi
                        cmd="python main.py --environment $METRO_ENV \
                                            --test \
                                            --seed $seed \
                                            --arch $arch \
                                            --epoch_max $EPOCH_MAX \
                                            --actor_lr $alr \
                                            --critic_lr $alr \
                                            --actor_mlp_layers $alay \
                                            --critic_mlp_layers $clay \
                                            --early_stopping 20"
                        if [ $PRINT_ONLY -eq 0 ]; then
                            eval "$cmd 2>/dev/null \
                                    | grep 'Average.*flows:\|Number' \
                                    | sed 's/[^0-9.()/]//g' \
                                    | tr '\n' ' ' \
                                    | tee -a hps_"$METRO_ENV"_$arch.txt"
                            echo " " | tee -a hps_"$METRO_ENV"_$arch.txt
                        else
                            echo $cmd
                        fi
                    done
                # done
                fi
                fi
            done
        done
    done
done