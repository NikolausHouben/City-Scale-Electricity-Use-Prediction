#!/usr/local/bin/bash

declare -A KEYS=(
    ["GLENDOVEER"]="13596.MWh 13597.MWh" #13598.MWh 13599.MWh CLIFFGATE.MWh NORTHEAST.MWh"
    # ["LENTS"]="13101.MWh HAPPY.VALLEY.MWh MT.SCOTT.MWh NORTH.MWh"
    # ["MIDWAY"]="DIVISION.MWh DOUGLAS.MWh LYNCH.MWh POWELLHURST.MWh"
    # ["RAMAPO"]="EMERALD.MWh GILBERT.MWh RAMAPO.13.MWh"
    # ["KELLY.BUTTE"]="BINNSMEAD.MWh FAIRLAWN.MWh MALL.205.MWh MCGREW.MWh"
)

# Iterate over the KEYS
for scale in "${!KEYS[@]}"; do
    for location in ${KEYS[$scale]}; do

        python train.py --scale "$scale" --location "$location"
    done
done

