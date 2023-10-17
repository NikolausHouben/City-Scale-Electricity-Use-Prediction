#!/usr/local/bin/bash

declare -A KEYS=(
    #["5_building.h5"]="building_0 building_1 building_2"
    #["4_neighborhood.h5"]="neighborhood_0 neighborhood_1 neighborhood_2"
    #["3_village.h5"]="village_0 village_1 village_2"
    #["2_town.h5"]="town_0 town_1 town_2"
    ["1_county.h5"]="Los_Angeles" # New_York Sacramento"
)

# Iterate over the KEYS
for scale in "${!KEYS[@]}"; do
    for location in ${KEYS[$scale]}; do

        python bin/train.py --scale "$scale" --location "$location" --evaluate True
    done
done

