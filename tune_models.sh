/usr/local/bin/bash tune_models.sh

declare -A KEYS=(
    #["5_building"]="building_0 building_1 building_2"
    #["4_neighborhood"]="neighborhood_0 neighborhood_1 neighborhood_2"
    #["3_village"]="village_0 village_1 village_2"
    #["2_town"]="town_0 town_1 town_2"
    ["1_county"]="Los_Angeles" # New_York Sacramento"
)

# Iterate over the KEYS
for scale in "${!KEYS[@]}"; do
    for location in ${KEYS[$scale]}; do
        python bin/tuning.py --scale "$scale" --location "$location" --n_sweeps 3
    done
done
