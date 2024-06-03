# cd "$(dirname "$0")"
# Set current working directory to the parent directory of the script
cd "$(dirname "$0")"
cd ..
time_takens=()
directory="./DQN/"
files=("dqn_tianshou.py")
args=("gym-pendulum" "gym-swimmer" "gym-halfcheetah")
for file in ${files[@]}; do
    for arg in ${args[@]}; do
        t_start=$(date +%s%N)
        python3.10 $directory$file $arg false
        # python3.10 $directory$file $arg True
        t_end=$(date +%s%N)
        time_takens+=($((t_end-t_start)))
    done
done


echo "Time taken for each run: ${time_takens[@]}"