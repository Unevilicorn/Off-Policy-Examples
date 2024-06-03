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
        python3.10 $directory$file $arg True
    done
done