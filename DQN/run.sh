# cd "$(dirname "$0")"
# Set current working directory to the parent directory of the script
cd "$(dirname "$0")"
cd ..
time_takens=()
directory="./DQN/"
files=("dqn.py" "dqn_stablebaseline3.py" "dqn_tianshou.py")
args=("gym-pendulum" "gym-swimmer" "gym-halfcheetah")
for arg in ${args[@]}; do
    for file in ${files[@]}; do
        python3.10 $directory$file $arg True
    done
done