for gamma in $(python -c "import numpy as np; print('\n'.join(map(str, np.logspace(-1, 1, 9))))"); do
    for sigma in $(python -c "import numpy as np; print('\n'.join(map(str, np.logspace(-1, 1, 9))))"); do
        for i in {1..5}; do
            echo "Running with gamma=$gamma, sigma=$sigma" 1>&2
            python simulate.py --gamma $gamma --sigma_epsilon $sigma | tail -n1 | jq -r '[.true.sigma_epsilon, .true.gamma, .true.distortion_factor, .agent.sigma_epsilon, .agent.gamma, .cumulative_loss.adaptive, .cumulative_loss.rational, .cumulative_loss.adaptive_advantage, .steps] | @csv'
        done
    done
done
