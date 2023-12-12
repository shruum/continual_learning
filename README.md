
## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following  

    `python main.py --dataset <dataset> --model <model> --buffer_size <buffer_size> --load_best_args`

 Examples:

    ```
    python main.py --dataset seq-mnist --model er --buffer_size 500 --load_best_args
    
    python main.py --dataset seq-cifar10 --model er --buffer_size 500 --load_best_args
    

## Requirements

- torch==1.7.0

- torchvision==0.9.0 

- quadprog==0.1.7
