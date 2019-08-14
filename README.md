# Captcha

## Tutorial

- <https://juejin.im/post/5c073bde51882546150ac75a>
- <https://www.jianshu.com/p/f2184bc6c1f2>

## Data

- <https://pan.baidu.com/s/1N7bDHxIM38Vu7x9Z2Qr0og>

## Files intro

- `glo_var` - global variables and constants
- `img_gen` - preprocess image and label
- `model` - definition of the CNN
- `train` - runnable. Training model
- `predict` - runnable. Predict and test the saved model instance

## Techniques & Trouble-shooting

### Saving RAM

<https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71>

### `loss` won't get down

1. Check data preprocess
2. Or add `Dropout` layer under `Dense` one and tune its rate to `0.25`
    - the current state might stuck in a local optimal (over-fitting) if not
3. or to make learning rate dynamic

PS: the optimal method `Adam` is perfect for this problem. No need to tune it.

### The `loss` still remains high

- Add more `Dense` layers
