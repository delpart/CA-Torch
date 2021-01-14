# CA-Torch
(Partial) Implementation of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/).

## Usage
```
python3 main.py --help
usage: main.py [-h] [--batchsize BATCHSIZE] [--poolsize POOLSIZE]
               [--casize CASIZE] [--progresspath PROGRESSPATH]
               [--modelpath MODELPATH] [--device DEVICE]
               target

Train a Neural Cellular Automata to grow into the shape of a target image.

positional arguments:
  target                target image

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -bs BATCHSIZE
                        batch size
  --poolsize POOLSIZE, -ps POOLSIZE
                        pool size
  --casize CASIZE, -cas CASIZE
                        CA size
  --progresspath PROGRESSPATH, -pp PROGRESSPATH
                        path to save progress visualisations
  --modelpath MODELPATH, -mp MODELPATH
                        path to save models
  --device DEVICE, -d DEVICE
                        device to use for training (auto, cuda or cpu)
```
