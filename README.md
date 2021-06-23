# lightning-segmentation

First attempt to write binary segmentation pipeline using [pytorch lightning](https://www.pytorchlightning.ai/) and [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch).

## Installation

Clone repo and install requirements

```bash
git clone https://github.com/gotsulyakk/lightning-segmentation.git
cd lightning-segmentation
pip install -r requirements.txt

cd lit-segmentation
```

## Usage

Train
```bash
python train.py --config {PATH/TO/CONFIG}
```
Demo inference
```bash
python demo.py --image {PATH/TO/IMAGE} --model_ckpt {PATH/TO/MODEL_CKPT} --config {PATH/TO/CONFIG}
```
## License
[MIT](https://choosealicense.com/licenses/mit/)
