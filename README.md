# Precure StyleGAN ADA

StyleGAN 2.0 [a2](#a1) + Adaptive Discriminator Augmentation

This project follows on from [the previous project: Precure StyleGAN](https://github.com/curegit/precure-stylegan).

FID = 26.41, dataset = 2.5k

![](examples/beauty.png)

## Requirements

- Python >= 3.8
- Chainer >= 7.2
- Pillow >= 7.1
- NumPy >= 1.17
- h5py
- tqdm

Use `requirements.txt` to install minimal dependencies for inferencing.

```sh
pip3 install -r requirements.txt
```



### For training

You may need GPU support to train your own models.

- CuPy (with CUDA & cuDNN)
- Matplotlib

### Extra

Install the following to run `visualize.py`.

- Pydot (with GraphViz)

## See also

- [StyleGAN FastAPI](https://github.com/curegit/stylegan-fastapi)

## License

[CC BY-NC 4.0](LICENSE)
