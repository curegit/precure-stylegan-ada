# Precure StyleGAN ADA

StyleGAN 2.0 + Adaptive Discriminator Augmentation

This project follows on from [the previous project: Precure StyleGAN](https://github.com/curegit/precure-stylegan).

FID = 26.41, dataset = 2.5k

![](examples/beauty.png)

## Requirements

- Python >= 3.8
- Chainer >= 7.2
- Pillow >= 9.1
- NumPy >= 1.17
- h5py
- tqdm

Use `requirements.txt` to install minimal dependencies for inferencing.

```sh
pip3 install -r requirements.txt
```

### For training

- CuPy (with CUDA & cuDNN)
- Matplotlib

You may need GPU support to train your own models.

Matplotlib is required to draw learning curves.

### Extra

Install the following to run `visualize.py`.

- Pydot (with GraphViz)

## Results

We use ψ = 1.0 to evaluate for each Fréchet Inception Distance (FID).

### Flickr-Faces-HQ (ψ = 0.9, FID = 15.61)

![FFHQ](examples/ffhq.png)

### Animal Faces-HQ (ψ = 0.8, FID = 7.64)

![AFHQ](examples/afhq.png)

## See also

- [StyleGAN FastAPI](https://github.com/curegit/stylegan-fastapi)
- [Precure StyleGAN](https://github.com/curegit/precure-stylegan) (old project)

## License

[CC BY-NC 4.0](LICENSE)
