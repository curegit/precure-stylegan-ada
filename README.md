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

You should need GPU support to train your own models.
Note that it does not support distributed training across multiple GPUs.

Matplotlib is required to draw learning curves.

### Extras

Install the following to run `visualize.py`.

- Pydot (with GraphViz)

## Script Synopses

## Results

We use ψ = 1.0 (no truncation applied) to evaluate for each Fréchet Inception Distance (FID).

### Flickr-Faces-HQ (ψ = 0.9, FID = 15.61)

![FFHQ](examples/ffhq.png)

### Animal Faces-HQ (ψ = 0.8, FID = 7.64)

![AFHQ](examples/afhq.png)

### Anime Faces (ψ = 0.7, FID = 13.81)

![Anime](examples/anime.png)

### MNIST (ψ = 1.1, FID = 2.61)

![MNIST](examples/mnist.png)

### Kuzushiji-49 (ψ = 1.0, FID = 3.77)

![Kuzushiji-49](examples/k49.png)

## See Also

- [StyleGAN FastAPI](https://github.com/curegit/stylegan-fastapi)
- [Precure StyleGAN](https://github.com/curegit/precure-stylegan) (old project)

## References

### Papers

- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)
- [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)
- [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis](https://arxiv.org/abs/1903.05628)
- [On Leveraging Pretrained GANs for Generation with Limited Data](https://arxiv.org/abs/2002.11810)

### Datasets

- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [Deep Learning for Classical Japanese Literature](https://arxiv.org/abs/1812.01718)
- [Anime-Face-Dataset](https://github.com/Mckinsey666/Anime-Face-Dataset)
- [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)

## License

[CC BY-NC 4.0](LICENSE)
