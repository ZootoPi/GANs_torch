# Conditional GAN

Gan được ứng dụng vào bài toán sinh số viết tay trong 2 bài trước, [GAN](../01.%20Introduction/README.md) và [DCGAN](../02.%20DCGAN/README.md). Kết quả thu được cũng khá khả quan :smile:

Tuy nhiên có 1 vấn đề nhỏ của ảnh được sinh ra: chúng ta không biết ảnh sinh ra là số gì, chỉ biết rằng đó là số. Vậy làm thế nào để có thể nói cho mô hình biết rằng hãy sinh ra số 1, số 2 đi? Conditional GAN sinh ra để giải quyết vấn đề đó.

## Kiến trúc cGAN

Conditional GAN được giới thiệu ngay sau khi GAN được ra mắt ([bài báo](https://arxiv.org/abs/1411.1784)) với ý tưởng khá đơn giản: nối thêm vector label vào input của cả bộ Generator và Discriminator.

![cGAN](images/cgan.png)

> Kiến trúc cGAN (ảnh lấy từ bài báo gốc)

Cúng không có quá nhiều để nói về mô hình này. Bắt tay triển khai thử thôi :muscle:

## Triển khai và kết quả

Mình sẽ sử dụng lại gần như [toàn bộ code của bài GAN](../01.%20Introduction/MNIST_GAN.ipynb) với 1 chút thay đổi nhỏ:

### Bộ Discriminator

### Bộ Generator

### Kết quả
