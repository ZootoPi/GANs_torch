# Deep Convolutional Generative Adversarial Network

Ở [bài trước]() chúng ta đã được giới thiệu về bài toán sinh ra chữ viết tay sử dụng mô hình GAN. Đây là một bài toán có dữ liệu là dạng hình ảnh, tuy nhiên bộ **Generator** và **Discriminator** đều đang sử dụng mạng neural thông thường. Với các bài toán dữ liệu hình ảnh thì kiến trúc mạng CNN ([Convolution Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)) đang được ưa chuộng và đã chứng minh hiệu quả của mình so với mạng neural truyền thống. Vậy có cách nào để kết hợp mô hình CNN với mô hình GAN không? Và hiệu quả của mô hình kết hợp đó như thế nào?

Câu trả lời chính là [Deep Convolutional Generative Adversarial Network - DCGAN](https://arxiv.org/abs/1511.06434). Về cơ bản DCGAN giữa nguyên kiến trúc của GAN gốc, chỉ thay đổi cấu tạo của bộ **Generator** và **Discriminator** từ mạng neural truyền thống sang sử dụng CNN

![cnn](images/cnn.png)

> Kiến trúc cơ bản của CNN (nguồn: https://en.wikipedia.org/wiki/Convolutional_neural_network)

Tiếp tục với ví dụ về sinh chữ viết tay ở [bài trước](), kiến trúc của bộ **Generator** và **Discriminator** sẽ trở thành như sau:

## Discriminator

Bộ Discriminator như thường lệ, là một bộ phân loại ảnh sử dụng CNN quen thuộc:

![discriminator](images/discriminator.png)

Bộ Discriminator nhận đầu vào là một ảnh có kích thước 28x28x1, đi qua 2 lớp convolution2d, được "duỗi thẳng" (flatten) và đi qua 1 lớp Dense cuối cùng để phân loại. Đây là một mô hình CNN cơ bản nên mình sẽ không đi vào quá chi tiết phần này. Dưới đây là code cho bộ Discriminator:

```python

```

## Generator

Bộ Generator thì hơi ngược lại với kiến trúc CNN thường thấy, khi nhận đầu vào là một vector 100 chiều, qua các lớp convolution để nhận lại được một ảnh có kích thước 28x28x1:

![Generator](images/generator.png)

Dưới đây là code pytorch của bộ Generator:

```python

```

## Huấn luyện và kết quả

Quá trình huấn luyện DCGAN hiện tại hoàn toàn giữ nguyên những gì đã làm với GAN, đã được trình bày trong [bài trước](). Toàn bộ code trong tutorial này được cho tại [DCGAN.ipynb](DCGAN.ipynb), mọi người hoàn toàn có thể thử chạy trực tiếp trên colab

Kết quả thu được sau khi train 20 epoch cũng khá khả quan :heart_eyes:
