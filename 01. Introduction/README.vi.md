# Giới thiệu về GAN

## GAN là gì?

Được đánh giá là ý tưởng thú vị nhất thập kỷ trong lĩnh vực Machine Learning (học máy), GAN đã giành được những thành công vang dội kể từ khi ra mắt vào năm 2014 ([paper](https://arxiv.org/abs/1406.2661)) bởi [Ian Goodfellow](https://en.wikipedia.org/wiki/Ian_Goodfellow) và đồng nghiệp. Vậy GAN là gì?

**GAN** viết tắt của `Generative Adversarial Network` (_xin phép không dịch ra tiếng Việt_), thuộc nhóm các mô hình _generative_, có nghĩa là có khả năng sinh ra dữ liệu mới.

![stylegan2](images/stylegan2-teaser.png)
Những hình ảnh người ở trên hoàn toàn không có thật mà được tạo ra từ mạng [StyleGAN2](https://github.com/NVlabs/stylegan2) - một kiến trúc mạng GAN do NVIDIA phát triển.

Từ `Adversarial` trong GAN có nghĩa là đối nghịch do GAN có cấu trúc gồm 2 mạng có nhiệm vụ trái ngược với nhau. Chi tiết hơn về cấu trúc GAN sẽ được trình bày ở phần dưới đây:

## Cấu trúc của GAN

GAN có 2 phần:

- **Generator**: Cố gắng tạo ra dữ liệu giống như dữ liệu trong dataset.
- **Discriminator**: Cố gắng phân biệt dữ liệu do bộ **Generator** tạo ra và dữ liệu trong dataset

![GAN example](images/gan-dzone.png)

> Ảnh được lấy từ [Working Principles of Generative Adversarial Networks (GANs)](https://dzone.com/articles/working-principles-of-generative-adversarial-netwo)

Bộ **Generator** và **Discriminator** giống như kẻ làm tiền giả và cảnh sát vậy. Kẻ làm tiền giả luôn cố gắng làm tiền giả giống thật nhất có thể còn cảnh sát thì có nhiệm vụ phân biệt tiền giả và tiền thật.

Ban đầu, kẻ làm tiền giả còn ít kinh nghiệm, đã tạo ra những đồng tiền giả quá dễ phân biệt, khiến cho cảnh sát nhanh chóng biết được đó là tiền giả:
![](images/bad_gan.svg)

Sau nhiều lần thất bại, kẻ làm tiền giả càng ngày càng tạo ra tiền giả giống thật hơn:
![](images/ok_gan.svg)

Cuối cùng, nếu kẻ làm tiền giả vẫn chưa bị bắt, hắn sẽ tạo ra những đồng tiền giả rất giống thật, đủ để qua mặt cảnh sát, khiến cho cảnh sát không thế phân biệt được đâu là tiền thật, đâu là tiền giả nữa:
![](images/good_gan.svg)

> Hình ảnh và ví dụ lấy từ [Overview of GAN Structure](https://developers.google.com/machine-learning/gan/gan_structure)

Kiến trúc của GAN được mô tả tổng quát như sau:
![](images/gan_diagram.svg)

Hai bộ **Generator** và **Discriminator** giống như tham gia một trò chơi đối kháng, khi mà một bên giành được lợi thế thì tương ứng bên kia bị bất lợi. Trong [lý thuyết trò chơi](https://vi.wikipedia.org/wiki/L%C3%BD_thuy%E1%BA%BFt_tr%C3%B2_ch%C6%A1i), tình huống này được gọi là [trò chơi có tổng bằng không](https://vi.wikipedia.org/wiki/Tr%C3%B2_ch%C6%A1i_c%C3%B3_t%E1%BB%95ng_b%E1%BA%B1ng_kh%C3%B4ng). Bạn này hướng thú có thể đọc để tìm hiểu thêm.

Phần tiếp theo sẽ là đi sâu vào chi tiết các thành phần và cách huấn luyện GAN. Để trực quan và dễ hiểu, mình sẽ lấy ví dụ bài toán xây dựng GAN để tạo ra chữ số viết tay giả, sử dụng bộ dataset quen thuộc [MNIST](http://yann.lecun.com/exdb/mnist). Toàn bộ phần code mình sẽ sử dụng pytorch.

## Bộ Generator

Như đã nói ở [phần trên](#cấu-trúc-của-gan), bộ **Generator** sẽ có:

- Đầu vào là một vector
- Đầu ra là một ảnh

Kiến trúc của bộ **Generator** có thể sử dụng bất cứ kiến trúc mạng neural nào. Mình sẽ bắt đầu với kiến trúc neural network kiểu sequence như sau:

![Kiến trúc bộ Generator](images/generator.png)

Input của mạng neural này mình chọn là một vector 100 chiều. Vector này sẽ được sinh random từ [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) hoặc [uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution). Trong pytorch, mình sinh ra input vector bằng dòng code sau:

```python
z = torch.FloatTensor(np.random.normal(0, 1, (latent_dim,)))
```

Sau đó, input vector sẽ được đi qua lần lượt các lớp ẩn (hidden layer) có kích thước 256, 512, 1024 và kết thúc ở lớp có kích thước 768. Kích thước các lớp ẩn có thể được chọn tuỳ ý, tuy nhiên kích thước lớp cuối cùng cần phải là 768 để có thể reshape về ảnh có kích thước 28x28 (kích thước ảnh trong bộ dataset MNIST)

Bạn có thể tham khảo toàn bộ code của khối Generator:

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 768),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 28, 28)
        return img
```

## Bộ Discriminator

Như đã giới thiệu ở các phần trên, bộ Discriminator có nhiệm vụ phân biệt đâu là ảnh chữ viết tay thật, đâu là ảnh được bộ Generator tạo ra. Vì vậy, bộ Discriminator có:

- Đầu vào là một ảnh
- Đầu ra là giá trị phân loại ảnh là thật hay giả

Đây chính là bài toán phân loại ảnh thường thấy, cụ thể là [binary classification](https://en.wikipedia.org/wiki/Binary_classification). Mình cũng xây dựng bộ Discriminator một cách đơn giản như sau:

![Kiến trúc bộ Discriminator](images/discriminator.png)

Đầu vào của mạng là một ảnh có kích thước 28x28 và được resize về một vector 768 chiều. Tiếp theo mạng có các lớp ẩn với kích thước lần lượt 1024, 512, 256. Và cuối cùng, lớp output là vector có một chiều, đại diện cho xác suất ảnh đầu vào là ảnh thật.

Dưới đây là code bộ Discriminator mình đã triển khai trong pytorch:

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        output = self.model(img_flat)

        return output
```

## Huấn luyện GAN
