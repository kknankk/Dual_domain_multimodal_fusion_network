import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from scipy.signal import stft, istft
import torchaudio

class ImagePatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d_model=256, in_channels=3):
        super(ImagePatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)  # (img_size, img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv_layer = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        B, C, H, W = image.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        image = self.conv_layer(image).flatten(2).transpose(1, 2)  # (B, H*W, D)
        return image

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),
                                          nn.Dropout(dropout),
                                          nn.GELU(),
                                          nn.Linear(d_ff, d_model),
                                          nn.Dropout(dropout))

    def forward(self, x):
        return self.feed_forward(x)

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super(AddNorm, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(d_model, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x)
        x_ = x
        x = self.dropout(x)
        x = self.feed_forward(x) + x_
        x = self.norm2(x)
        return x


class Image2TextGate(nn.Module):
    def __init__(self, n, d_model):
        super(Image2TextGate, self).__init__()
        self.n = n
        self.avg_pool = nn.AvgPool1d(kernel_size=n)
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.select_para = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))

    def forward(self, image):
        B, N, C = image.shape
        assert N == self.n
        image = image * torch.view_as_complex(self.select_para)
        image = image.permute(0, 2, 1)  # (B, C, N)
        image = self.avg_pool(image.real)  # (B, C, 1)
        image = self.conv_layer(image)  # (B, C, 1)
        image = image.permute(0, 2, 1)  # (B, 1, C)
        return image

class Text2ImageGate(nn.Module):
    def __init__(self, s, d_model):
        super(Text2ImageGate, self).__init__()
        self.s = s
        self.avg_pool = nn.AvgPool1d(kernel_size=s)
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.select_para = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))

    def forward(self, text):
        text = text * torch.view_as_complex(self.select_para)  # (B, S, C)
        text = text.permute(0, 2, 1)
        text = self.avg_pool(text.real)  # (B, C, 1)
        text = self.conv_layer(text)  # (B, C, 1)
        text = text.permute(0, 2, 1)  # (B, 1, C)
        return text

class ImageFrequencySelection(nn.Module):
    def __init__(self, s, d_model):
        super(ImageFrequencySelection, self).__init__()

        self.text_gate = Text2ImageGate(s, d_model)

    def forward(self, image, text):
        """
        image: (B, N, C)  N=h*w  in frequency domain
        """
        text_gate = self.text_gate(text)
        image = image * text_gate
        return image

class TextFrequencySelection(nn.Module):
    def __init__(self, n, d_model):
        super(TextFrequencySelection, self).__init__()

        self.image_gate = Image2TextGate(n, d_model)

    def forward(self, text, image):
        image_gate = self.image_gate(image)
        text = text * image_gate
        return text

pi=3.1415926

class FtLayer(nn.Module):
    def __init__(self, d_model, s, n, num_filter=2, dropout=0.,use_bank=True):
        super(FtLayer, self).__init__()
        self.s = s
        self.n = n
        self.use_bank = use_bank
        self.num_filter = num_filter

        self.text_weight = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))
        self.text_filter_bank = nn.Parameter(torch.randn(num_filter, s, d_model, 2, dtype=torch.float32))

        self.image_weight = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))
        self.image_filter_bank = nn.Parameter(torch.randn(num_filter, n, d_model, 2, dtype=torch.float32))

        self.text_frequency_select = TextFrequencySelection(n, d_model)
        self.image_frenquency_select = ImageFrequencySelection(s, d_model)

        self.text_add_norm = AddNorm(d_model, dropout)
        self.image_add_norm = AddNorm(d_model, dropout)

    def filter(self, x, length, filter_bank, weight):


        if self.use_bank:
            # print(x.device)
            power = (x * x) / length
            # print(power.device)#cpu
            # print(power.device, filter_bank[0].device, cos.device)
            Y = []
            for k in range(self.num_filter):
                cos = torch.cos(torch.as_tensor((2 * (k + 1) - 1) * pi / 2 * self.num_filter))
                Y.append(power * filter_bank[k] * cos)
               

            C = torch.stack(Y)  # (filter, batch, s, dim)
            x = torch.sum(C, dim=0)  # (batch, s, dim)
        else:
            x = x * weight

        return x

    def forward(self, ecg, image, spatial_size=None):
        x_ecg = ecg#[B,12,4096]
        # print(f'FtLayer ecg input {ecg.shape}')
        # batch_size,feature1,feature2,dim=ecg.shape #[B,257,17,d_model]
        # ecg = ecg.reshape(batch_size, feature1 * feature2, dim)#[B,257*17,12]
        initial_ecg=ecg
        
        
        # print(f'ftlayer ecg {ecg.shape}')#[batch,4096,128]
        B, S, D = ecg.shape
        # print(f'before stft ecg device {ecg.device}')#cuda
        # print(f'B,S,D {ecg.shape}')#[16,257*17,d_model]
        # print(f'FtLayer S {S}') #257*17
        # print(f'FtLayer s {self.s}')#257*17
        # assert S  == self.s #s should be 1542
        

        x_image = image
        B, N, C = image.shape #should be [B,224*224,3]
        # print(f'FtLayer N {N}')#196
        # print(f'FtLayer n {self.n}')#99
        assert N // 2 + 1 == self.n #n should be 25089
        # if spatial_size:
        #     a, b = spatial_size
        # else:
        #     a = b = int(math.sqrt(N))

        # fft
        # _text = torch.fft.rfft(text, dim=1, norm='ortho')
        # _ecg=ecg
        ecg1=ecg.detach().cpu().numpy()
        ecg1=ecg1.transpose(0,2,1)
        # ecg = ecg.permute(0, 2, 1)
        # print(f'111 ecg {ecg.shape}')#[batch,128,4096]
        # ecg = ecg.permute(0, 2, 1)  # 直接改变形状为 [B, L, C]

        # print(f'before sitf ecg1 {ecg.shape}') #[batch,4096,128]

        f,t, _ecg = stft(ecg1,fs=400, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]
       

# 使用 torchaudio 进行 STFT#输入需要是[channel,timestamp]

        # window = torch.hann_window(window_length).to(ecg.device)  
        # _ecg = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256)(ecg)
#         n_fft = 512
#         hop_length = 256

# # 创建窗口并将其移动到相同的设备
#         window = torch.hann_window(n_fft).to(ecg.device)

# # 使用 torchaudio 进行 STFT
#         _ecg = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, window=window)(ecg)
        # print(f'_ecg 1111{_ecg.shape}')

        # print(f'after stft _ecg2 {_ecg.shape}')#[batch,128,257,17]
        _ecg=_ecg.transpose(0,2,3,1)

        a,b,c,d=_ecg.shape
        _ecg=_ecg.reshape(a,b*c,d)
        B,S,D=_ecg.shape
        assert S  == self.s
        # print(f'after transpose _ecg2 {_ecg.shape}')#[batch,4369,128]
        _ecg = torch.tensor(_ecg, dtype=torch.complex64)
        ecg_device = ecg.device  # 获取 ecg 的设备
        _ecg = _ecg.to(ecg_device)
        # print(f' after stft _ecg device{_ecg.device}')#cuda
        # print(f'after stft in FURU {_ecg.shape}')


        _image = torch.fft.rfft(image, dim=1, norm='ortho') #complex
        # print(f'_image shape {_image.shape}')#[batch,99,128]
        
        # is_complex = _image.is_complex()
        # print(f"Is the 0 _image data complex? {is_complex}")
        # is_complex = ecg.is_complex()
        # print(f"Is the 0 ECG data complex? {is_complex}")
        # frequency filter
        
        _ecg = self.filter(_ecg, self.s, torch.view_as_complex(self.text_filter_bank),
                            torch.view_as_complex(self.text_weight))
        _image = self.filter(_image, self.n, torch.view_as_complex(self.image_filter_bank),
                             torch.view_as_complex(self.image_weight))
        is_complex = _ecg.is_complex()
        # print(f"Is the 1 ECG data complex? {is_complex}")
        is_complex = _image.is_complex()
        # print(f"Is the 1 _image data complex? {is_complex}")
        # frequency select
        # _ecg = self.text_frequency_select(_ecg, _image)
        # _image = self.image_frenquency_select(_image, _ecg)
        ecg = self.text_frequency_select(_ecg, _image)
        image = self.image_frenquency_select(_image, _ecg)
        # print(f'after select ecg {ecg.shape}')
        # print(f'after select cxr {image.shape}')#[batch,99,128]
        # is_complex = ecg.is_complex()
        # print(f"Is the 2 ECG data complex? {is_complex}") #True
        # is_complex = image.is_complex()
        # print(f"Is the 2 image data complex? {is_complex}")
#====TODO:标记我删去了ifft
        # ifft
        # text = torch.fft.irfft(_text, n=S, dim=1, norm='ortho')
        image = torch.fft.irfft(_image, n=N, dim=1, norm='ortho')
        image = image.view(B, N, C)
        ecg=ecg.detach().cpu().numpy()
        # print(f'before istft ecg {ecg.shape}')
        # ecg=ecg.transpose(0,2,1)
        # print(f'before istft trans ecg {ecg.shape}')#[batch,4396,128]
        ecg=ecg.reshape(a,b,c,d)#[batch,257,17,128]
        ecg=ecg.transpose(0,3,1,2)
        # print(f'after reshape stft ecg {ecg.shape}')
        _,ecg = istft(ecg,fs=400, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]
        ecg = torch.tensor(ecg)
        ecg_device = initial_ecg.device  # 获取 ecg 的设备
        ecg = ecg.to(ecg_device)
        # print(f' after stft _ecg device{_ecg.device}')#cuda
        # print(f'after istft in FURU {ecg.device}')
        # print(f'after istft ecg {ecg.shape}')
        # print(f'x_ecg {x_ecg.shape}')#16,4096,128
        ecg=ecg.permute(0,2,1)
        # print(f'final ')
        # 

        

        # add & norm
        text = self.text_add_norm(ecg + x_ecg)
        image = self.image_add_norm(image + x_image)
        # print(f'after ftlayer ecg {text.shape}; cxr {image.shape}')
        #[batch,4096,128];[batch,196,128]
        return text, image


class FtBlock(nn.Module):
    def __init__(self, d_model, s, n, num_layer=1, num_filter=2, dropout=0.):
        """
        :param d_model:
        :param s: seq_len / 2 + 1
        :param h:
        :param w:
        :param n:
        """
        super(FtBlock, self).__init__()
        self.ft = nn.ModuleList([FtLayer(d_model, s, n, num_filter, dropout) for _ in range(num_layer)])

    def forward(self, text, image):
        for ft_layer in self.ft:
            text, image = ft_layer(text, image)

        return text, image

class Fusion(nn.Module):
    def __init__(self, d_model, act_layer=torch.tanh):
        super(Fusion, self).__init__()

        self.text_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.image_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.fusion_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.act_layer = act_layer

    def forward(self, text, image):
        alpha = self.js_div(text, image)
        a=torch.matmul(text, self.text_weight)
        b=torch.matmul(image, self.image_weight)
        print(f'before fusion text {text.shape}')
        print(f'before fusion image {image.shape}')
        print(f'before fusion self.text_weight {self.text_weight.shape}')
        print(f'before fusion self.image_weight {self.image_weight.shape}')
        # print(f'in fusion part ecg {a.shape}; cxr {b.shape}')#[batch,128]; [batch,128]
        fusion = torch.matmul(text, self.text_weight) + torch.matmul(image, self.image_weight)
        f = (1-alpha) * fusion + alpha * text + alpha * image

        return f

    @staticmethod
    def js_div(p, q):
        """
        Function that measures JS divergence between target and output logits:
        """
        M = (p + q) / 2
        kl1 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        gamma = 0.5 * kl1 + 0.5 * kl2
        return gamma

class MLP(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim, num_class, act_layer=nn.ReLU, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_dim, outputs_dim)
        self.norm2 = nn.LayerNorm(outputs_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(outputs_dim, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act_layer(x)
        x = self.fc3(x)
        return x


class FSRU(nn.Module):
    def __init__(self,  d_text=12, seq_len=4369, img_size=224, patch_size=16, d_model=128,
                 num_filter=2, num_class=4, num_layer=1, dropout=0., mlp_ratio=4.):
        super(FSRU, self).__init__()

        # Text

        self.text_encoder = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=d_model, kernel_size=17, stride=1, padding=8),  # 卷积层

            # nn.Linear(d_text, d_model),
                                        #   nn.LayerNorm(d_model),
                                          )
        # s = seq_len // 2 + 1
#=====未对ecg做rfft,所以特征维度不会变为1/2
        self.ecg_norm=nn.LayerNorm(d_model)
        s=seq_len


        # Image
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        self.fusion = Fusion(d_model)

        self.mlp = MLP(d_model, int(mlp_ratio*d_model), d_model, num_class, dropout=dropout)

        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
            # trunc_normal_(m.weight, std=.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image):
        # text = text.long()
        # text = self.text_embed(text)  # (batch, seq, dim)
        # print(f'input ecg {text.shape}')#[Batch,4096,12]
        # print(f'image {image.shape}')
        # print(f' FUSU input ecg {text.dtype}')
        # print(f'FUSU INPUT 1 ECG {text.shape}')
        text=text.permute(0,2,1)
        # print(f'input ecg {text.shape}')
        text = self.text_encoder(text)
        # print(f'after encoder ecg {text.shape}')
        text=text.permute(0,2,1)
        text=self.ecg_norm(text)

        image = image.to(torch.float32)
        image = self.img_patch_embed(image)
        image = image + self.img_pos_embed
        image = self.img_pos_drop(image)

        text, image = self.FourierTransormer(text, image)

        text = torch.max(text, dim=1)[0]
        image = torch.max(image, dim=1)[0]

        f = self.fusion(text, image)  # (batch, d_model)

        outputs = self.mlp(f)

        return text, image, outputs, f
