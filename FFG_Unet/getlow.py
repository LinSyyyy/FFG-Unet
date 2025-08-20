import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F

class WeightUpdateModule(nn.Module):
      def __init__(self, num_channels):
          super(WeightUpdateModule, self).__init__()
          self.fc1 = nn.Conv1d(num_channels*2, num_channels,kernel_size=1)  # 全连接层
          self.fc2 = nn.Linear(num_channels, num_channels)  # 全连接层
          self.relu = nn.ReLU()  # ReLU 激活
          self.sigmoid = nn.Sigmoid()  # Sigmoid 激活
#
      def forward(self, x):
          x3 = F.max_pool2d(x, 2, 2)
          x2 = F.avg_pool2d(x, 2, 2)
          x_pooled = torch.cat((x2,x3),dim=1)
          x_pooled = x_pooled.mean(dim=[2, 3])
          # x_pooled = torch.cat((x2,x3),dim=1)

          x_pooled = self.fc1(x_pooled.transpose(1,0)).transpose(1,0)
          weights = self.fc2(x_pooled)  # (1, 16)
          weights = self.relu(weights)  # (1, 16)
          weights = self.sigmoid(weights)  # (1, 16)
          weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
          return weights_expanded



def extract_low_freq_wavelet(image, wavelet='haar'):
    image_np = image.squeeze(0).detach().cpu().numpy()  # (C, H, W)
    coeffs = pywt.dwt2(image_np, wavelet)
    cA, (cH, cV, cD) = coeffs


    low_freq = torch.tensor(cA).unsqueeze(0)  # (1, H/2, W/2)
    low_freq_upsampled = F.interpolate(low_freq, size=image.shape[2:], mode='bilinear', align_corners=False)
    return low_freq_upsampled


class extract_low_freq_wavelet_multichannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.WeightUpdateModule = WeightUpdateModule(num_channels=16)

    def forward(self,images,wavelet='haar'):
        low_freqs = []
        # oo = self.WeightUpdateModule(images)
        for i in range(images.shape[0]):
            image_np = images[i].detach().cpu().numpy()  # (C, H, W)
            coeffs = pywt.dwt2(image_np, wavelet)
            cA, (cH, cV, cD) = coeffs

            low_freq = torch.tensor(cA).unsqueeze(0)  # (1, H/2, W/2)
            low_freq_upsampled = F.interpolate(low_freq, size=images[i].shape[1:], mode='bilinear', align_corners=False)

            low_freqs.append(low_freq_upsampled)

        return torch.cat(low_freqs, dim=0)


if __name__=="__main__":

    features1 = torch.randn(1, 16, 128, 128).cuda()
    features2 = torch.randn(1, 16, 64, 64).cuda()
    features3 = torch.randn(1, 16, 32, 32).cuda()
    extract_low_freq_wavelet_multichannel = extract_low_freq_wavelet_multichannel().cuda()

    low_freq1 = extract_low_freq_wavelet_multichannel(features1)  # (1, 16, 128, 128)
    low_freq2 = extract_low_freq_wavelet_multichannel(features2)  # (1, 32, 64, 64)
    low_freq3 = extract_low_freq_wavelet_multichannel(features3)  # (1, 128, 32, 32)

    print("Low Frequency 1 Shape:", low_freq1.shape)  # (1, 16, 128, 128)
    print("Low Frequency 2 Shape:", low_freq2.shape)  # (1, 32, 64, 64)
    print("Low Frequency 3 Shape:", low_freq3.shape)  # (1, 128, 32, 32)
