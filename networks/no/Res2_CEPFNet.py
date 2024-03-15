import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from networks.res2net import res2net50
import Constants

nonlinearity = partial(F.relu, inplace=True)
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3,stride=1,padding=1,bias=True)
        self.norm1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nonlinearity
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs

class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear',
                              align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
        feat = self.conv_out(feat)
        return feat

class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(2 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(2 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat)], dim=1)
        feat = self.conv_out(feat)
        return feat


class GPG_2(nn.Module):
    def  __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)
        feat = self.conv_out(feat)
        return feat

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class CEPF_Net_(nn.Module):
    def __init__(self, num_classes=Constants.BINARY_CLASS, ccm=True, norm_layer=nn.BatchNorm2d):
        super(CEPF_Net_, self).__init__()

        filters = [256, 512, 1024, 2048]
        expan = [512, 1024, 2048]
        spatial_ch = [256, 256]

        model = res2net50(pretrained=True)
        self.firstconv = model.conv1
        self.firstbn = model.bn1
        self.firstrelu = model.relu
        self.firstmaxpool = model.maxpool
        self.encoder1 = model.layer1
        self.encoder2 = model.layer2
        self.encoder3 = model.layer3
        self.encoder4 = model.layer4

        self.mce_2 = GPG_2([spatial_ch[-1], expan[0], expan[1], expan[2]], width=spatial_ch[-1], up_kwargs=up_kwargs)
        self.mce_3 = GPG_3([expan[0], expan[1], expan[2]], width=expan[0], up_kwargs=up_kwargs)
        self.mce_4 = GPG_4([expan[1], expan[2]], width=expan[1], up_kwargs=up_kwargs)

        self.dblock = DACblock(filters[3])
        self.spp = SPPblock(filters[3])

        self.decoder4 = DecoderBlock(filters[3]+4, filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Conv4 = ConvBlock(filters[3], filters[2])

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Conv3 = ConvBlock(filters[2], filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=int(32))
        self.Conv2 = ConvBlock(filters[1], filters[0])

        self.decoder1 = DecoderBlock(filters[0], 64)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        e0 = self.firstrelu(x)
        x = self.firstmaxpool(e0)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        m2 = self.mce_2(e1, e2, e3, e4)
        m3 = self.mce_3(e2, e3, e4)
        m4 = self.mce_4(e3, e4)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4)
        e4 = self.Att4(g=d4, x=m4)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Conv4(d4)

        d3 = self.decoder3(d4)
        e3 = self.Att3(g=d3, x=m3)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Conv3(d3)

        d2 = self.decoder2(d3)
        x2 = self.Att2(g=d2, x=m2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Conv2(d2)

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)