clear all;
close all;
net=load('network5.mat');
net=net.ans;
addpath testdata;
im=double(imread('multi1.tif'));
% im=double(im.c(1:500,1:500,1:4));

up_scale=2;
[im_b,v1,v2]=Ihs_forward(im);
im_l = imresize(im_b, 1/up_scale, 'bicubic');
im_b2 = imresize(im_l, up_scale, 'bicubic');

meanimb=mean2(im_b2);
im_b2=im_b2/4095;
im_b2=im_b2-meanimb;



weights_conv1=double(net.layers{1}.weights{1});
biases_conv1=double(net.layers{1}.weights{2});
weights_conv2=double(net.layers{3}.weights{1});
biases_conv2=double(net.layers{3}.weights{2});
weights_conv3=double(net.layers{5}.weights{1});
biases_conv3=double(net.layers{5}.weights{2});

weights_conv1=reshape(weights_conv1,9,9,64);
weights_conv2=reshape(weights_conv2,8,8,32);
weights_conv3=reshape(weights_conv3,5,5,32);
biases_conv1=reshape(biases_conv1,64,1);
biases_conv2=reshape(biases_conv2,32,1);

[conv1_patchsize,conv1_patchsize,conv1_filters] = size(weights_conv1);
[conv2_patchsize,conv2patchsize,conv2_filters] = size(weights_conv2);
[conv3_patchsize,conv3_patchsize,conv3_filters] = size(weights_conv3);
[hei, wid] = size(im_b2);
%conv1
conv1_data = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data(:,:,i) = imfilter(im_b2, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data(:,:,i) = max(conv1_data(:,:,i) + biases_conv1(i), 0);
end
%conv2
conv2_data = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    conv2_data(:,:,i) = conv2_data(:,:,i) + imfilter(conv1_data(:,:,i), weights_conv2, 'same', 'replicate');
    conv2_data(:,:,i) = max(conv2_data(:,:,i) + biases_conv2(i), 0);
end
%conv3
conv3_data = zeros(hei, wid);
for i = 1 : conv3_filters
%     conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data(:,:) = conv3_data(:,:) + imfilter(conv2_data(:,:,i), weights_conv3, 'same', 'replicate');
end
% SRCNN reconstruction
im_h = conv3_data(:,:) + biases_conv3;
% 
im_h1=im_h*4095;
im_h2=im_h1+meanimb;
im_b2=im_b2*4095;
im_b2=im_b2+meanimb;

result=Ihs_reverse(im_h2(:,1:end-1),v1,v2);
im_b_color=Ihs_reverse(im_b2(:,1:end-1),v1,v2);
figure;imshow(uint16(20*im(:,:,1:3)));
figure;imshow(uint16(80*im_b_color));
figure;imshow(uint16(result/10000));

figure;imshow(uint16(im_h2/100000));

