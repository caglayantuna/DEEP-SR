clear all;

close all;

[data,label]=generate_data2();
%gathering data

% Normalization
normdata=data/4095;
data=single(normdata);


normlabel=label/4095;
label=single(normlabel);

network1=load('network0308.mat');
network1=network1.net;

network1.layers{end}=struct('type','softmax');

sonuc=vl_simplenn(network1,data(:,:,:,6021));
data1=data(:,:,:,6021);
label1=label(:,:,:,6021);

% figure;imshow(uint16(1000000*sonuc(6).x));
% figure;imshow(uint16(1000000*data1));
% figure;imshow(uint16(1000000*label1));

figure; imshow(uint16(100*4095*sonuc(6).x + mean(data,4)));title('sr');
figure; imshow(uint16(100*4095*data1 + mean(data,4)));title('data');
figure; imshow(uint16(100*4095*label1+ mean(data,4)));title('gt');


