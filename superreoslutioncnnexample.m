clear all;
close all;
M=5000;
data=zeros(140,140,1,M);
%gathering data
data=data/255;
imdb.images.data = imresize(imresize(data,0.25),4);
imdb.images.data_mean = mean(data,4);
imdb.images.labels = data ;
imdb.images.set=[ones(1,48*M/50), 2*ones(1,1*M/50) , 3*ones(1,1*M/50)];

imdb.images.data = single(imdb.images.data);
imdb.images.data_mean = single(imdb.images.data_mean);
imdb.images.labels = single(imdb.images.labels);
imdb.images.data = bsxfun(@minus, imdb.images.data, imdb.images.data_mean ) ;
imdb.images.labels = bsxfun(@minus, imdb.images.labels, imdb.images.data_mean ) ;

lr=0.001;
initialBias=0.001;
filters_num1=64;
filters_num2=32;

% Define network architecture
net.layers = {} ;

k1=0.01;

net.layers{end+1} = struct('type', 'conv', 'filters', k1 * randn(9, 9, 1, filters_num1, 'single'), 'biases', zeros(1, filters_num1, 'single'), 'stride', 1, 'pad', 2, 'filtersLearningRate', lr, 'biasesLearningRate', lr, 'filtersWeightDecay', 1, 'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', 'filters', k1 * randn(1, 1, filters_num1, filters_num2, 'single'), 'biases', initialBias*ones(1, filters_num2, 'single'), 'stride', 1, 'pad', 2, 'filtersLearningRate', lr, 'biasesLearningRate', lr, 'filtersWeightDecay', 1, 'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', 'filters', k1 * randn(5, 5, filters_num2, 1, 'single'), 'biases', initialBias*zeros(1, 1, 'single'), 'stride', 1, 'pad', 2, 'filtersLearningRate', lr, 'biasesLearningRate', lr, 'filtersWeightDecay', 1, 'biasesWeightDecay', 0) ;

net.layers{end+1} = struct('type', 'softmaxloss') ;
net_init=net;
net.meta.inputSize = [33 33 1 1];
opts.train.batchSize = 10 ;
opts.train.numEpochs = 10 ;
opts.train.continue = false ;
% opts.train.useGpu = false ;
opts.train.learningRate = [0.001*ones(1, 12) 0.0001*ones(1,6) 0.00001] ;

[net, info] = cnn_train(net, imdb, @getBatch, ...
opts.train, ...
'val', find(imdb.images.set == 2)) ;

test_set=imdb.images.data(:,:,:,(imdb.images.set==3));
test_label=imdb.images.labels(:,:,:,(imdb.images.set==3));
test_set=test_set(:,:,:,1:10);
test_label=test_label(:,:,:,1:10);

stagedPredictions = vl_simplenn(net, test_set);
y_hat = stagedPredictions(end-1).x;

figure; imshow(y_hat(:,:,:,1) + imdb.images.data_mean);title('sr');
figure; imshow(test_set(:,:,:,1) + imdb.images.data_mean);title('data');
figure; imshow(test_label(:,:,:,1)+ imdb.images.data_mean);title('gt');

my_err=(test_label-y_hat).^2;
my_err=sum(my_err(:));

init_err=(test_label - test_set).^2;
init_err=sum(init_err(:));

disp(sprintf('my err is %f and init_err is %f',my_err,init_err));