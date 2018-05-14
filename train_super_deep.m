function model=train_super_deep();
addpath matconvnet/matlab
run vl_setupnn
net=deepsupernet2();
imdb=imdbdeep();
[data,label]=generate_data();
%gathering data

% Normalization
% normdata=data/255;
data=single(data);
label=single(label);
% normlabel=label/255;
net.layers{end}.class = label ;
%IMDB
imdb.images.data_mean =mean(data,4);
imdb.images.label_mean =mean(label,4);
imdb.images.set=[ones(1,20000), 2*ones(1,2000),3*ones(1,227)]; 
imdb.images.labels=label ;
imdb.images.data =data;
imdb.images.data = bsxfun(@minus, imdb.images.data, imdb.images.data_mean ) ;
imdb.images.labels = bsxfun(@minus, imdb.images.labels, imdb.images.label_mean ) ;
%Train Opts
TrainOpts.batchSize = 256;
TrainOpts.numEpochs = 1000;
TrainOpts.continue = true;
TrainOpts.errorFunction = 'RMSE';
TrainOpts.momentum = 0.9;
TrainOpts.plotDiagnostics = false;
% TrainOpts= vl_argparse(TrainOpts, varargin) ;

model= cnn_train(net, imdb, @getBatch, TrainOpts) ;
end
