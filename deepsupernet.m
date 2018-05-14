function net=deepsupernet()
net.layers = {} ;
filters_num1=64;
filters_num2=32;
lr=0.001;
initialBias=0.001;
k1=0.01;
net.layers{end+1} = struct('type', 'conv',...
                          'weights', {{k1 * randn(9, 9, 1, filters_num1,'single'),initialBias*zeros(1, filters_num1,'single')}},...
                          'stride',1, 'pad', 0, 'filtersLearningRate', lr, 'biasesLearningRate', lr, 'filtersWeightDecay', 1,...
                          'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', 'weights',{{k1 * randn(1, 1, filters_num1, filters_num2,'single'),initialBias*zeros(1, filters_num2,'single')}}, 'stride', 1, 'pad', 0, 'filtersLearningRate', lr, 'biasesLearningRate', lr, 'filtersWeightDecay', 1, 'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', 'weights', {{k1 * randn(5, 5, filters_num2, 1,'single'),initialBias*zeros(1, 1,'single')}}, 'stride', 1, 'pad', 0, 'filtersLearningRate', lr, 'biasesLearningRate', lr, 'filtersWeightDecay', 1, 'biasesWeightDecay', 0) ;

net.layers{end+1} = struct('type', 'softmax') ;
net.meta.inputSize = [32 32 1 1];
net = vl_simplenn_tidy(net);
% net.layers{end}.class=ones(32,32);
end
