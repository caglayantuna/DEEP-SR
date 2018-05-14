function net=deepsupernet2()

f = 0.001 ;
initialBias=0.001;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(9, 9, 1, 64, 'single'), initialBias * zeros(1, 64, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 4);
net.layers{end}.learningRate = 0.0001 * ones(1, numel(net.layers{end}.weights));
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1, 1, 64, 32, 'single'), initialBias * zeros(1, 32, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0);
net.layers{end}.learningRate = 0.0001 * ones(1, numel(net.layers{end}.weights));
net.layers{end+1} = struct('type', 'relu') ;
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5, 5, 32, 1, 'single'), initialBias * zeros(1, 1, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 2);
net.layers{end}.learningRate = 0.00001 * ones(1, numel(net.layers{end}.weights));
net.meta.inputSize = [32 32 1 1];

net.layers{end+1} = struct('type', 'RMSEloss');
net = vl_simplenn_tidy(net) ;

end
