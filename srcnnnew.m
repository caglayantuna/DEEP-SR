function im_h= VDSR_Matconvnet(model,im_b)
load(model);

weights_conv1=single(net.layers{1}.weights{1});
biases_conv1=single(net.layers{1}.weights{2});
weights_conv2=single(net.layers{3}.weights{1});
biases_conv2=single(net.layers{3}.weights{2});
weights_conv3=single(net.layers{5}.weights{1});
biases_conv3=single(net.layers{5}.weights{2});

convfea = vl_nnconv(im_b,weights_conv1,biases_conv1,'Pad',0);
convfea2 = vl_nnrelu(convfea);
convfea3 = vl_nnconv(convfea2,weights_conv2,biases_conv2,'Pad',0);
convfea4 = vl_nnrelu(convfea3);
convfea5 = vl_nnconv(convfea4,weights_conv3,biases_conv3,'Pad',0);
im_h=convfea5;
end
