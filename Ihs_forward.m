function [I v1 v2]=Ihs_forward(im)

I=(im(:,:,1)/3)+(im(:,:,2)/3)+(im(:,:,3)/3);
v1=((-1/sqrt(6))*im(:,:,1))+((-1/sqrt(6))*im(:,:,2))+((2/sqrt(6)))*im(:,:,3);
v2=((1/sqrt(6))*im(:,:,1))+((-1/sqrt(6))*im(:,:,2));

end
