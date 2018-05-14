function result=Ihs_reverse(I,v1,v2)
result(:,:,1)=I+((-1/sqrt(6))*v1)+((3/sqrt(6))*v2);
result(:,:,2)=I+((-1/sqrt(6))*v1)+((-3/sqrt(6))*v2);
result(:,:,3)=I+((2/sqrt(6))*v1);
end