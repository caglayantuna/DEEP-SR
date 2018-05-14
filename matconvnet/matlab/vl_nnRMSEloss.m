function Y = vl_nnRMSEloss(X,c,dzdy)

c=imresize(c,[32 32],'bicubic');

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    
    a = abs(X-c).^2;
    Y = sum(a(:));
    Y = Y / size(X,4);
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    Y = +((X-c))*dzdy;
    
end
end