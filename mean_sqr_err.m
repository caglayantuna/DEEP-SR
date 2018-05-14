function Y = mean_sqr_err(X, c, dzdy)
%Mean Square Error Summary of this function goes here
%   Detailed explanation goes here

assert(numel(X) == numel(c));

d = size(X);

assert(all(d == size(c)));

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    
    a = abs(X-c).^2;
    Y = sum(a(:));
    Y = Y / size(X,4);
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    Y = +((X-c))*dzdy;
    
end

end