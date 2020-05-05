function logp = logGammaPdf(X, a, b)
% logp(i) = log p(X(i) | a, b) 
% a is the shape,
% b is the rate, i.e. 1/scale

% This file is from pmtk3.googlecode.com



logZ = gammaln(a) - a.*log(b);
logp = (a-1).*log(X) - b.*X - logZ;
logp = logp(:); 

end
