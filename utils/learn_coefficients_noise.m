function S = learn_coefficients_noise(B, X,  lambda2)
BtB = B'*B;
BtX=B'*X;
  S=(BtB+lambda2*eye(size(B,2)))\(BtX);
S(isnan(S))=0;
return;