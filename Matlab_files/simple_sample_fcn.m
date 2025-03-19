function [out] = simple_sample_fcn(x,param)
%Get all the parameters out of the vector 
i=1;len=1; %len is the length of the next parameter
l=param(i:i+len-1);

i=i+len;len=1;
n=param(i:i+len-1);

i=i+len;len=1;
N=param(i:i+len-1);

i=i+len;len=l*n;
theta=reshape(param(i:i+len-1),l,n);

i=i+len;len=l;
tau=param(i:i+len-1);

i=i+len;len=N*n;
X=reshape(param(i:i+len-1),n,N);

i=i+len;len=1;
factor_cov=param(i:i+len-1);

i=i+len;len=2;
hypcov=param(i:i+len-1);

i=i+len;len=l;
c=param(i:i+len-1)';

i=i+len;len=N;
temp=param(i:i+len-1);

%Computing the sample
% This would be you dH(p,q)
out=(c*cos(theta*x'+tau))'+kernel_se(x',X,factor_cov,hypcov)'*temp;

end

