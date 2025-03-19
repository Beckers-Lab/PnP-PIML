function [samples] = create_sample_object(gp,data,n_cos, n_samples)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

X_data=data(:,1);
Y_data=data(:,2);
Z_data=data(:,3);

hyp.lik=gp.Sigma^2; %Sigma
hyp.cov=zeros(3,1); %sigma_f and 2 x sigma_l (two dimensions)
hyp.cov(1)=gp.KernelInformation.KernelParameters(2); %Sigma_f
hyp.cov(2)=gp.KernelInformation.KernelParameters(1)*2; %Sigma_l
hyp.cov(3)=gp.KernelInformation.KernelParameters(1)*2; %Sigma_l

sample_obj.X=[X_data(:),Y_data(:)]; %input training data
sample_obj.Y=Z_data(:); %output training data
sample_obj.l=n_cos; %higher gives better approximation
sample_obj.factor = 1; %how much std
sample_obj.hyp=hyp; %hyperparameters

kernel=@(hyp,x1,x2) kernel_se(x1,x2,hyp.cov(1),hyp.cov(2:end));

sample_obj.K=kernel(hyp, sample_obj.X',sample_obj.X')+eye(length(sample_obj.X))*sample_obj.hyp.lik;
sample_obj.ndata=length(sample_obj.Y);
sample_obj.invK=inv(sample_obj.K);

% create necessary parameter vector used for all samples



samples=[];
for i=1:n_samples
% Here we start to create one specific sample. You need to repeat this part
% for another sample
    sample_obj.tau=2*pi*rand(sample_obj.l,1);
    sample_obj.theta=randn(sample_obj.l,length(sample_obj.hyp.cov(2:end)))./repmat(sample_obj.hyp.cov(2:end)',sample_obj.l,1);
    sample_obj.Ysigma=sample_obj.Y-sample_obj.hyp.lik*randn(sample_obj.ndata,1);
    sample_obj.w=randn(sample_obj.l,1);
    sample_obj.temp=sample_obj.invK*(sample_obj.Ysigma-prior_sample(sample_obj.X',sample_obj));

    % create necessary parameter vector for specific sample
    clear param
    param{1}=sample_obj.l;
    param{2}=size(sample_obj.X,2);
    param{3}=size(sample_obj.X,1);
    param{4}=sample_obj.theta;
    param{5}=sample_obj.tau;
    param{6}=sample_obj.X';
    param{7}=sample_obj.factor*sample_obj.hyp.cov(1);
    param{8}=sample_obj.hyp.cov(2:end);
    param{9}=sqrt(2*sample_obj.factor*sample_obj.hyp.cov(1)/sample_obj.l)*sample_obj.w';
    param{10}=sample_obj.temp;

%Coll all parameters in a big vector
params=[];
for i=1:length(param)
   params=[params;param{i}(:)];
end
samples=[samples,params];
end

function func = prior_sample(x,sample_obj)
    n=size(x,2);
    tau=repmat(sample_obj.tau,1,n);
    func=(sqrt(2*sample_obj.factor*sample_obj.hyp.cov(1)/sample_obj.l)*sample_obj.w'*cos((sample_obj.theta)*x+tau))';
end

end

