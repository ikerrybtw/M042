function [ output_args ] = EM_labelf( no )
% generate labels using EM / MoG with 2 mixtures, adapted from script
% EM_label
%   Detailed explanation goes here
options = statset('MaxIter',1000);
% read in TP53 cell features
cellY = importdata(['TextFiles\TP53\TP53_',num2str(no),'.txt']);
Y=cellY.data;
% eliminate some cells with NaN features or too many zeros
% assumption is these are not real cells
ll=find(isnan(Y));
[m,n]=size(Y);
A=ones(m,n);
A(ll)=0;
kk=sum(A')';
temp=Y(kk==72,:);
Y=temp;
% we played around with col to find cell distribution that best matches
% Carlos' feedback, so this can be changed to only 56, or [56,60] or
% something else etc.
% we can also try out PCA on all custom intensity features and reduce them
% to 1-3 dimensions
col=[53,56,60];
data=Y(:,col);
[n,m]=size(data);
% fit gaussian mixture
obj=fitgmdist(data,2,'Options',options);
% get posterior probabilities, we save these for each sample
P=posterior(obj,data);

% show=Y(:,56);
save P
end

