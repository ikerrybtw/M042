function [ coeff, score, latent, explain, mu2, mu, sigma] = PCA_custom(data, rank, version)
% X_pca = (score * coeff' + repmat(mu2, m, 1)) .* repmat(sigma, m, 1) +
% repmat(mu, m ,1);
if version == 0 % matches what we learned in class 
    [m,n]=size(data);
    mu=nanmean(data);
    data=data-repmat(mu, m, 1);
    vari=nanvar(data);
    sigma=sqrt(vari);
    data=data./repmat(sigma, m, 1);
    [coeff, score, latent, ~, explain, mu2] = pca(data, 'NumComponents', rank);
elseif version == 1 % seems to capture variance more successfully
    [m,n]=size(data);
    mu2 = zeros(1,n);
    sigma = ones(1,n);
    [coeff, score, latent, ~, explain, mu] = pca(data, 'NumComponents', rank);
end
end

