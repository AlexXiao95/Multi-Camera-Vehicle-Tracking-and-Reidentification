function [feature_d, Vp, f_m] = my_pca(feature,dim)

f_m = mean(feature,1);
feature = bsxfun(@minus,feature,f_m);
C = feature' * feature;

[V,D]= eig(C);
[d,idx] = sort(diag(D),'descend');
%d = diag(D);

sum(d(1:dim))/sum(d)
% D = diag(d(idx));
V = V(:,idx);



%check the order of D
Vp = V(:,1:dim);
feature_d = feature * Vp;
end