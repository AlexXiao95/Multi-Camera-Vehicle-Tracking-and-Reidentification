clear,close all,clc
cls = 400;
points = rand(cls,1000);
points = bsxfun(@rdivide, points, sum(points,2));
sim = points*points';
sim_nodiag = sim;
sim_nodiag(1:cls+1:end) = 0;
[~,sim_cls] = max(sim_nodiag,[],1);

sim_match = sim_cls(sim_cls) == [1:cls];
sim_ratio = sum(sim_match / cls)
% S = bsxfun(@rdivide, S, sqrt(sum(S.^2,1)));