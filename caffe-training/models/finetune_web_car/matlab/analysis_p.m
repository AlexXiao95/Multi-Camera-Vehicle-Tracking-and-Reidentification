clear,close all,clc
network ='overfeat';
dataDir = '/home/ljyang/work/data/matlab/';
prefix = [network,'_finetune_web_car_train_iter_'];
load(['mixing_matrix_', network]);
cls = 431;
mixing_nodiag = mixing_matrix;
mixing_nodiag(1:cls+1:end) = 0;
mix_cls = zeros(cls,1);
[mix_cls_prob,mix_cls] = max(mixing_nodiag,[],2);
mix_diag = diag(mixing_matrix);
mix_mask = mix_diag < mean(mix_diag);
mix_cls_easy = (mix_mask & mix_cls);
%%
% sum(mix_mask)
for i=1:10
iter = i*1000;
sp = 37917;

score = load([dataDir, prefix, num2str(iter)]);
sc = score.feats(1:sp,:);
sc = bsxfun(@minus, sc, max(sc,[],2));
sc = exp(sc);
sc = bsxfun(@rdivide, sc, sum(sc,2)); 
S = sc'*sc / (sp/cls);
S_diag = diag(S);
sim_cls = zeros(cls,1);

S_nodiag = S;
S_nodiag(1:cls+1:end) = 0;

[sim_v,sim_cls] = max(S_nodiag,[],2);
sim_mask = sim_v > 0.001;
sim_match = sim_cls(sim_cls) == [1:cls]';
sim_ratio(i) = sum(sim_match / cls);
% sum((sim_cls == mix_cls) & mix_mask)
sum((sim_cls == mix_cls) & sim_mask)
sum(sim_mask)
% S = bsxfun(@rdivide, S, sqrt(sum(S.^2,1)));
end