clear,close all,clc
network ='alexnet';
dataDir = '/home/ljyang/work/data/matlab/';
prefix = [network,'_finetune_web_car_view_1_train_iter_'];
[paths, models, views] = textread('../lists/classification_view_train','%s %d %d');
[~,ia,ic] = unique(models);
model_n = length(ia);
ia = [0;ia];
sp = 37917;
view_n = 5;
% load(['mixing_matrix_', network]);
% cls = 431;
% mixing_nodiag = mixing_matrix;
% mixing_nodiag(1:cls+1:end) = 0;
% mix_cls = zeros(cls,1);
% [mix_cls_prob,mix_cls] = max(mixing_nodiag,[],2);
% mix_diag = diag(mixing_matrix);
% mix_mask = mix_diag < mean(mix_diag);
% mix_cls_easy = (mix_mask & mix_cls);
%%
% sum(mix_mask)
for i=1:10
iter = i*1000;
score = load([dataDir, prefix, num2str(iter)]);
sc = score.feats(1:sp,:);
% sc = bsxfun(@minus, sc, max(sc,[],2));
% sc = exp(sc);
% sc = bsxfun(@rdivide, sc, sum(sc,2)); 
% S = sc'*sc / (sp/cls);
% S_diag = diag(S);
% sim_cls = zeros(cls,1);
[~,model_sort] = sort(sc,2,'descend');
model_pred = model_sort(:,1);
model_correct = (model_pred==models+1);
model_top5 = any(model_sort(:,1:5)==repmat(models+1,1,5),2);
ac_model_view = zeros(model_n, view_n);
t5_model_view = zeros(model_n, view_n);
for i=1:model_n
    ids = [ia(i)+1:ia(i+1)];
    for j=1:view_n
        sample_v = sum(views(ids)==j-1);
        ac_model_view(i,j) = sum(views(ids)==j-1 & model_correct(ids))/sample_v;
%         t5_model_view(i,j) = sum(views(ids)==j-1 & model_top5(ids))/sample_v;
    end
end
var_model = std(ac_model_view,0,2);
% hist(var_model,20);pause;
mean(ac_model_view)

% mean(t5_model_view)
% S_nodiag = S;
% S_nodiag(1:cls+1:end) = 0;
% 
% [sim_v,sim_cls] = max(S_nodiag,[],2);
% sim_mask = sim_v > 0.001;
% sim_match = sim_cls(sim_cls) == [1:cls]';
% sim_ratio(i) = sum(sim_match / cls);
% % sum((sim_cls == mix_cls) & mix_mask)
% sum((sim_cls == mix_cls) & sim_mask)
% sum(sim_mask)
% S = bsxfun(@rdivide, S, sqrt(sum(S.^2,1)));
end