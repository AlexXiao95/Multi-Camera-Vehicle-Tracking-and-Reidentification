clear,close all
model ='alexnet';
score_model = load(['/home/ljyang/work/data/matlab/',model,...
    '_finetune_sv_car2_281.mat']);
% score_color = load('/home/ljyang/work/data/score/matlab/googlenet_finetune_sv_car_281_color.mat');
[paths, models, colors] = textread('test_sv_car_281','%s %d %d');

sample_n = length(colors);
score_model = score_model.feats(1:sample_n,:);
[~,model_pred] = max(score_model,[],2);
model_match = (model_pred == models+1);
% score_color = score_color.feats(1:sample_n,:);
load test_n
% [~,ia,ic] = unique(models);
model_n = length(test_n);
accu = zeros(model_n,1);
test_n_accu = [0;cumsum(test_n)];
for i=1:model_n
%     accu(i) = mean(model_pred(test_n_accu(i)+1:test_n_accu(i+1))==i);
 accu(i) = mean(model_match(test_n_accu(i)+1:test_n_accu(i+1)));
end
mean(accu)
    