clear,close all
network='overfeat';
score_model = load(['/home/ljyang/work/data/matlab/',network,'_finetune_web_car.mat']);
[paths, models] = textread('../lists/classification_test','%s %d');

sample_n = length(models);
score_model = score_model.feats(1:sample_n,:);
% load test_n
[~,ia,ic] = unique(models);
model_n = length(ia);
ia = [0;ia];
ac_model_avg = 0;
ac_model_t5 = 0;
[~,model_sort] = sort(score_model,2,'descend');
model_pred = model_sort(:,1);
model_correct = (model_pred==models+1);
model_top5 = any(model_sort(:,1:5)==repmat(models+1,1,5),2);
mixing_matrix = zeros(model_n,model_n);
for i=1:model_n
    ids = [ia(i)+1:ia(i+1)];
    sample_c = ia(i+1)-ia(i);
    ac_model_avg = ac_model_avg + sum(model_correct(ids))/sample_c;
    ac_model_t5 = ac_model_t5 + sum(model_top5(ids))/sample_c;
    for j=1:model_n
        mixing_matrix(i,j) = sum(model_pred(ids) == j)/sample_c;
    end
end
ac_model_avg = ac_model_avg/model_n;
ac_model_t5 = ac_model_t5/model_n;
imagesc(mixing_matrix)
%%
save(['mixing_matrix_',network],'mixing_matrix');
