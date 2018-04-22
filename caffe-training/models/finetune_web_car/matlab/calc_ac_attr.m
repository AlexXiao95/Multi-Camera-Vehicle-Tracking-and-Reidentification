clear,close all,clc
network='alexnet';
load attr_m
% load attr_reg
label = cell(5,1);
[~,label{1},label{2}] = textread('../lists/test_attr_cont','%s %f %f');
[~,label{3},label{4},label{5}] = textread('../lists/test_attr_disc','%s %d %d %d');  


sp = length(label{1});%sum(im_n_v,1);

mean_tr(1) = mean(s_attr(:,1));
std_tr(1) = std(s_attr(:,1));
mean_tr(2) = mean(s_attr(:,2));
std_tr(2) = std(s_attr(:,2));

s_v_attr(:,1) = (s_v_attr(:,1) - mean_tr(1))/std_tr(1);
s_v_attr(:,2) = (s_v_attr(:,2) - mean_tr(2))/std_tr(2);

attr_cont_err = zeros(2,2);
scores = load(['/home/ljyang/work/data/matlab/',network,'_finetune_attr_cont']);
    for l=1:2
        
        score = scores.feats(1:sp,l);
        %gt_var = var(gt_label)
        
        attr_cont_err(l,1) = mean(abs(label{l}))*std_tr(l);
        attr_cont_err(l,2) = mean(abs(score - label{l}))*std_tr(l);

    end
attr_disc_ac = zeros(3,1);
for l=1:3
scores = load(['/home/ljyang/work/data/matlab/',network,'_finetune_attr_disc',num2str(l)]);
    
        
        score = scores.feats(1:sp,:);
        %gt_var = var(gt_label)
        [~,preds] = max(score,[],2);
        
        attr_disc_ac(l) = sum(preds-1==label{l+2})/sp;
    
end

save attr_res attr_cont_err attr_disc_ac