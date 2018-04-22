clear,close all,clc
% pca->joint bayesian->test
network='googlenet';
featDir = '/home/ljyang/work/data/matlab/';
f =load([featDir,network,'_finetune_verification_train']);

% Read the image list file used by caffe
[paths,labels] = textread('../lists/verification_train','%s %d');
n =length(labels);
feature = f.feats(1:n,:);
dim = 200;
[feature_d, Vp, f_m] = my_pca(feature,dim);
model = trainBayesian2(feature_d,labels);
save([network,'_jb_model'], 'model');
%%
clc
load([network,'_jb_model']);
mode={'easy','medium','hard'};
n_match = 10000;
n_unmatch=n_match;
n_v = (n_match+n_unmatch)*2;

for m=1:3
    ft = load([featDir,network,'_finetune_verification_test_',mode{m}]);
   
    feature_t = ft.feats(1:n_v,:);
    feature_t = bsxfun(@minus, feature_t, f_m);
    feature_td = feature_t * Vp;

    score_match = verifyBayesian(model,feature_td(1:n_match,:),...
        feature_td(2*n_match+1:3*n_match,:));
    score_unmatch = verifyBayesian(model,feature_td(n_match+1:2*n_match,:),...
        feature_td(3*n_match+1:4*n_match,:));


    % draw curve
    thresh=-50:0.1:50;
    len = length(thresh);
    fp{m}=zeros(len,1);
    recall{m}=zeros(len,1);
    for i=1:length(thresh)
        recall{m}(i) = sum(score_match > thresh(i))/n_match;
        fp{m}(i) = sum(score_unmatch > thresh(i))/n_unmatch;
    end
    
    
    % semilogx(fp,recall);
    close all;
    plot(fp{m},recall{m});pause;
    th=-10;

    ac = (sum(score_match > th) + sum(score_unmatch <=th))/(n_match +n_unmatch)

end
save([network,'_cnn_jb_roc'], 'fp', 'recall');