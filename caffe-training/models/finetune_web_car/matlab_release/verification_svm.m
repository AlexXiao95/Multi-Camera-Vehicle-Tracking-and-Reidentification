clear,close all,clc
%pca->svm->test
network='googlenet';
featDir = '/home/ljyang/work/data/matlab/';
f =load([featDir,network,'_finetune_verification_train']);

[paths,labels] = textread('../lists/verification_train','%s %d');
n =length(labels);
feature = f.feats(1:n,:);
[~,ia,ic] = unique(labels);
im_n_v = ia - [0;ia(1:end-1)];

usepca=1;
% normalize feature to 0~1
if usepca
    dim = 200;
    [feature, Vp, f_m] = my_pca(feature,dim);
    f_max = max(abs(feature(:)));
    feature=feature/f_max;
else
%no pca
    f_max = max(abs(feature(:))); 
    feature = feature./f_max;
end
%%
n_match = 50000;
n_unmatch= 50000;
[pairs_match, pairs_unmatch] = gen_verif_pairs(im_n_v, n_match, n_unmatch);




train_data =[feature(pairs_match(:,1),:),feature(pairs_match(:,2),:);...
    feature(pairs_unmatch(:,1),:),feature(pairs_unmatch(:,2),:)];
train_label=[ones(n_match,1);zeros(n_unmatch,1)];
%%
disp('start to train svm model');
ss=tic;

%polynomial kernel
model = svmtrain(train_label,double(train_data),'-s 0 -t 1 -g 1 -d 2 -r 0 -c 1 -h 0');
save([network,'_poly_svm_model'], 'model');
disp('finished!');toc(ss);
%%
load([network,'_poly_svm_model']);
mode={'easy','medium','hard'};
n_match = 10000;
n_unmatch=n_match;
n_v = (n_match+n_unmatch)*2;

for m=1:3
    ft = load([featDir,network,'_finetune_verification_test_',mode{m}]);
    feature_t = ft.feats(1:n_v,:);
    if usepca
        feature_t = bsxfun(@minus, feature_t, f_m);
        feature_t = feature_t*Vp;
        feature_t = feature_t / f_max;
    else
        % no pca
        feature_t = feature_t./f_max;
    end


    test_data = [feature_t(1:n_match,:),feature_t(2*n_match+1:3*n_match,:);...
                 feature_t(n_match+1:2*n_match,:),feature_t(3*n_match+1:4*n_match,:)];
    test_label = [ones(n_match,1);zeros(n_unmatch,1)];
    %[preds, ac, dec_values] = predict(test_label,sparse(test_data),model);
    [preds,ac,dec_values] = svmpredict(test_label,double(test_data),model);

    % draw curve
    %%
    thresh=-10:0.02:10;
    len = length(thresh);
    fp{m}=zeros(len,1);
    recall{m}=zeros(len,1);
    dec_pos=0;
    for i=1:length(thresh)
        recall{m}(i) = sum(dec_values(1:n_match) > thresh(i))/n_match;
        fp{m}(i) = sum(dec_values(n_match+1:n_match+n_unmatch) > thresh(i))/n_unmatch;
%         if dec_values(
    end
%     close all;
%     plot(fp{m},recall{m});pause;

end

save([network,'_cnn_svm_roc'], 'fp', 'recall');


