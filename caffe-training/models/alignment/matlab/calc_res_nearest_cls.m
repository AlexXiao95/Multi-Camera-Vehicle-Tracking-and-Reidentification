clear,close all
dataDir = '/home/ljyang/work/data/matlab/';
model = 'alexnet_c_p_f';
load train_shape_param
attr = {'class','view_h','view_v','view_r','focal'};
load test_images_3d
load model_3d_param
imDir = '/home/ljyang/work/data/3d_test/';
sp = 811;%size of testing data
% test_list = '/home/ljyang/work/caffe/caffe-multigpu/models/alignment/lists/transfer_test';
% f=fopen(test_list);
% textread(f,'%s %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
gt_shape = [test_images.shape_param];
shape_len = length(test_images(1).shape_param);
gt_shape =reshape(gt_shape,shape_len,sp)';
gt_pose = [test_images.pose]';
gt_focal = [test_images.focal]';
attr_type = [0,0,0,0,1];%1 for continuous, 0 for discrete

h_space = 6;
v_space = 6;
r_space = 6;
r_max = 6;
pred_pose = zeros(sp,3);
for i=1:5
    f = load([dataDir,model,'_',attr{i}]);
    preds = f.feats(1:sp,:);
    if attr_type(i)==0
        [~,pred_id] = max(preds,[],2);
    end
    if i==2
        pred_pose(:,1) = (pred_id-1) * h_space;
    elseif i==3
        pred_pose(:,2) = (pred_id-1) * v_space;
    elseif i==4
        pred_pose(:,3) = (pred_id-1) * r_space - r_max;
    elseif i==1
        %directly use shape of the predicted class
        pred_cls = pred_id;
        pred_shape = train_shape_param(pred_cls,:);
    elseif i==5
        pred_focal = preds;
    end
end
shape_mse = mean(sum((pred_shape - gt_shape).^2,2));
pose_error = abs(gt_pose - pred_pose);
pose_error(:,1) = min(pose_error(:,1),360-pose_error(:,1));
%mean angular error
pose_mae = mean(pose_error,1);
%mean ratio error for focal 
focal_mre = mean(exp(abs(log(gt_focal)-pred_focal))-1);
mean_shape = zeros(10,1);
shape_mse_mean = mean(sum((repmat(mean_shape',sp,1)-gt_shape).^2,2));

    
