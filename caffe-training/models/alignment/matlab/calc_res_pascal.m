clear,close all
sp=932;
dataDir = '/home/ljyang/work/data/matlab/';
model = 'alexnet_domain2_p_pascal';
attr = {'shape','view_h','view_v','view_r','focal'};
load gt_pose_pascal
imDir = '/home/ljyang/work/data/test_pascal/';
% labelDir = '/home/ljyang/work/data/CompCars/label/';

h_space = 6;
v_space = 6;
r_space = 6;
r_max = 6;
pred_pose = zeros(sp,3);
for i=2:4
    f = load([dataDir,model,'_',attr{i}]);
    preds = double(f.feats(1:sp,:));
    
    [~,pred_view] = max(preds,[],2);
    
    if i==2
        pred_pose(:,1) = 360-(pred_view-1) * h_space;%pascal version
    elseif i==3
        pred_pose(:,2) = (pred_view-1) * v_space;
    elseif i==4
        pred_pose(:,3) = (pred_view-1) * r_space - r_max;
    end
end
pose_diff = abs(gt_pose - pred_pose);
pose_diff(:,1) = min(pose_diff(:,1),360-pose_diff(:,1));
% pose_error_tot = 
% gedeosic distance of R_gt and R_pred
display=1;
for i=1:sp
    
    
    R_gt = compute_rotation_matrix(gt_pose(i,:));
    R_pred = compute_rotation_matrix(pred_pose(i,:));
    pose_error(i) = compute_R_distance(R_gt,R_pred)*180/pi;
    if display
        
        
    end
end

%%
pose_medErr = median(pose_error);
pose_ac = sum(pose_error<30)/sp;