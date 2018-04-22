clear,close all
dataDir = '/home/ljyang/work/data/matlab/';
model = 'alexnet_domain2_p';
attr = {'shape','view_h','view_v','view_r','focal'};
load test_images_3d
load model_3d_info_ds
imDir = '/home/ljyang/work/data/CompCars/image/';
labelDir = '/home/ljyang/work/data/CompCars/label/';
sp = 811;%size of testing data
% test_list = '/home/ljyang/work/caffe/caffe-multigpu/models/alignment/lists/transfer_test';
% f=fopen(test_list);
% textread(f,'%s %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
% gt_shape = [test_images.shape_param];
% shape_len = length(test_images(1).shape_param);
% gt_shape =reshape(gt_shape,shape_len,sp)';
gt_pose = [test_images.pose]';
gt_focal = [test_images.focal]';
attr_type = [1,0,0,0,1];%1 for continuous, 0 for discrete

h_space = 6;
v_space = 6;
r_space = 6;
r_max = 6;
pred_pose = zeros(sp,3);
for i=1:5
    f = load([dataDir,model,'_',attr{i}]);
    preds = double(f.feats(1:sp,:));
    if attr_type(i)==0
        [~,pred_view] = max(preds,[],2);
    end
    if i==2
        pred_pose(:,1) = (pred_view-1) * h_space;
    elseif i==3
        pred_pose(:,2) = (pred_view-1) * v_space;
    elseif i==4
        pred_pose(:,3) = (pred_view-1) * r_space - r_max;
    elseif i==1
        pred_shape = preds;
    elseif i==5
        pred_focal = preds;
    end
end
% shape_mse = mean(sum((pred_shape - gt_shape).^2,2));
% shape_diff = sum((pred_shape - gt_shape).^2,2);
pose_error = abs(gt_pose - pred_pose);
pose_error(:,1) = min(pose_error(:,1),360-pose_error(:,1));
%mean angular error
pose_mae = mean(pose_error,1);
for i=1:sp   
    R_gt = compute_rotation_matrix(gt_pose(i,:));
    R_pred = compute_rotation_matrix(pred_pose(i,:));
    pose_geod_err(i) = compute_R_distance(R_gt,R_pred);
end
pose_medErr = median(pose_geod_err)*180/pi;
pose_ac = sum(pose_geod_err<=pi/6)/sp;
%mean ratio error for focal 
focal_mre = mean(exp(abs(log(gt_focal)-pred_focal))-1);
%full error for 3d model


%display
theta_n = 90;
phi_n = 90;
theta_n_d = 30;
phi_n_d = 30;
shape_se_polar = zeros(sp,1);
shape_se_ds = zeros(sp,1);
shape_se_m = zeros(sp,1);
display = 0;
for i=1:sp
    p = test_images(i).path;
    im =imread([imDir,p]);
    [l,w,~] = size(im);
    pose = test_images(i).pose;
    focal = test_images(i).focal;
   
    polar_recon = polar_m' + Vp * pred_shape(i,:)';
    polar_recon = reshape(polar_recon,theta_n_d,phi_n_d);
    polar_gt = imresize(model_3d_info(test_images(i).model_3d_id).shape_polar,[theta_n_d,phi_n_d],'bilinear');
%     polar_recon_us = imresize(polar_recon,[theta_n,phi_n],'bilinear');
    diff = (polar_recon - polar_gt);
    diff_pca = (pred_shape(i,:) - model_3d_info(test_images(i).model_3d_id).shape_param_ds);
    diff_m = model_3d_info(test_images(i).model_3d_id).shape_param_ds;
    shape_se_polar(i) = sum(diff(:).^2);
    shape_se_ds(i) = sum(diff_pca(:).^2);
    shape_se_m(i) = sum(diff_m(:).^2);
    if display
         figure(1);
%          set(gcf, 'Renderer','OpenGL');    
        imshow(im);
        title(sprintf('gt pose %d %d %d gt focal %f pred pose %d %d %d pred focal %f',...
            pose(1),pose(2),pose(3),focal,pred_pose(i,1),pred_pose(i,2),...
            pred_pose(i,3),exp(pred_focal(i))));
        label = textread([labelDir,p(1:end-3),'txt'],'%d');
        bbox = label(3:6);
        [x3d,faces] = get_polar_mesh(polar_recon);
        x2d = project_3d(x3d,pred_pose(i,:),exp(pred_focal(i)),bbox);

        hold on;
    %     figure;
    %     scatter(x2d(:,1),x2d(:,2),'filled');
        patch('vertices', x2d, 'faces', faces, ...
                'FaceColor', 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
%         set(gca,'xlim',[-1 w+1])
%         set(gca,'ylim',[-1 l+1])
        axis off;
        hold off;

        figure(2);
        display_polar_mesh(polar_recon);
        title(sprintf('shape diff %f', shape_diff(i)));
        pause;
    end
end
shape_mse_polar = mean(shape_se_polar);
shape_mse_ds = mean(shape_se_ds);
shape_se_m = mean(shape_se_m);