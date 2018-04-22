clear, close all
load model_brand_idx
% load model_name_3d
% load bbox_syn3
load model_3d_info_ds
load test_images_3d
len = length(test_images);
realDir ='/home/ljyang/work/data/CompCars/image/';
labDir = '/home/ljyang/work/data/CompCars/label/';
% synDir ='/home/ljyang/work/alignment/3d_models/car_models/obj_merge/';
savDir ='/home/ljyang/work/data/3d_test/';
% train_filename = ['../lists/transfer_train'];
test_filename = ['../lists/transfer_test_sr'];
% val_filename= ['../lists/transfer_val'];
% f_train = fopen(train_filename,'w');
f_test = fopen(test_filename,'w');
% f_val = fopen(val_filename,'w');
pad = 0.07;
param_dim = 10;
h_space = 6;
v_space = 6;
r_space = 6;
theta_n_d = 30;
phi_n_d =30;
% view_d_proj = compute_proj(h_space,v_space,v_max,r_space,r_max);
for i=1:len
    
        
       id = test_images(i).model_3d_id;
       shape = model_3d_info(id).shape_polar;
       shape_ds = imresize(shape,[theta_n_d,phi_n_d],'bilinear');
       shape_ds  =shape_ds(:) - polar_m';
%         param_3d = test_images(i).shape_param;
        
        p = test_images(i).path;
        label = textread([labDir,p(1:end-4),'.txt'],'%d');

%         im =imread([realDir,p]);
%         bbox = label(3:6);
%         bbox_pad = gen_bbox_pad(im,bbox,pad,0);
%         im_crop = im(bbox_pad(2):bbox_pad(4),bbox_pad(1):bbox_pad(3),:);
%                     imshow(im_crop);pause;
%         im_crop = bbox_pad_crop(im,bbox,pad);
        pos = strfind(p,'/');
        im_fd = p(1:pos(3));
        if ~exist([savDir,im_fd],'dir')
            mkdir([savDir,im_fd]);
        end
        im_path = [savDir,p];
        %im_path_mirror = [savDir,p(1:end-4),...
         %   '_m.jpg'];
%         imwrite(im_crop,im_path);
        %imwrite(im_crop(:,end:-1:1,:),im_path_mirror);
        %original image
        fprintf(f_test,'%s %d ',im_path,0);  
%         fprintf(f_test,'%d ',0);%dummy class label
        for d = 1:theta_n_d*phi_n_d
            fprintf(f_test,'%.3f ',shape_ds(d));
        end
  
        fprintf(f_test,'-1 -1 -1 ');
        
        fprintf(f_test,'%f\n',log(test_images(i).focal));
        %mirrored image
                
        
        
    
end
fclose(f_test);
% fclose(f_test);