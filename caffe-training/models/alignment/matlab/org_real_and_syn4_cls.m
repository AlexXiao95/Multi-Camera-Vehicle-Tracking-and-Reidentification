clear, close all
load model_brand_idx
% load model_name_3d
load bbox_syn4
load model_3d_param
len = length(model_3d_param);
realDir ='/home/ljyang/work/data/CompCars/image/';
labDir = '/home/ljyang/work/data/CompCars/label/';
synDir ='/home/ljyang/work/alignment/3d_models/car_models/obj_merge2/';
savDir ='/home/ljyang/work/data/real_and_syn4/';
train_filename = ['../lists/transfer_train_cls'];
% test_filename = ['../lists/transfer_test'];
val_filename= ['../lists/transfer_val_cls'];
f_train = fopen(train_filename,'w');
% f_test = fopen(test_filename,'w');
f_val = fopen(val_filename,'w');
pad = 0.07;

syn_im = 720;
h_space = 6;
h_n = 360/h_space;
v_space = 6;
v_max =18;
v_n = (v_max/v_space)+1;
r_space = 6;
r_max = 6;
r_n = 2 * r_max/r_space+1;
% param_dim = 10;
% view_d_proj = compute_proj(h_space,v_space,v_max,r_space,r_max);
cls_id=-1;
train_shape_param = zeros(68,10);
for i=1:len
    if model_3d_param{i,3}==0 %for train
        
       cls_id = cls_id+1;
%         param_3d = model_3d_param{i,4};
        train_shape_param(cls_id+1,:) = model_3d_param{i,4};
        model_id = model_3d_param{i,2};
        make_id = model_brand_idx(model_id);
        year = dir([realDir,num2str(make_id),'/',num2str(model_id)]);
        year = year(3:end);
        %no need to randperm
        for k=1:length(year)
            list = dir([realDir,num2str(make_id),'/',num2str(model_id),...
                '/',year(k).name,'/*.jpg']);
            for j=1:length(list)
%                 c=c+1;
                label = textread([labDir,num2str(make_id),'/',...
                    num2str(model_id),'/',year(k).name,'/',list(j).name(1:end-4),...
                    '.txt'],'%d');
                
%                 im =imread([realDir,num2str(make_id),'/',...
%                 num2str(model_id),'/',year(k).name,'/',list(j).name]);
%                 bbox = label(3:6);
%                 bbox_pad = gen_bbox_pad(im,bbox,pad,0);
%                 im_crop = im(bbox_pad(2):bbox_pad(4),bbox_pad(1):bbox_pad(3),:);
%                     imshow(im_crop);pause;
%                 if ~exist([savDir,'real/',num2str(model_id)],'dir');
%                     mkdir([savDir,'real/',num2str(model_id)]);
%                 end
                im_path = [savDir,'real/',num2str(model_id),'/',list(j).name];
                im_path_mirror = [savDir,'real/',num2str(model_id),'/',list(j).name(1:end-4),...
                    '_m.jpg'];
%                 imwrite(im_crop,im_path);
%                 imwrite(im_crop(:,end:-1:1,:),im_path_mirror);
                %original image
                fprintf(f_train,'%s %d %d ',im_path,0,cls_id);                
                fprintf(f_val,'%s %d %d ',im_path,0,cls_id);
                
                fprintf(f_train,'-1 -1 -1 -1\n');
                fprintf(f_val,'-1 -1 -1 -1\n');
                %mirrored image
                fprintf(f_train,'%s %d %d ',im_path_mirror,0,cls_id);  
                fprintf(f_val,'%s %d %d ',im_path_mirror,0,cls_id);
                
                fprintf(f_train,'-1 -1 -1 -1\n');
                fprintf(f_val,'-1 -1 -1 -1\n');
            end
        end
        pose = dlmread([synDir,model_3d_param{i,1},'/',model_3d_param{i,1},...
                '_pose_info.txt']);
            
        for im_id = 1:syn_im
            im_path = [synDir,model_3d_param{i,1},'/',model_3d_param{i,1},...
                '_',num2str(im_id-1),'.png'];
%             im_syn = imread(im_path);
%             bbox = bbox_syn(i,im_id,:);
%             bbox_pad = gen_bbox_pad(im_syn,bbox,pad,0);
%             im_crop = im_syn(bbox_pad(2):bbox_pad(4),bbox_pad(1):bbox_pad(3),:);
%             im_crop = double(im_crop)/255 - 0.57;
%             if size(im_crop,3)==1
%                 im_crop = repmat(im_crop,[1,1,3]);
%             end
%             imshow(im_crop);pause;
%             if ~exist([savDir,'syn/',num2str(model_id)],'dir');
%                 mkdir([savDir,'syn/',num2str(model_id)]);
%             end
            im_path = [savDir,'syn/',num2str(model_id),'/',num2str(im_id-1),'.jpg'];
%             imwrite(im_crop,im_path);
%             view_h = floor(pose(im_id,2)/h_space);
%             view_v = floor(pose(im_id,3)/v_space);
%             view_r = floor((pose(im_id,4)+r_max)/r_space);
            view_r = mod(im_id-1,r_n);
            remainder = floor((im_id-1)/r_n);
            view_h = mod(remainder,h_n);
            view_v = floor(remainder / h_n);
            focal = pose(im_id,5);
%             view_d = view_d_proj(view_h+1); 
            fprintf(f_train,'%s %d %d ',im_path,1,cls_id);
            
            fprintf(f_train,'%d %d %d %.3f\n',view_h, view_v, view_r,log(focal));
            %fprintf(f_syn,'%s %d %d %d %d %d\n',im_path,1,cls_id,view_h,view_v,view_d);
        
        end
    end
end
% fclose(f_syn);
fclose(f_train);
fclose(f_val);
save train_shape_param train_shape_param
% fclose(f_test);