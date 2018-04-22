clear, close all


imDir ='/home/ljyang/work/alignment/FG3DCar/dataset/test/';
origDir = '/home/ljyang/work/alignment/FG3DCar/dataset/original/';
% load ground truth
load('/home/ljyang/work/alignment/FG3DCar/3dmodel/ground_truth','manualParam');

% get image list
model_list = dir(imDir);
model_list = model_list(3:end);
images=[];
for i=1:length(model_list)
    sub_list = dir([imDir,model_list(i).name,'/*.jpg']);    
    images = [images;sub_list];
end
len = length(images);
% lab_list = dir([labDir,'*.mat']);
% len = length(lab_list);
% synDir ='/home/ljyang/work/alignment/3d_models/car_models/obj_merge/';
savDir ='/home/ljyang/work/data/test_fg3d/';
if ~exist([savDir],'dir')
    mkdir([savDir]);
end
% train_filename = ['../lists/transfer_train'];
test_filename = ['../lists/transfer_test_fg3d'];
if ~exist(savDir,'dir')
    mkdir(savDir);
end
% val_filename= ['../lists/transfer_val'];
% f_train = fopen(train_filename,'w');
f_test = fopen(test_filename,'w');
% f_val = fopen(val_filename,'w');
pad = 0.07;
param_dim = 10;
h_space = 6;
v_space = 6;
r_space = 6;
% view_d_proj = compute_proj(h_space,v_space,v_max,r_space,r_max);
c=0;
% mp = cell2mat(manualParam);
% annos = manualParam{:}.filename;
for i=1:len
%     labPath = [labDir,gtids{i},'.mat'];
    p = images(i).name;       
    pos = find(cellfun(@(x)(strcmp(p,x.filename)),manualParam),1,'first');
    im =imread([origDir,p]);
    
        bbox = manualParam{pos}.box;        
        bbox = round([bbox.x,bbox.y,bbox.w+bbox.x,bbox.h+bbox.y]);
      
        bbox_pad = gen_bbox_pad(im,bbox,pad,0);
        im_crop = im(bbox_pad(2):bbox_pad(4),bbox_pad(1):bbox_pad(3),:);
%                     imshow(im_crop);pause;
%         im_crop = bbox_pad_crop(im,bbox,pad);
%         pos = strfind(p,'/');
%         im_fd = p(1:pos(3));
%         
        im_path = [savDir,p];
        %im_path_mirror = [savDir,p(1:end-4),...
         %   '_m.jpg'];
%         imwrite(im_crop,im_path);
        %imwrite(im_crop(:,end:-1:1,:),im_path_mirror);
        %original image
        fprintf(f_test,'%s %d ',im_path,0);  
%         fprintf(f_test,'%d ',0);%dummy class label
        for d = 1:param_dim
            fprintf(f_test,'%.3f ',0);
        end
  
        fprintf(f_test,'-1 -1 -1 0\n');
    
%         fprintf(f_test,'%f\n',test_images(i).focal);
        %mirrored image
                
    
        
    
end
fclose(f_test);
