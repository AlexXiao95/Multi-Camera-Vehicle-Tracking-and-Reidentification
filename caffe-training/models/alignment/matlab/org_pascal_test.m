clear, close all
VOCinit;
[gtids,t]=textread(sprintf(VOCopts.clsimgsetpath,'car',VOCopts.testset),'%s %d');
len =length(gtids);
%%
imDir ='/home/ljyang/work/alignment/PASCAL3D+_release1.0/Images/car_pascal/';
labDir = '/home/ljyang/work/alignment/PASCAL3D+_release1.0/Annotations/car_pascal/';
% lab_list = dir([labDir,'*.mat']);
% len = length(lab_list);
% synDir ='/home/ljyang/work/alignment/3d_models/car_models/obj_merge/';
savDir ='/home/ljyang/work/data/test_pascal/';
% train_filename = ['../lists/transfer_train'];
test_filename = ['../lists/transfer_test_pascal'];
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
for i=1:len
    labPath = [labDir,gtids{i},'.mat'];
    if exist(labPath,'file')
        
    anno = load(labPath,'record');
    anno = anno.record;
    p = [gtids{i},'.jpg'];       

%     im =imread([imDir,p]);
    carIdxSet = find(ismember({anno.objects(:).class},'car'));
    for carIdx = carIdxSet
        if anno.objects(carIdx).viewpoint.distance==0
            continue;
        end
        c=c+1;
        anno_cur = anno.objects(carIdx);
        bbox = anno_cur.bbox;
        gt_pose(c,1) = anno_cur.viewpoint.azimuth;
        gt_pose(c,2) = anno_cur.viewpoint.elevation;
        gt_pose(c,3) = anno_cur.viewpoint.theta;
%         bbox_pad = gen_bbox_pad(im,bbox,pad,0);
%         im_crop = im(bbox_pad(2):bbox_pad(4),bbox_pad(1):bbox_pad(3),:);
%                     imshow(im_crop);pause;
%         im_crop = bbox_pad_crop(im,bbox,pad);
%         pos = strfind(p,'/');
%         im_fd = p(1:pos(3));
%         if ~exist([savDir,im_fd],'dir')
%             mkdir([savDir,im_fd]);
%         end
        im_path = [savDir,gtids{i},'_',num2str(carIdx),'.jpg'];
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
    end
%         fprintf(f_test,'%f\n',test_images(i).focal);
        %mirrored image
                
    end
        
    
end
fclose(f_test);
save gt_pose_pascal gt_pose
% fclose(f_test);