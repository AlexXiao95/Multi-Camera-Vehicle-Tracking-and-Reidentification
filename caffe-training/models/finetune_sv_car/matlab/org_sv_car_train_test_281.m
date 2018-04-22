clear,close all,clc
datDir='/home/ljyang/work/data/models_org_281/resize';
% color label meaning:
% -1: invalid
% 0~9: valid colors
color_list = load('/home/ljyang/work/data/models_org_281/color_list_org','color_list_org');
color_list = color_list.color_list_org;
%%
% no longer use crop testing
%cropDir='/home/ljyang/work/data/models_org_274/test_crop';

rng('default');%rand seed
n=281; %model number
train_r = 0.7; %training data ratio
test_r = 1 - train_r;
im_n = zeros(n,1);
test_n = zeros(n,1);
train_filename=['train_sv_car_',num2str(n)];
train_nodup_filename=['train_sv_car_',num2str(n),'_nodup'];
test_filename=['test_sv_car_',num2str(n)];
% train_filename=['train_sv_car_',num2str(n)];
% train_nodup_filename=['train_sv_car_',num2str(n),'_nodup'];
% test_filename=['test_sv_car_',num2str(n)];
full_l=256;
min_s = 80;

f_train = fopen([train_filename],'w');
f_test = fopen([test_filename],'w');
f_train_nodup = fopen([train_nodup_filename],'w');

for i=1:n
    
        im_list = dir([datDir,'/',num2str(i),'/*.jpg']);
        im_n(i) = length(im_list);
        %fetch color label
        color = zeros(im_n(i),1);
        for j=1:im_n(i)
            im_id = find(cellfun(@(x)(strcmp(x,im_list(j).name)),color_list(:,1)),1,'first');
            if isempty(im_id)
                error(['image not found: ',im_list(j).name]);
            end
            color(j) = color_list{im_id,2};
        end
        train_n = round(im_n(i)*train_r);
        test_n(i) = im_n(i) - train_n;
        p = randperm(im_n(i));%randomize
%%%%%%%%%%%%%%%%%%%%%%%%%% training %%%%%%%%%%%%%%%%%%%
        %duplicate images to make each model
        % has minimum min_s samples
        dup_times = max(1,round(min_s/train_n));
       
        for k=1:train_n
           for dup=1:dup_times
            fprintf(f_train,[datDir,'/',num2str(i),'/',im_list(p(k)).name]);
            fprintf(f_train,' %d %d\n',i-1, color(p(k)));%class label, color label
           
           end
           fprintf(f_train_nodup,[datDir,'/',num2str(i),'/',im_list(p(k)).name]);
           fprintf(f_train_nodup,' %d %d\n',i-1, color(p(k)));
        end


%%%%%%%%%%%%%%%%%%%%%%%%%% traditional testing %%%%%%%%%%%%%%%%%%%
%
%
        for k=train_n+1:im_n(i)
            fprintf(f_test, [datDir,'/',num2str(i),'/',im_list(p(k)).name]);
            fprintf(f_test,' %d %d\n',i-1, color(p(k)));
        end
   
end

fclose(f_train);
fclose(f_test);
fclose(f_train_nodup);
save test_n test_n
