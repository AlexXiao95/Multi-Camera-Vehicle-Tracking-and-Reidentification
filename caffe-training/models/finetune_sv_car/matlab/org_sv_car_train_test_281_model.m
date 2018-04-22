clear,close all,clc
datDir='/home/ljyang/work/data/models_org_281/resize2';

%%
% no longer use crop testing
%cropDir='/home/ljyang/work/data/models_org_274/test_crop';

rng('default');%rand seed
n=281; %model number
train_r = 0.7; %training data ratio
test_r = 1 - train_r;
im_n = zeros(n,1);
test_n = zeros(n,1);
train_filename=['train_sv_car_',num2str(n),'_model'];
train_list_filename=['train_surveillance.txt'];
test_filename=['test_sv_car_',num2str(n),'_model'];
test_list_filename=['test_surveillance.txt'];
% train_filename=['train_sv_car_',num2str(n)];
% train_nodup_filename=['train_sv_car_',num2str(n),'_nodup'];
% test_filename=['test_sv_car_',num2str(n)];
full_l=256;
min_s = 80;

f_train = fopen([train_filename],'w');
f_test = fopen([test_filename],'w');
f_train_list = fopen([train_list_filename],'w');
f_test_list = fopen([test_list_filename],'w');
for i=1:n
    
        im_list = dir([datDir,'/',num2str(i),'/*.jpg']);
        im_n(i) = length(im_list);
       
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
            fprintf(f_train,' %d\n',i-1);%class label, color label
           
           end
           fprintf(f_train_list,[num2str(i),'/',im_list(p(k)).name,'\n']);
        end


%%%%%%%%%%%%%%%%%%%%%%%%%% traditional testing %%%%%%%%%%%%%%%%%%%
%
%
        for k=train_n+1:im_n(i)
            fprintf(f_test, [datDir,'/',num2str(i),'/',im_list(p(k)).name]);
            fprintf(f_test,' %d\n',i-1);
            fprintf(f_test_list,[num2str(i),'/',im_list(p(k)).name,'\n']);
        end
   
end

fclose(f_train);
fclose(f_test);
fclose(f_train_list);
fclose(f_test_list);
save test_n test_n
